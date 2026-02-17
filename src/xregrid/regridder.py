from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

from xregrid.utils import update_history, get_crs_info
from xregrid.grid import (
    _get_mesh_info,
    _bounds_to_vertices,
    _get_grid_bounds,
    _create_esmf_grid,
)
from xregrid.core import _apply_weights_core, _setup_worker_cache
from xregrid.parallel import (
    _assemble_weights_task,
    _get_weights_sum_task,
    _get_nnz_task,
    _compute_chunk_weights,
    _sync_cache_from_worker_data,
)

if TYPE_CHECKING:
    import dask.distributed
    from scipy.sparse import csr_matrix

# Global cache for the driver to store distributed futures
# Keyed by (client_id, weight_key)
_DRIVER_CACHE: dict = {}


class Regridder:
    """
    Optimized ESMF-based regridder for xarray DataArrays and Datasets.

    This regridder supports both eager (NumPy) and lazy (Dask) backends.
    It uses ESMPy to generate weights and applies them using xarray.apply_ufunc.

    Attributes
    ----------
    source_grid_ds : xr.Dataset
        The source grid dataset containing 'lat' and 'lon'.
    target_grid_ds : xr.Dataset
        The target grid dataset containing 'lat' and 'lon'.
    method : str
        The regridding method (e.g., 'bilinear', 'conservative').
    mask_var : str, optional
        The variable name in source_grid_ds to use as a mask.
    filename : str
        The path to save/load weights.
    skipna : bool
        Whether to handle NaNs by re-normalizing weights.
    na_thres : float
        Threshold for NaN handling.
    periodic : bool
        Whether the grid is periodic in longitude.
    """

    # Internal state default values (Aero Protocol: Robustness)
    source_grid_ds: Optional[xr.Dataset] = None
    target_grid_ds: Optional[xr.Dataset] = None
    method: str = "bilinear"
    mask_var: Optional[str] = None
    filename: str = "weights.nc"
    skipna: bool = False
    na_thres: float = 1.0
    periodic: bool = False
    provenance: list[str] = []

    _shape_source: Optional[Tuple[int, ...]] = None
    _shape_target: Optional[Tuple[int, ...]] = None
    _dims_source: Optional[Tuple[str, ...]] = None
    _dims_target: Optional[Tuple[str, ...]] = None
    _is_unstructured_src: bool = False
    _is_unstructured_tgt: bool = False
    _total_weights: Optional[Union[np.ndarray, dask.distributed.Future]] = None
    _weights_matrix: Optional[Union[csr_matrix, dask.distributed.Future]] = None
    _dask_client: Optional[dask.distributed.Client] = None
    _dask_futures: Optional[list[dask.distributed.Future]] = None

    def __init__(
        self,
        source_grid_ds: xr.Dataset,
        target_grid_ds: xr.Dataset,
        method: str = "bilinear",
        mask_var: Optional[str] = None,
        reuse_weights: bool = False,
        filename: str = "weights.nc",
        skipna: bool = False,
        na_thres: float = 1.0,
        periodic: bool = False,
        mpi: bool = False,
        parallel: bool = False,
        compute: bool = True,
        extrap_method: Optional[str] = None,
        extrap_dist_exponent: float = 2.0,
    ) -> None:
        """
        Initialize the Regridder.

        Parameters
        ----------
        source_grid_ds : xr.Dataset
            Contain 'lat' and 'lon'.
        target_grid_ds : xr.Dataset
            Contain 'lat' and 'lon'.
        method : str, default 'bilinear'
            Regridding method (bilinear, conservative, nearest_s2d, nearest_d2s, patch).
        mask_var : str, optional
            Variable name for mask (1=valid, 0=masked).
        reuse_weights : bool, default False
            Load weights from filename if it exists.
        filename : str, default 'weights.nc'
            Path to weights file.
        skipna : bool, default False
            Handle NaNs in input data by re-normalizing weights.
        na_thres : float, default 1.0
            Threshold for NaN handling.
        periodic : bool, default False
            Whether the grid is periodic in longitude.
        mpi : bool, default False
            Whether to use MPI for parallel weight generation.
            Requires running with mpirun and having mpi4py installed for gathering.
        parallel : bool, default False
            Whether to use Dask for parallel weight generation.
            Requires 'dask' and 'distributed' installed.
            Cannot be True if mpi=True.
        compute : bool, default True
            If True, compute weights immediately when parallel=True.
            If False, submitting tasks but delaying gathering until .compute() is called.
            Only relevant if parallel=True.
        extrap_method : str, optional
            Extrapolation method (nearest_s2d, nearest_idw, creep_fill).
        extrap_dist_exponent : float, default 2.0
            Exponent for IDW extrapolation.
        """
        if mpi and parallel:
            raise ValueError(
                "Cannot use both MPI and Dask (parallel=True) simultaneously."
            )

        if parallel:
            import importlib.util

            if importlib.util.find_spec("dask.distributed") is None:
                raise ImportError(
                    "Dask distributed is required for parallel=True. "
                    "Please install it via `pip install dask distributed`."
                )

        # Initialize ESMF Manager (required for some environments)
        try:
            import esmpy

            if mpi:
                # Use MULTI logkind for MPI parallelization (Aero Protocol)
                # Some versions of esmpy don't support logkind in Manager constructor
                try:
                    self._manager = esmpy.Manager(
                        logkind=esmpy.LogKind.MULTI, debug=False
                    )
                except TypeError:
                    self._manager = esmpy.Manager(debug=False)
            else:
                self._manager = esmpy.Manager(debug=False)
        except ImportError:
            self._manager = None

        self.source_grid_ds = source_grid_ds
        self.target_grid_ds = target_grid_ds
        self.method = method
        self.mask_var = mask_var
        self.filename = filename
        self.skipna = skipna
        self.na_thres = na_thres
        self.periodic = periodic
        self.parallel = parallel
        self.compute_on_init = compute
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent

        # Determine coordinate system for consistency (Aero Protocol: Robustness)
        try:
            import esmpy

            self._coord_sys = (
                esmpy.CoordSys.SPH_DEG if periodic else esmpy.CoordSys.CART
            )
        except ImportError:
            self._coord_sys = None

        # Robust coordinate handling: internally sort coordinates to be ascending
        # to ensure ESMF weight generation is stable and avoid boundary issues.
        # (Aero Protocol: User doesn't have to worry about monotonicity)
        self.source_grid_ds, self._src_was_sorted = self._normalize_grid(source_grid_ds)
        self.target_grid_ds, self._tgt_was_sorted = self._normalize_grid(target_grid_ds)

        try:
            import esmpy

            self.method_map = {
                "bilinear": esmpy.RegridMethod.BILINEAR,
                "conservative": esmpy.RegridMethod.CONSERVE,
                "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
                "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
                "patch": esmpy.RegridMethod.PATCH,
            }

            self.extrap_method_map = {
                "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
                "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
                "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
            }
        except ImportError:
            self.method_map = {}
            self.extrap_method_map = {}

        # Internal state
        self._shape_source: Optional[Tuple[int, ...]] = None
        self._shape_target: Optional[Tuple[int, ...]] = None
        self._dims_source: Optional[Tuple[str, ...]] = None
        self._dims_target: Optional[Tuple[str, ...]] = None
        self._is_unstructured_src: bool = False
        self._is_unstructured_tgt: bool = False
        self._total_weights: Optional[np.ndarray] = None
        self._weights_matrix: Optional[Any] = None
        self._loaded_method: Optional[str] = None
        self._loaded_periodic: Optional[bool] = None
        self._loaded_extrap: Optional[str] = None
        self.generation_time: Optional[float] = None
        self._dask_futures: Optional[list] = None
        self._dask_client: Optional[Any] = None
        self._dask_start_time: Optional[float] = None
        self.provenance: list[str] = []

        if reuse_weights and os.path.exists(filename):
            self._load_weights()
            # Validate loaded weights against provided grids and parameters
            self._validate_weights()
        else:
            self._generate_weights()
            if reuse_weights:
                self._save_weights()

    @classmethod
    def from_weights(
        cls: Type["Regridder"],
        filename: str,
        source_grid_ds: xr.Dataset,
        target_grid_ds: xr.Dataset,
        **kwargs: Any,
    ) -> "Regridder":
        """
        Create a Regridder from a pre-computed weights file.

        Parameters
        ----------
        filename : str
            Path to the weights file.
        source_grid_ds : xr.Dataset
            The source grid dataset.
        target_grid_ds : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Additional arguments passed to the Regridder constructor.
            These will be validated against the weights file.

        Returns
        -------
        Regridder
            The initialized Regridder instance.
        """
        return cls(
            source_grid_ds,
            target_grid_ds,
            filename=filename,
            reuse_weights=True,
            **kwargs,
        )

    def _normalize_grid(self, ds: xr.Dataset) -> Tuple[xr.Dataset, bool]:
        """
        Normalize coordinate names and ensure they are in a predictable order.

        Parameters
        ----------
        ds : xr.Dataset
            The grid dataset to normalize.

        Returns
        -------
        ds : xr.Dataset
            The normalized dataset.
        was_sorted : bool
            Whether the dataset was sorted during normalization.
        """
        was_sorted = False
        try:
            # Only for rectilinear 1D coordinates
            lat_da = ds.cf["latitude"]
            lon_da = ds.cf["longitude"]

            # Must be 1D and not shared (unstructured grids share dimensions)
            if (
                lat_da.ndim == 1
                and lon_da.ndim == 1
                and lat_da.dims[0] != lon_da.dims[0]
            ):
                lat_dim = lat_da.dims[0]
                lon_dim = lon_da.dims[0]

                # Only sort if dimension coordinates are numeric (Aero Protocol: Robustness)
                if np.issubdtype(ds[lat_dim].dtype, np.number) and np.issubdtype(
                    ds[lon_dim].dtype, np.number
                ):
                    # Aero Protocol: Use indexes for monotonicity check to remain lazy.
                    # Indexes are always in memory in xarray, so this doesn't trigger
                    # computation of dask-backed coordinates.
                    is_lat_asc = ds.indexes[lat_dim].is_monotonic_increasing
                    is_lon_asc = ds.indexes[lon_dim].is_monotonic_increasing

                    if not (is_lat_asc and is_lon_asc):
                        ds = ds.sortby([lat_dim, lon_dim])
                        was_sorted = True
        except (KeyError, AttributeError, ValueError):
            pass
        return ds, was_sorted

    def _validate_weights(self) -> None:
        """
        Validate loaded weights against the provided source and target grids.

        Ensures that shapes, dimension names, regridding method, and periodicity
        match the requested configuration to maintain scientific integrity.

        Raises
        ------
        ValueError
            If the loaded weights do not match the current regridding configuration.
        """
        # Get current grid info
        _, _, src_shape, src_dims, _ = self._get_mesh_info(self.source_grid_ds)
        _, _, dst_shape, dst_dims, _ = self._get_mesh_info(self.target_grid_ds)

        if src_shape != self._shape_source:
            raise ValueError(
                f"Source grid shape {src_shape} does not match "
                f"loaded weights source shape {self._shape_source}"
            )
        if dst_shape != self._shape_target:
            raise ValueError(
                f"Target grid shape {dst_shape} does not match "
                f"loaded weights target shape {self._shape_target}"
            )

        # Check regridding parameters
        if self._loaded_method is not None and self._loaded_method != self.method:
            raise ValueError(
                f"Requested method '{self.method}' does not match "
                f"loaded weights method '{self._loaded_method}'"
            )

        if self._loaded_periodic is not None and self._loaded_periodic != self.periodic:
            raise ValueError(
                f"Requested periodic={self.periodic} does not match "
                f"loaded weights periodic={self._loaded_periodic}"
            )

        if self._loaded_extrap is not None:
            current_extrap = self.extrap_method or "none"
            if current_extrap != self._loaded_extrap:
                raise ValueError(
                    f"Requested extrap_method='{current_extrap}' does not match "
                    f"loaded weights extrap_method='{self._loaded_extrap}'"
                )

        if hasattr(self, "_loaded_skipna") and self._loaded_skipna is not None:
            if self._loaded_skipna != self.skipna:
                raise ValueError(
                    f"Requested skipna={self.skipna} does not match "
                    f"loaded weights skipna={self._loaded_skipna}"
                )

        if hasattr(self, "_loaded_na_thres") and self._loaded_na_thres is not None:
            if abs(self._loaded_na_thres - self.na_thres) > 1e-6:
                raise ValueError(
                    f"Requested na_thres={self.na_thres} does not match "
                    f"loaded weights na_thres={self._loaded_na_thres}"
                )

    def _get_mesh_info(
        self, ds: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
        """
        Instance-level wrapper for _get_mesh_info.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to inspect.

        Returns
        -------
        lat : xr.DataArray
            Latitude coordinate.
        lon : xr.DataArray
            Longitude coordinate.
        shape : tuple of int
            Grid shape.
        dims : tuple of str
            Spatial dimension names.
        is_unstructured : bool
            True if the grid is unstructured.
        """
        return _get_mesh_info(ds)

    def _bounds_to_vertices(self, b: xr.DataArray) -> Union[xr.DataArray, np.ndarray]:
        """
        Instance-level wrapper for _bounds_to_vertices.

        Parameters
        ----------
        b : xr.DataArray
            The coordinate bounds.

        Returns
        -------
        Union[xr.DataArray, np.ndarray]
            The vertex coordinates.
        """
        return _bounds_to_vertices(b)

    def _get_grid_bounds(
        self, ds: xr.Dataset
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Instance-level wrapper for _get_grid_bounds.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to inspect.

        Returns
        -------
        lat_b : np.ndarray, optional
            Latitude bounds.
        lon_b : np.ndarray, optional
            Longitude bounds.
        """
        return _get_grid_bounds(ds)

    def _create_esmf_object(
        self, ds: xr.Dataset, is_source: bool = True
    ) -> Tuple[Any, list[str], Optional[np.ndarray]]:
        """
        Creates an ESMF Grid or LocStream and updates internal metadata.

        Parameters
        ----------
        ds : xr.Dataset
            The grid dataset.
        is_source : bool, default True
            Whether this is the source grid or target grid.

        Returns
        -------
        grid : esmpy.Grid or esmpy.LocStream
            The created ESMF object.
        provenance : list of str
            Provenance messages from grid creation.
        """
        lon, lat, shape, dims, is_unstructured = self._get_mesh_info(ds)

        if is_source:
            self._shape_source = shape
            self._dims_source = dims
            self._is_unstructured_src = is_unstructured
        else:
            self._shape_target = shape
            self._dims_target = dims
            self._is_unstructured_tgt = is_unstructured

        return _create_esmf_grid(
            ds,
            self.method,
            periodic=self.periodic if is_source else False,
            mask_var=self.mask_var if is_source else None,
            coord_sys=self._coord_sys,
        )

    def _generate_weights(self) -> None:
        """
        Generate regridding weights using ESMPy.

        This is the core weight generation method for serial or MPI-based execution.
        """
        import esmpy

        if self.parallel:
            self._generate_weights_dask(compute=self.compute_on_init)
            return

        start_time = time.perf_counter()
        src_obj, src_prov, src_orig_idx = self._create_esmf_object(
            self.source_grid_ds, is_source=True
        )
        dst_obj, dst_prov, dst_orig_idx = self._create_esmf_object(
            self.target_grid_ds, is_source=False
        )
        self.provenance.extend(src_prov)
        self.provenance.extend(dst_prov)

        if isinstance(src_obj, esmpy.Mesh):
            meshloc = (
                esmpy.MeshLoc.ELEMENT
                if self.method == "conservative"
                else esmpy.MeshLoc.NODE
            )
            src_field = esmpy.Field(src_obj, name="src", meshloc=meshloc)
        else:
            src_field = esmpy.Field(src_obj, name="src")

        if isinstance(dst_obj, esmpy.Mesh):
            meshloc = (
                esmpy.MeshLoc.ELEMENT
                if self.method == "conservative"
                else esmpy.MeshLoc.NODE
            )
            dst_field = esmpy.Field(dst_obj, name="dst", meshloc=meshloc)
        else:
            dst_field = esmpy.Field(dst_obj, name="dst")

        try:
            regrid_method = self.method_map[self.method]
        except KeyError:
            available_methods = ", ".join(self.method_map.keys())
            raise ValueError(
                f"Method '{self.method}' is not supported. "
                f"Available methods are: {available_methods}"
            )

        regrid_kwargs = {
            "regrid_method": regrid_method,
            "unmapped_action": esmpy.UnmappedAction.IGNORE,
            "factors": True,
        }

        if self.extrap_method:
            regrid_kwargs["extrap_method"] = self.extrap_method_map[self.extrap_method]
            regrid_kwargs["extrap_dist_exponent"] = self.extrap_dist_exponent

        if self.mask_var and self.mask_var in self.source_grid_ds:
            regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

        if self.method == "conservative":
            regrid_kwargs["norm_type"] = esmpy.NormType.FRACAREA

        # Build Regrid object
        regrid = esmpy.Regrid(src_field, dst_field, **regrid_kwargs)

        # Explicit check for overlaps
        fl, fil = regrid.get_factors()
        if fl is None or fil is None:
            raise RuntimeError(
                "ESMPy failed to find any overlaps between source and target grids. "
                "Check that coordinates are correct and ranges overlap. "
                "For global grids, ensure 'periodic=True' is set."
            )

        weights = regrid.get_weights_dict(deep_copy=True)

        # Map to original indices if Mesh elements were triangulated
        row_dst = weights["row_dst"] - 1
        col_src = weights["col_src"] - 1

        if dst_orig_idx is not None and self.method == "conservative":
            row_dst = dst_orig_idx[row_dst]
        if src_orig_idx is not None and self.method == "conservative":
            col_src = src_orig_idx[col_src]

        # Handle MPI gathering if multiple ranks are present
        pet_count = esmpy.pet_count()
        local_pet = esmpy.local_pet()

        if pet_count > 1:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                # Gather all weights to rank 0
                all_weights = comm.gather(weights, root=0)

                if local_pet == 0:
                    # Concatenate all gathered weights
                    rows = []
                    cols = []
                    data = []
                    for i, w in enumerate(all_weights):
                        r = w["row_dst"] - 1
                        c = w["col_src"] - 1
                        # Note: orig_idx is not easily gathered via ESMF weights dict
                        # but here we are in serial-equivalent rank 0 gathering.
                        # Actually, each rank might have different triangulation?
                        # No, triangulation should be deterministic.
                        if dst_orig_idx is not None:
                            r = dst_orig_idx[r]
                        if src_orig_idx is not None:
                            c = src_orig_idx[c]
                        rows.append(r)
                        cols.append(c)
                        data.append(w["weights"])

                    rows = np.concatenate(rows)
                    cols = np.concatenate(cols)
                    data = np.concatenate(data)
                else:
                    # Non-root ranks will have empty weight arrays to avoid duplicate computation.
                    # Only the root rank builds the full sparse matrix for Dask-based application.
                    rows = np.array([], dtype=np.int32)
                    cols = np.array([], dtype=np.int32)
                    data = np.array([], dtype=np.float64)
            except ImportError:
                import warnings

                warnings.warn(
                    "Multiple ESMF PETs detected but mpi4py is not installed. "
                    "Weights will not be gathered to rank 0, which may lead to incorrect results "
                    "if the application is not also parallelized via MPI."
                )
                rows = weights["row_dst"] - 1
                cols = weights["col_src"] - 1
                data = weights["weights"]
        else:
            rows = row_dst
            cols = col_src
            data = weights["weights"]

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        self._weights_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            # Optimization: Use sum(axis=1) instead of memory-intensive ones multiplication
            self._total_weights = np.array(self._weights_matrix.sum(axis=1)).flatten()

        self.generation_time = time.perf_counter() - start_time

    def _generate_weights_dask(self, compute: bool = True) -> None:
        """
        Generate regridding weights using Dask parallel workers.

        Splits the target grid into chunks and distributes weight generation tasks
        across a Dask cluster.

        Parameters
        ----------
        compute : bool, default True
            Whether to immediately trigger computation and gather weights.
        """
        import dask.distributed

        self._dask_start_time = time.perf_counter()

        # Get grid info and populate internal state
        # Source
        _, _, src_shape, src_dims, is_unstructured_src = self._get_mesh_info(
            self.source_grid_ds
        )
        self._shape_source = src_shape
        self._dims_source = src_dims
        self._is_unstructured_src = is_unstructured_src

        # Target
        _, _, dst_shape, dst_dims, is_unstructured_dst = self._get_mesh_info(
            self.target_grid_ds
        )
        self._shape_target = dst_shape
        self._dims_target = dst_dims
        self._is_unstructured_tgt = is_unstructured_dst

        # Get client
        try:
            client = dask.distributed.get_client()
        except ValueError:
            # Create a local cluster if none exists
            cluster = dask.distributed.LocalCluster()
            client = dask.distributed.Client(cluster)

        self._dask_client = client

        # Determine number of chunks. Use number of workers * 2 usually good heuristic
        n_workers = len(client.scheduler_info()["workers"])
        n_chunks_total = max(1, n_workers * 2)

        futures = []

        # Scatter source grid info (minimal)
        source_grid_only = self.source_grid_ds.copy(deep=False)
        for var in list(source_grid_only.data_vars):
            if var != self.mask_var:
                source_grid_only = source_grid_only.drop_vars(var)
        src_ds_future = client.scatter(source_grid_only, broadcast=True)

        if is_unstructured_dst:
            # Unstructured target: Split along the single dimension
            dim0 = dst_dims[0]
            size0 = self.target_grid_ds.sizes[dim0]
            n_chunks = min(n_chunks_total, size0)
            indices = np.array_split(np.arange(size0), n_chunks)

            for idx in indices:
                if len(idx) == 0:
                    continue
                i_start, i_end = idx[0], idx[-1] + 1
                chunk_ds = self.target_grid_ds.isel({dim0: slice(i_start, i_end)})
                # For unstructured, we only need start and end indices
                dest_slice_info = (i_start, i_end, 0, 0, 0)

                future = client.submit(
                    _compute_chunk_weights,
                    src_ds_future,
                    chunk_ds,
                    self.method,
                    dest_slice_info,
                    self.extrap_method,
                    self.extrap_dist_exponent,
                    self.mask_var,
                    self.periodic,
                    coord_sys=self._coord_sys,
                )
                futures.append(future)
        else:
            # Structured target: 2D split
            dim0 = dst_dims[0]
            dim1 = dst_dims[1] if len(dst_dims) > 1 else None

            size0 = self.target_grid_ds.sizes[dim0]
            size1 = self.target_grid_ds.sizes[dim1] if dim1 else 1

            if dim1 and n_chunks_total > 1:
                # Try to make chunks approximately square-ish for 2D grids
                n0 = int(np.sqrt(n_chunks_total * size0 / size1))
                n0 = max(1, min(n0, size0))
                n1 = max(1, n_chunks_total // n0)
                n1 = max(1, min(n1, size1))
            else:
                n0 = min(n_chunks_total, size0)
                n1 = 1

            # Split indices
            indices0 = np.array_split(np.arange(size0), n0)
            indices1 = np.array_split(np.arange(size1), n1)

            for idx0 in indices0:
                if len(idx0) == 0:
                    continue
                for idx1 in indices1:
                    if len(idx1) == 0:
                        continue

                    # Slice target grid
                    i0_start, i0_end = idx0[0], idx0[-1] + 1
                    i1_start, i1_end = (idx1[0], idx1[-1] + 1) if dim1 else (0, 1)

                    sel_dict = {dim0: slice(i0_start, i0_end)}
                    if dim1:
                        sel_dict[dim1] = slice(i1_start, i1_end)

                    chunk_ds = self.target_grid_ds.isel(sel_dict)

                    # Pass slice info instead of massive array to workers (Aero Protocol: Driver Efficiency)
                    dest_slice_info = (i0_start, i0_end, i1_start, i1_end, size1)

                    future = client.submit(
                        _compute_chunk_weights,
                        src_ds_future,
                        chunk_ds,
                        self.method,
                        dest_slice_info,
                        self.extrap_method,
                        self.extrap_dist_exponent,
                        self.mask_var,
                        self.periodic,
                        coord_sys=self._coord_sys,
                    )
                    futures.append(future)

        self._dask_futures = futures

        if compute:
            self.compute()

    def persist(self) -> "Regridder":
        """
        Ensure tasks are submitted to the cluster.

        Since this implementation uses eager task submission (Futures),
        the tasks are already running or pending on the cluster.
        This method returns self for API consistency with Dask.

        Returns
        -------
        Regridder
            The regridder instance (self).
        """
        if not self.parallel:
            return self

        # If we later switch to dask.delayed, this would trigger client.compute(delayed_objs)
        if self._dask_futures is None and self._weights_matrix is None:
            # This arguably shouldn't happen in current logic unless something failed
            pass

        return self

    def compute(self) -> None:
        """
        Trigger computation of weights if using Dask and not yet computed.

        Performs weight concatenation on the Dask cluster to avoid driver-side OOM.
        The resulting weights are kept as a Future on the cluster until needed.
        """
        if not self.parallel or self._weights_matrix is not None:
            return

        if not self._dask_futures:
            return

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        # Perform concatenation on a worker to protect driver memory (Aero Protocol)
        # We use top-level task functions to avoid capturing 'self' and mocks.
        self._weights_matrix = self._dask_client.submit(
            _assemble_weights_task, self._dask_futures, n_src, n_dst
        )

        if self.skipna:
            # Compute total weights sum on worker too
            self._total_weights = self._dask_client.submit(
                _get_weights_sum_task, self._weights_matrix
            )

        if self._dask_start_time:
            self.generation_time = time.perf_counter() - self._dask_start_time

        # Clear futures to free memory
        self._dask_futures = None

    def _save_weights(self) -> None:
        """
        Save regridding weights and metadata to a NetCDF file.

        Only the root rank (PET 0) performs file I/O.
        """
        try:
            import esmpy

            if esmpy.local_pet() != 0:
                return  # Only rank 0 saves weights
        except ImportError:
            pass

        # Use weights property to ensure they are gathered if remote
        weights_matrix = self.weights

        # Convert to COO to access row and col attributes
        weights_coo = weights_matrix.tocoo()

        ds_weights = xr.Dataset(
            data_vars={
                "row": (["n_s"], weights_coo.row + 1),
                "col": (["n_s"], weights_coo.col + 1),
                "S": (["n_s"], weights_coo.data),
            },
            attrs={
                "n_src": self._weights_matrix.shape[1],
                "n_dst": self._weights_matrix.shape[0],
                "shape_src": list(self._shape_source) if self._shape_source else [],
                "shape_dst": list(self._shape_target) if self._shape_target else [],
                "dims_src": list(self._dims_source) if self._dims_source else [],
                "dims_target": list(self._dims_target) if self._dims_target else [],
                "is_unstructured_src": int(self._is_unstructured_src),
                "is_unstructured_tgt": int(self._is_unstructured_tgt),
                "method": self.method,
                "periodic": int(self.periodic),
                "skipna": int(self.skipna),
                "na_thres": self.na_thres,
                "provenance": "; ".join(self.provenance) if self.provenance else "",
                "extrap_method": self.extrap_method or "none",
                "extrap_dist_exponent": self.extrap_dist_exponent,
                "generation_time": self.generation_time
                if self.generation_time
                else 0.0,
            },
        )
        update_history(ds_weights, "Weights generated by Regridder")
        ds_weights.to_netcdf(self.filename)

    def _load_weights(self) -> None:
        """
        Load regridding weights and metadata from a NetCDF file.
        """
        with xr.open_dataset(self.filename) as ds_weights:
            ds_weights.load()
            rows = ds_weights["row"].values - 1
            cols = ds_weights["col"].values - 1
            data = ds_weights["S"].values
            n_src = ds_weights.attrs["n_src"]
            n_dst = ds_weights.attrs["n_dst"]

            def _to_tuple(attr: Any) -> Tuple[Any, ...]:
                """
                Convert attribute to tuple.

                Parameters
                ----------
                attr : Any
                    The attribute to convert.

                Returns
                -------
                Tuple
                    The converted tuple.
                """
                if isinstance(attr, str):
                    # Handle cases where attributes might be stored as string representations
                    attr = attr.strip("()[]").replace(" ", "").split(",")
                    return tuple(int(x) if x.isdigit() else x for x in attr if x)
                return tuple(attr)

            self._shape_source = _to_tuple(ds_weights.attrs["shape_src"])
            self._shape_target = _to_tuple(ds_weights.attrs["shape_dst"])
            self._dims_source = _to_tuple(ds_weights.attrs["dims_src"])
            self._dims_target = _to_tuple(ds_weights.attrs["dims_target"])
            self._is_unstructured_src = bool(ds_weights.attrs["is_unstructured_src"])
            self._is_unstructured_tgt = bool(ds_weights.attrs["is_unstructured_tgt"])
            self._loaded_periodic = bool(ds_weights.attrs.get("periodic", False))
            self._loaded_method = ds_weights.attrs.get("method")
            self._loaded_extrap = ds_weights.attrs.get("extrap_method", "none")
            self._loaded_skipna = bool(ds_weights.attrs.get("skipna", False))
            self._loaded_na_thres = float(ds_weights.attrs.get("na_thres", 1.0))
            self.generation_time = ds_weights.attrs.get("generation_time")
            loaded_prov = ds_weights.attrs.get("provenance", "")
            if loaded_prov:
                self.provenance = loaded_prov.split("; ")

        self._weights_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            # Optimization: Use sum(axis=1) instead of memory-intensive ones multiplication
            self._total_weights = np.array(self._weights_matrix.sum(axis=1)).flatten()

    @property
    def weights(self) -> csr_matrix:
        """
        Get the sparse weight matrix, gathering it from the cluster if necessary.

        Returns
        -------
        csr_matrix
            The regridding weight matrix.
        """
        if self._weights_matrix is None:
            if self.parallel and self._dask_futures:
                self.compute()
            else:
                raise RuntimeError("Weights have not been generated yet.")

        if hasattr(self._weights_matrix, "key"):
            self._weights_matrix = self._dask_client.gather(self._weights_matrix)
            if hasattr(self._total_weights, "key"):
                self._total_weights = self._dask_client.gather(self._total_weights)

        return self._weights_matrix

    def diagnostics(self) -> xr.Dataset:
        """
        Generate spatial diagnostics of the regridding weights.

        Returns
        -------
        xr.Dataset
            Dataset on the target grid containing:
            - weight_sum: Sum of weights for each destination cell.
            - unmapped_mask: Boolean mask (1 for unmapped cells, 0 for mapped).
        """
        if self._weights_matrix is None:
            raise RuntimeError("Weights have not been generated yet.")

        if hasattr(self._weights_matrix, "key"):
            # Aero Protocol: Distributed lazy diagnostics
            import dask.array as da
            import dask.distributed

            if self._total_weights is None:
                self._total_weights = self._dask_client.submit(
                    _get_weights_sum_task, self._weights_matrix
                )

            # Convert Future to Dask array to preserve laziness
            n_dst = int(np.prod(self._shape_target))

            weights_sum_da = da.from_delayed(
                dask.delayed(self._total_weights), shape=(n_dst,), dtype=np.float64
            )

            weights_sum_2d = weights_sum_da.reshape(self._shape_target)
            unmapped_2d = (weights_sum_2d == 0).astype(np.int8)
        else:
            # Eager diagnostics
            weights_sum = np.array(self._weights_matrix.sum(axis=1)).flatten()
            unmapped = (weights_sum == 0).astype(np.int8)

            # Reshape to target grid shape
            weights_sum_2d = weights_sum.reshape(self._shape_target)
            unmapped_2d = unmapped.reshape(self._shape_target)

        coords = {}
        if self.target_grid_ds is not None:
            coords = {
                c: self.target_grid_ds.coords[c]
                for c in self.target_grid_ds.coords
                if self._dims_target is not None
                and set(self.target_grid_ds.coords[c].dims).issubset(
                    set(self._dims_target)
                )
            }

        dims_target = self._dims_target
        if dims_target is None:
            # Fallback for mock objects without full initialization (Aero Protocol: Robustness)
            if self._shape_target is not None:
                dims_target = tuple(f"dim_{i}" for i in range(len(self._shape_target)))
            else:
                dims_target = ()

        ds = xr.Dataset(
            data_vars={
                "weight_sum": (dims_target, weights_sum_2d),
                "unmapped_mask": (dims_target, unmapped_2d),
            },
            coords=coords,
        )

        # Propagate CRS metadata (Aero Protocol: No Ambiguous Plots)
        target_crs_obj = get_crs_info(self.target_grid_ds)
        if target_crs_obj:
            ds.attrs["crs"] = target_crs_obj.to_wkt()

        update_history(ds, "Generated spatial diagnostics from Regridder weights.")
        return ds

    def quality_report(
        self, skip_heavy: bool = False, format: str = "dict"
    ) -> Union[dict[str, Any], xr.Dataset]:
        """
        Generate a scientific quality report of the regridding weights.

        Parameters
        ----------
        skip_heavy : bool, default False
            If True, skip metrics that require full weight matrix summation
            (expensive for massive grids).
        format : str, default 'dict'
            The output format: 'dict' or 'dataset'.

        Returns
        -------
        dict or xr.Dataset
            Quality metrics including unmapped points and weight sums.
            - unmapped_count: Number of destination points with no weights.
            - unmapped_fraction: Fraction of unmapped destination points.
            - weight_sum_min: Minimum sum of weights across destination points.
            - weight_sum_max: Maximum sum of weights across destination points.
            - weight_sum_mean: Mean sum of weights across destination points.
            - n_src: Number of source points.
            - n_dst: Number of destination points.
            - n_weights: Total number of non-zero weights.
        """
        if self._weights_matrix is None:
            raise RuntimeError("Weights have not been generated yet.")

        # Aero Protocol: Distributed metrics.
        # Compute metrics on the cluster if weights are remote to avoid driver OOM.
        is_remote = hasattr(self._weights_matrix, "key")

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        n_weights = -1
        if is_remote:
            if not skip_heavy:
                # Compute nnz on cluster (Aero Protocol: Optimized Distributed Metrics)
                try:
                    import dask.distributed

                    client = self._dask_client or dask.distributed.get_client()
                    n_weights_future = client.submit(
                        _get_nnz_task, self._weights_matrix
                    )
                    # We wait for the scalar result (Aero Protocol: Expected block for reports)
                    n_weights = int(n_weights_future.result())
                except (ImportError, ValueError, AttributeError):
                    n_weights = -1
            else:
                # If skip_heavy=True and remote, we don't even do a roundtrip
                n_weights = -1
        else:
            n_weights = int(self._weights_matrix.nnz)

        report = {
            "n_src": n_src,
            "n_dst": n_dst,
            "n_weights": n_weights,
            "method": self.method,
            "periodic": self.periodic,
        }

        if not skip_heavy:
            ds_diag = self.diagnostics()
            weights_sum = ds_diag.weight_sum
            unmapped_mask = ds_diag.unmapped_mask

            unmapped_count = int(unmapped_mask.sum())

            report.update(
                {
                    "unmapped_count": unmapped_count,
                    "unmapped_fraction": float(unmapped_count / n_dst),
                    "weight_sum_min": float(weights_sum.where(unmapped_mask == 0).min())
                    if unmapped_count < n_dst
                    else 0.0,
                    "weight_sum_max": float(weights_sum.max()),
                    "weight_sum_mean": float(weights_sum.mean()),
                }
            )

        if format == "dataset":
            ds_report = xr.Dataset(
                data_vars={
                    k: ([], v, {"description": f"Quality metric: {k}"})
                    for k, v in report.items()
                    if isinstance(v, (int, float)) or np.issubdtype(type(v), np.number)
                },
                attrs={
                    "method": self.method,
                    "periodic": int(self.periodic),
                    "provenance": "; ".join(self.provenance),
                },
            )
            update_history(ds_report, "Generated scientific quality report.")
            return ds_report

        return report

    def weights_to_xarray(self) -> xr.Dataset:
        """
        Export regridding weights and metadata as an xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing 'row', 'col', and 'S' (weights).
        """
        # Use weights property to ensure they are gathered if remote
        weights_matrix = self.weights

        weights_coo = weights_matrix.tocoo()
        ds = xr.Dataset(
            data_vars={
                "row": (["n_s"], weights_coo.row + 1),
                "col": (["n_s"], weights_coo.col + 1),
                "S": (["n_s"], weights_coo.data),
            },
            attrs={
                "n_src": self._weights_matrix.shape[1],
                "n_dst": self._weights_matrix.shape[0],
                "shape_src": list(self._shape_source) if self._shape_source else [],
                "shape_dst": list(self._shape_target) if self._shape_target else [],
                "method": self.method,
                "periodic": int(self.periodic),
                "provenance": "; ".join(self.provenance),
            },
        )
        return ds

    def __repr__(self) -> str:
        """
        String representation of the Regridder.

        Returns
        -------
        str
            Summary of the regridder configuration.
        """
        quality_str = "quality=deferred"
        n_dst = int(np.prod(self._shape_target)) if self._shape_target else 0

        # Aero Protocol: Avoid remote calls and expensive reports in __repr__
        is_remote = hasattr(self._weights_matrix, "key")
        if is_remote:
            quality_str = "quality=lazy"
        elif n_dst > 0:
            try:
                if n_dst < 1_000_000:
                    # Only show quality if already eager and not too massive
                    report = self.quality_report()
                    quality_str = f"unmapped={report['unmapped_fraction']:.2%}"
                else:
                    quality_str = "quality=deferred"
            except Exception:
                quality_str = "quality=unknown"

        return (
            f"Regridder(method={self.method}, "
            f"src_shape={self._shape_source}, "
            f"dst_shape={self._shape_target}, "
            f"periodic={self.periodic}, "
            f"{quality_str})"
        )

    def plot_weights(self, row_idx: int, **kwargs: Any) -> Any:
        """
        Track A: Visualize source points contributing to a specific destination point.

        Parameters
        ----------
        row_idx : int
            The index of the destination point (0-based).
        **kwargs : Any
            Additional arguments passed to plot_static.

        Returns
        -------
        Any
            The plot object.
        """
        from .viz import plot_weights as _plot_weights

        return _plot_weights(self, row_idx, **kwargs)

    def plot_diagnostics(self, mode: str = "static", **kwargs: Any) -> Any:
        """
        Visualize spatial diagnostics of the regridding weights.

        Follows the Aero Protocol's Two-Track Rule:
        - mode='static' (Track A): Publication-quality plot using Matplotlib/Cartopy.
        - mode='interactive' (Track B): Exploratory plot using HvPlot/HoloViews.

        Parameters
        ----------
        mode : str, default 'static'
            The plotting mode: 'static' or 'interactive'.
        **kwargs : Any
            Additional arguments passed to the plotting functions.

        Returns
        -------
        Any
            The plot object.
        """
        from .viz import plot_diagnostics as _plot_static
        from .viz import plot_diagnostics_interactive as _plot_interactive

        if mode == "static":
            return _plot_static(self, **kwargs)
        elif mode == "interactive":
            rasterize = kwargs.pop("rasterize", True)
            return _plot_interactive(self, rasterize=rasterize, **kwargs)
        else:
            raise ValueError(
                f"Unknown plotting mode: '{mode}'. Must be 'static' or 'interactive'."
            )

    def __call__(
        self,
        obj: Union[xr.DataArray, xr.Dataset, Any],
        skipna: Optional[bool] = None,
        na_thres: Optional[float] = None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply regridding to an input DataArray or Dataset.

        Parameters
        ----------
        obj : xarray.DataArray or xarray.Dataset
            The input data to regrid.
        skipna : bool, optional
            Whether to handle NaNs by re-normalizing weights.
            If None, uses the value set during initialization.
        na_thres : float, optional
            Threshold for NaN handling.
            If None, uses the value set during initialization.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The regridded data.
        """
        if self.parallel and self._weights_matrix is None:
            self.compute()

        if skipna is None:
            skipna = self.skipna
        if na_thres is None:
            na_thres = self.na_thres

        # Gather weights if input is eager (NumPy) but weights are lazy (Dask Future)
        # (Aero Protocol: Flexibility)
        is_lazy_input = False
        if isinstance(obj, xr.DataArray):
            is_lazy_input = hasattr(obj.data, "dask")
        elif isinstance(obj, xr.Dataset):
            # Check if any data variable is dask-backed
            is_lazy_input = any(hasattr(v.data, "dask") for v in obj.data_vars.values())

        if not is_lazy_input and hasattr(self._weights_matrix, "key"):
            self._weights_matrix = self._dask_client.gather(self._weights_matrix)
            if hasattr(self._total_weights, "key"):
                self._total_weights = self._dask_client.gather(self._total_weights)

        if isinstance(obj, xr.Dataset):
            # Sort input object if source grid was normalized
            if self._src_was_sorted:
                obj = obj.sortby([self._dims_source[0], self._dims_source[1]])
            res = self._regrid_dataset(obj, skipna=skipna, na_thres=na_thres)
            # Restore original target coordinate order if it was sorted
            if self._tgt_was_sorted:
                # Use sel to restore order without full reindexing if possible
                res = res.sel({d: self.target_grid_ds[d] for d in self._dims_target})
            return res
        elif isinstance(obj, xr.DataArray):
            if self._src_was_sorted:
                obj = obj.sortby([self._dims_source[0], self._dims_source[1]])
            res = self._regrid_dataarray(obj, skipna=skipna, na_thres=na_thres)
            if self._tgt_was_sorted:
                res = res.sel({d: self.target_grid_ds[d] for d in self._dims_target})
            return res
        # Handle uxarray objects if they don't pass isinstance(xr.Dataset)
        elif hasattr(obj, "uxgrid"):
            if hasattr(obj, "data_vars"):
                return self._regrid_dataset(obj, skipna=skipna, na_thres=na_thres)
            else:
                return self._regrid_dataarray(obj, skipna=skipna, na_thres=na_thres)
        else:
            raise TypeError("Input must be an xarray.DataArray or xarray.Dataset.")

    def _regrid_dataarray(
        self,
        da_in: xr.DataArray,
        update_history_attr: bool = True,
        _processed_ids: Optional[set[Union[int, str]]] = None,
        skipna: Optional[bool] = None,
        na_thres: Optional[float] = None,
    ) -> xr.DataArray:
        """
        Regrid a single DataArray, including auxiliary spatial coordinates.

        Parameters
        ----------
        da_in : xr.DataArray
            The input DataArray.
        update_history_attr : bool, default True
            Whether to update the history attribute.
        _processed_ids : set of int or str, optional
            Set of object IDs or names already being processed to avoid infinite recursion.
        skipna : bool, optional
            Whether to handle NaNs. If None, uses initialization default.
        na_thres : float, optional
            NaN threshold. If None, uses initialization default.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        if _processed_ids is None:
            _processed_ids = set()

        if skipna is None:
            skipna = self.skipna
        if na_thres is None:
            na_thres = self.na_thres

        # If skipna is True, we need _total_weights.
        # If it was not computed during init, compute it now.
        if skipna and self._total_weights is None and self._weights_matrix is not None:
            if hasattr(self._weights_matrix, "key"):
                # Distributed path: compute total weights on cluster
                self._total_weights = self._dask_client.submit(
                    _get_weights_sum_task, self._weights_matrix
                )
            else:
                # Eager path: compute locally and flatten to 1D
                self._total_weights = np.array(
                    self._weights_matrix.sum(axis=1)
                ).flatten()

        # Identify auxiliary coordinates that need regridding (Aero Protocol: Scientific Hygiene)
        aux_coords_to_regrid = {}

        # Track this DataArray to prevent mutual recursion (Aero Protocol: Robustness)
        # Using both ID and name for maximum safety
        _processed_ids.add(id(da_in))
        if da_in.name is not None:
            _processed_ids.add(str(da_in.name))

        for c_name, c_da in da_in.coords.items():
            # Avoid infinite recursion
            if id(c_da) in _processed_ids or c_name in _processed_ids:
                continue

            if c_name not in da_in.dims and all(
                d in c_da.dims for d in self._dims_source
            ):
                # This is an auxiliary spatial coordinate
                aux_coords_to_regrid[c_name] = self._regrid_dataarray(
                    c_da,
                    update_history_attr=False,
                    _processed_ids=_processed_ids,
                    skipna=skipna,
                    na_thres=na_thres,
                )

        # CF-Awareness: Map logical dimensions to physical dimension names in da_in
        # (Aero Protocol: Flexibility)

        input_core_dims = list(self._dims_source)

        # Check if we need to rename dimensions of da_in to match expected source dims
        # To be truly Aero-robust, we should check if da_in has all dims in self._dims_source
        missing_dims = [d for d in self._dims_source if d not in da_in.dims]
        if missing_dims:
            # Attempt CF-based mapping
            try:
                # Heuristic: Map cf latitude dims to the first part of _dims_source
                # and cf longitude dims to the rest.
                # This is complex for general cases, so we use a safe renaming approach.
                if not self._is_unstructured_src:
                    # Assume self._dims_source is (lat_dim, lon_dim) for rectilinear
                    # or (y, x) for curvilinear
                    if len(self._dims_source) == 2:
                        da_in = da_in.cf.rename(
                            {
                                "latitude": self._dims_source[0],
                                "longitude": self._dims_source[1],
                            }
                        )
                else:
                    # Unstructured: just one dimension
                    try:
                        da_in = da_in.cf.rename(
                            {da_in.cf["latitude"].dims[0]: self._dims_source[0]}
                        )
                    except (KeyError, AttributeError):
                        # Handle uxarray
                        if hasattr(da_in, "uxgrid"):
                            # Find the unstructured dimension
                            for d in da_in.dims:
                                if d in [
                                    "n_face",
                                    "n_node",
                                    "n_edge",
                                    "nCells",
                                    "nVertices",
                                    "nEdges",
                                ]:
                                    da_in = da_in.rename({d: self._dims_source[0]})
                                    break
            except (KeyError, AttributeError, ValueError):
                # Fallback to original dims; xr.apply_ufunc will raise if they don't match
                pass

        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        weights_arg = self._weights_matrix
        total_weights_arg = self._total_weights
        weights_key_arg = None

        # Optimization: Use worker-local cache for weights and total_weights to avoid
        # serialization overhead when using Dask. (Aero Protocol: Dask Efficiency)
        if hasattr(da_in.data, "dask"):
            client = self._dask_client
            if client is None:
                try:
                    import dask.distributed

                    client = dask.distributed.get_client()
                except (ImportError, ValueError):
                    client = None

            if client is not None:
                # Optimization: Identify weights by their Dask key if available, or memory ID
                if hasattr(self._weights_matrix, "key"):
                    weights_key_arg = f"weights_{self._weights_matrix.key}"
                else:
                    weights_key_arg = f"weights_{id(self._weights_matrix)}"

                # Use client ID to ensure cache is valid for current cluster
                client_id = getattr(client, "id", id(client))

                # Ensure weights are in worker-local cache (Aero Protocol: Efficiency)
                if (client_id, weights_key_arg) not in _DRIVER_CACHE:
                    if hasattr(self._weights_matrix, "key"):
                        # Truly Distributed: matrix is already a Future on the cluster.
                        # Sync cache on ALL current workers using robust client.run
                        client.replicate(self._weights_matrix)
                        client.run(
                            _sync_cache_from_worker_data,
                            self._weights_matrix.key,
                            weights_key_arg,
                        )
                    else:
                        # Eager matrix: run on all workers
                        client.run(
                            _setup_worker_cache, weights_key_arg, self._weights_matrix
                        )
                    _DRIVER_CACHE[(client_id, weights_key_arg)] = True
                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    if hasattr(self._total_weights, "key"):
                        tw_key = f"tw_{self._total_weights.key}"
                    else:
                        tw_key = f"tw_{id(self._total_weights)}"

                    if (client_id, tw_key) not in _DRIVER_CACHE:
                        if hasattr(self._total_weights, "key"):
                            client.replicate(self._total_weights)
                            client.run(
                                _sync_cache_from_worker_data,
                                self._total_weights.key,
                                tw_key,
                            )
                        else:
                            client.run(_setup_worker_cache, tw_key, self._total_weights)
                        _DRIVER_CACHE[(client_id, tw_key)] = True
                    total_weights_arg = tw_key

        # Use allow_rechunk=True to support chunked core dimensions
        # and move output_sizes to dask_gufunc_kwargs for future compatibility
        # vectorize=False because _apply_weights_core handles non-core dimensions
        out = xr.apply_ufunc(
            _apply_weights_core,
            da_in,
            kwargs={
                "weights_matrix": weights_arg,
                "dims_source": self._dims_source,
                "shape_target": self._shape_target,
                "skipna": skipna,
                "total_weights": total_weights_arg,
                "na_thres": na_thres,
                "weights_key": weights_key_arg,
            },
            input_core_dims=[input_core_dims],
            output_core_dims=[temp_output_core_dims],
            dask="parallelized",
            vectorize=False,
            output_dtypes=[da_in.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {
                    d: s for d, s in zip(temp_output_core_dims, self._shape_target)
                },
                "allow_rechunk": True,
            },
        )

        out = out.rename(
            {temp: orig for temp, orig in zip(temp_output_core_dims, self._dims_target)}
        )

        # Preserve name and attributes
        out.name = da_in.name
        out.attrs.update(da_in.attrs)
        out.encoding.update(da_in.encoding)

        # Assign coordinates from target grid (including scalar coords like grid_mapping)
        # (Aero Protocol: Scientific Hygiene)
        target_coords_to_assign = {}
        target_gm_name = None

        for c in self.target_grid_ds.coords:
            # Include coordinates that match target dimensions OR are scalar
            if set(self.target_grid_ds.coords[c].dims).issubset(set(self._dims_target)):
                target_coords_to_assign[c] = self.target_grid_ds.coords[c]
                # Identify if this is a grid_mapping coordinate
                if "grid_mapping_name" in self.target_grid_ds.coords[c].attrs:
                    target_gm_name = c

        # Also check data_vars in target_grid_ds for grid_mapping variables
        if target_gm_name is None:
            for v in self.target_grid_ds.data_vars:
                if "grid_mapping_name" in self.target_grid_ds[v].attrs:
                    target_gm_name = v
                    target_coords_to_assign[v] = self.target_grid_ds[v]

        out = out.assign_coords(target_coords_to_assign)

        # Update grid_mapping attribute (Aero Protocol: Scientific Hygiene)
        if target_gm_name:
            out.attrs["grid_mapping"] = target_gm_name
            if "grid_mapping" in out.encoding:
                out.encoding["grid_mapping"] = target_gm_name
        else:
            # If target has no grid mapping, remove source one as it's no longer valid for this grid
            if "grid_mapping" in out.attrs:
                del out.attrs["grid_mapping"]
            if "grid_mapping" in out.encoding:
                del out.encoding["grid_mapping"]

        # Propagate CRS metadata (Aero Protocol: Scientific Hygiene)
        target_crs_obj = get_crs_info(self.target_grid_ds)
        if target_crs_obj:
            out.attrs["crs"] = target_crs_obj.to_wkt()
        elif "crs" in out.attrs:
            # Remove source CRS as it's no longer valid
            del out.attrs["crs"]

        # Re-attach regridded auxiliary coordinates
        if aux_coords_to_regrid:
            out = out.assign_coords(aux_coords_to_regrid)

        # Update history for provenance
        if update_history_attr:
            try:
                import esmpy

                esmpy_version = getattr(esmpy, "__version__", "unknown")
            except ImportError:
                esmpy_version = "unknown"

            history_msg = (
                f"Regridded using xregrid.Regridder (ESMF/esmpy={esmpy_version}, "
                f"method={self.method}, periodic={self.periodic}, skipna={skipna}, "
                f"na_thres={na_thres}"
            )
            if self.extrap_method:
                history_msg += f", extrap_method={self.extrap_method}"
            history_msg += ")"

            for msg in self.provenance:
                history_msg += f". {msg}"
            if self.generation_time:
                history_msg += f". Weight generation time: {self.generation_time:.4f}s"
            update_history(out, history_msg)

        return out

    def _regrid_dataset(
        self,
        ds_in: xr.Dataset,
        skipna: Optional[bool] = None,
        na_thres: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Regrid all data variables and auxiliary coordinates in a Dataset.

        Parameters
        ----------
        ds_in : xr.Dataset
            The input Dataset.
        skipna : bool, optional
            Whether to handle NaNs.
        na_thres : float, optional
            NaN threshold.

        Returns
        -------
        xr.Dataset
            The regridded Dataset.
        """
        if skipna is None:
            skipna = self.skipna
        if na_thres is None:
            na_thres = self.na_thres

        regridded_items: dict[str, Union[xr.DataArray, Any]] = {}

        # 1. Regrid data variables
        for name, da in ds_in.data_vars.items():
            # CF-Awareness: Check for spatial dimensions using logical axes (Aero Protocol)
            is_regriddable = False
            if all(dim in da.dims for dim in self._dims_source):
                is_regriddable = True
            else:
                try:
                    # Check if variable has logical latitude and longitude
                    spatial_dims = set(da.cf["latitude"].dims) | set(
                        da.cf["longitude"].dims
                    )
                    if spatial_dims.issubset(set(da.dims)):
                        is_regriddable = True
                except (KeyError, AttributeError):
                    pass

            if is_regriddable:
                # Initialize _processed_ids with the name and ID of the current variable
                # to prevent it from trying to regrid itself if it appears as a coordinate.
                regridded_items[name] = self._regrid_dataarray(
                    da,
                    update_history_attr=False,
                    _processed_ids={id(da), name},
                    skipna=skipna,
                    na_thres=na_thres,
                )
            else:
                regridded_items[name] = da

        out = xr.Dataset(regridded_items, attrs=ds_in.attrs)

        # 2. Scientific Hygiene: Regrid auxiliary spatial coordinates and preserve others (Aero Protocol)
        # and ensure grid_mapping from target grid is attached.
        for c in ds_in.coords:
            if c in out.coords:
                continue

            # Check if this coordinate depends on spatial dimensions
            if all(d in ds_in.coords[c].dims for d in self._dims_source):
                # If it's not a dimension coordinate, regrid it
                if c not in ds_in.dims:
                    out = out.assign_coords(
                        {
                            c: self._regrid_dataarray(
                                ds_in.coords[c],
                                update_history_attr=False,
                                _processed_ids={id(ds_in.coords[c]), c},
                                skipna=skipna,
                                na_thres=na_thres,
                            )
                        }
                    )
            else:
                # Not dependent on spatial dims, just preserve it
                out = out.assign_coords({c: ds_in.coords[c]})

        # 3. Handle grid_mapping and scalar coordinates from target grid (Aero Protocol)
        target_gm_name = None
        for c in self.target_grid_ds.coords:
            if c not in out.coords:
                if set(self.target_grid_ds.coords[c].dims).issubset(
                    set(self._dims_target)
                ):
                    out = out.assign_coords({c: self.target_grid_ds.coords[c]})

            # Identify target grid mapping variable
            if "grid_mapping_name" in self.target_grid_ds.coords[c].attrs:
                target_gm_name = c

        # Also check target data_vars for grid_mapping
        if target_gm_name is None:
            for v in self.target_grid_ds.data_vars:
                if "grid_mapping_name" in self.target_grid_ds[v].attrs:
                    target_gm_name = v
                    if target_gm_name not in out.coords:
                        out = out.assign_coords(
                            {target_gm_name: self.target_grid_ds[v]}
                        )

        # Update global grid_mapping attribute if it exists
        if target_gm_name:
            if "grid_mapping" in out.attrs:
                out.attrs["grid_mapping"] = target_gm_name
        else:
            # Remove invalid grid_mapping
            if "grid_mapping" in out.attrs:
                del out.attrs["grid_mapping"]

        # Propagate CRS metadata (Aero Protocol: Scientific Hygiene)
        target_crs_obj = get_crs_info(self.target_grid_ds)
        if target_crs_obj:
            out.attrs["crs"] = target_crs_obj.to_wkt()
        elif "crs" in out.attrs:
            # Remove source CRS as it's no longer valid
            del out.attrs["crs"]

        # Update history for provenance
        try:
            import esmpy

            esmpy_version = getattr(esmpy, "__version__", "unknown")
        except ImportError:
            esmpy_version = "unknown"

        history_msg = (
            f"Regridded Dataset using xregrid.Regridder (ESMF/esmpy={esmpy_version}, "
            f"method={self.method}, periodic={self.periodic}, skipna={skipna}, "
            f"na_thres={na_thres}"
        )
        if self.extrap_method:
            history_msg += f", extrap_method={self.extrap_method}"
        history_msg += ")"

        for msg in self.provenance:
            history_msg += f". {msg}"
        if self.generation_time:
            history_msg += f". Weight generation time: {self.generation_time:.4f}s"

        update_history(out, history_msg)

        return out
