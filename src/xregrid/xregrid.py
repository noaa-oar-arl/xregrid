from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

try:
    import esmpy
except ImportError:
    esmpy = None

from .utils import update_history


if TYPE_CHECKING:
    pass


# Global cache for workers to reuse ESMF source objects
_WORKER_CACHE: dict = {}


def _get_mesh_info(
    ds: xr.Dataset,
) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
    """Detect grid type and extract coordinates and shape."""
    try:
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]
    except (KeyError, AttributeError):
        if "lat" in ds and "lon" in ds:
            lat = ds["lat"]
            lon = ds["lon"]
        else:
            raise KeyError(
                "Could not find latitude/longitude coordinates. "
                "Ensure they are named 'lat'/'lon' or have CF attributes."
            )

    if lat.ndim == 2:
        # Curvilinear
        if lon.ndim == 2 and lon.dims != lat.dims and set(lon.dims) == set(lat.dims):
            lon = lon.transpose(*lat.dims)
        return lon, lat, lat.shape, lat.dims, False
    elif lat.ndim == 1:
        if lat.dims == lon.dims:
            # Unstructured (e.g. MPAS)
            return lon, lat, lat.shape, lat.dims, True
        else:
            # Rectilinear
            lon_mesh, lat_mesh = xr.broadcast(lon, lat)

            # Ensure they have the correct order (lat, lon) for the shape
            if lat.ndim == 2 and lon.ndim == 2:
                if lat.dims != lon.dims and set(lat.dims) == set(lon.dims):
                    lon = lon.transpose(*lat.dims)

            lon_mesh = lon_mesh.transpose(lat.dims[0], lon.dims[0])
            lat_mesh = lat_mesh.transpose(lat.dims[0], lon.dims[0])

            return (
                lon_mesh,
                lat_mesh,
                (lat.size, lon.size),
                (lat.dims[0], lon.dims[0]),
                False,
            )
    else:
        raise ValueError("Latitude and longitude must be 1D or 2D.")


def _bounds_to_vertices(b: xr.DataArray) -> np.ndarray:
    """Convert bounds to vertices for ESMF."""
    if b.ndim == 2 and b.shape[-1] == 2:
        return np.concatenate([b.values[:, 0], b.values[-1:, 1]])
    elif b.ndim == 3 and b.shape[-1] == 4:
        y_size, x_size, _ = b.shape
        vals = b.values
        res = np.empty((y_size + 1, x_size + 1))
        res[:-1, :-1] = vals[:, :, 0]
        res[:-1, -1] = vals[:, -1, 1]
        res[-1, -1] = vals[-1, -1, 2]
        res[-1, :-1] = vals[-1, :, 3]
        return res
    return b.values


def _get_grid_bounds(
    ds: xr.Dataset,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract grid cell boundaries from a dataset."""
    try:
        lat_b_da = ds.cf.get_bounds("latitude")
        lon_b_da = ds.cf.get_bounds("longitude")
        return _bounds_to_vertices(lat_b_da), _bounds_to_vertices(lon_b_da)
    except (KeyError, AttributeError, ValueError):
        if "lat_b" in ds and "lon_b" in ds:
            lat_b = (
                ds["lat_b"].values if hasattr(ds["lat_b"], "values") else ds["lat_b"]
            )
            lon_b = (
                ds["lon_b"].values if hasattr(ds["lon_b"], "values") else ds["lon_b"]
            )
            return lat_b, lon_b
    return None, None


def _create_esmf_grid(
    ds: xr.Dataset,
    method: str,
    periodic: bool = False,
    mask_var: Optional[str] = None,
) -> Union[esmpy.Grid, esmpy.LocStream]:
    """Creates an ESMF Grid or LocStream."""
    import esmpy

    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)

    if is_unstructured:
        if method not in ["nearest_s2d", "nearest_d2s"]:
            raise NotImplementedError(
                f"Method '{method}' is not yet supported for unstructured grids."
            )
        locstream = esmpy.LocStream(shape[0], coord_sys=esmpy.CoordSys.SPH_DEG)
        locstream["ESMF:Lon"] = lon.values.astype(np.float64)
        locstream["ESMF:Lat"] = lat.values.astype(np.float64)
        return locstream
    else:
        lon_f = lon.values.T
        lat_f = lat.values.T
        shape_f = lon_f.shape

        num_peri_dims = 1 if periodic else None
        periodic_dim = 0 if periodic else None
        pole_dim = 1 if periodic else None

        lat_b, lon_b = _get_grid_bounds(ds)

        if (lat_b is None or lon_b is None) and method == "conservative":
            try:
                ds_with_bounds = ds.cf.add_bounds(["latitude", "longitude"])
                lat_b, lon_b = _get_grid_bounds(ds_with_bounds)
                if lat_b is not None and lon_b is not None:
                    update_history(
                        ds,
                        f"Automatically generated cell boundaries for {method} regridding.",
                    )
            except Exception:
                pass

        has_bounds = lat_b is not None and lon_b is not None
        if method == "conservative" and not has_bounds:
            raise ValueError(
                "Conservative regridding requires cell boundaries (bounds). "
                "Ensure your dataset has 'lat_b' and 'lon_b' or CF-compliant bounds."
            )

        staggerlocs = [esmpy.StaggerLoc.CENTER]
        if has_bounds:
            staggerlocs.append(esmpy.StaggerLoc.CORNER)

        grid = esmpy.Grid(
            np.array(shape_f),
            staggerloc=staggerlocs,
            coord_sys=esmpy.CoordSys.SPH_DEG,
            num_peri_dims=num_peri_dims,
            periodic_dim=periodic_dim,
            pole_dim=pole_dim,
        )

        grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)[...] = lon_f.astype(
            np.float64
        )
        grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)[...] = lat_f.astype(
            np.float64
        )

        if has_bounds:
            if lon_b.ndim == 1 and lat_b.ndim == 1:
                lon_b_vals, lat_b_vals = np.meshgrid(lon_b, lat_b)
            else:
                lon_b_vals, lat_b_vals = lon_b, lat_b

            lon_b_vals_f = lon_b_vals.T
            lat_b_vals_f = lat_b_vals.T

            if periodic:
                lon_b_vals_f = lon_b_vals_f[:-1, :]
                lat_b_vals_f = lat_b_vals_f[:-1, :]

            grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)[...] = (
                lon_b_vals_f.astype(np.float64)
            )
            grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)[...] = (
                lat_b_vals_f.astype(np.float64)
            )

        if mask_var and mask_var in ds:
            grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
            grid.get_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)[
                ...
            ] = ds[mask_var].values.T.astype(np.int32)
        return grid


def _compute_chunk_weights(
    source_ds: xr.Dataset,
    chunk_ds: xr.Dataset,
    method: str,
    global_indices: np.ndarray,
    extrap_method: Optional[str] = None,
    extrap_dist_exponent: float = 2.0,
    mask_var: Optional[str] = None,
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """
    Worker function to compute weights for a specific chunk of the target grid.
    Uses worker-local caching for source ESMF objects.
    """
    try:
        import esmpy

        # Initialize Manager if not already done in this process
        esmpy.Manager(debug=False)

        # 1. Get or create source ESMF Field
        # We use id(source_ds) as a key. Dask ensures the same object is reused on a worker.
        src_cache_key = (id(source_ds), method, periodic, mask_var)
        if src_cache_key in _WORKER_CACHE:
            src_field = _WORKER_CACHE[src_cache_key]
        else:
            src_obj = _create_esmf_grid(source_ds, method, periodic, mask_var)
            src_field = esmpy.Field(src_obj, name="src")
            _WORKER_CACHE[src_cache_key] = src_field

        # 2. Create target ESMF object (chunk is small, no need to cache)
        dst_obj = _create_esmf_grid(chunk_ds, method, periodic=False, mask_var=None)
        dst_field = esmpy.Field(dst_obj, name="dst")

        # 3. Setup regridding parameters
        method_map = {
            "bilinear": esmpy.RegridMethod.BILINEAR,
            "conservative": esmpy.RegridMethod.CONSERVE,
            "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
            "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
            "patch": esmpy.RegridMethod.PATCH,
        }
        extrap_method_map = {
            "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
            "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
            "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
        }

        regrid_kwargs = {
            "regrid_method": method_map[method],
            "unmapped_action": esmpy.UnmappedAction.IGNORE,
            "factors": True,
        }
        if extrap_method:
            regrid_kwargs["extrap_method"] = extrap_method_map[extrap_method]
            regrid_kwargs["extrap_dist_exponent"] = extrap_dist_exponent

        if isinstance(src_field.grid, esmpy.Grid) and mask_var:
            regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

        # 4. Generate weights
        regrid = esmpy.Regrid(src_field, dst_field, **regrid_kwargs)
        weights = regrid.get_weights_dict(deep_copy=True)

        rows = global_indices[weights["row_dst"] - 1]
        cols = weights["col_src"] - 1
        data = weights["weights"]

        return rows, cols, data, None

    except Exception as e:
        import traceback

        return (
            np.array([]),
            np.array([]),
            np.array([]),
            f"{str(e)}\n{traceback.format_exc()}",
        )


def _apply_weights_core(
    data_block: np.ndarray,
    weights_matrix: Any,
    dims_source: Tuple[str, ...],
    shape_target: Tuple[int, ...],
    skipna: bool = False,
    total_weights: Optional[np.ndarray] = None,
    na_thres: float = 1.0,
) -> np.ndarray:
    """
    Apply regridding weights to a data block (NumPy array).

    Parameters
    ----------
    data_block : np.ndarray
        The input data block. Core dimensions must be at the end.
    weights_matrix : scipy.sparse.csr_matrix
        The sparse weight matrix.
    dims_source : tuple of str
        The names of the source spatial dimensions.
    shape_target : tuple of int
        The shape of the target spatial grid.
    skipna : bool, default False
        Whether to handle NaNs by re-normalizing weights.
    total_weights : np.ndarray, optional
        Pre-computed sum of weights for each destination cell.
    na_thres : float, default 1.0
        Threshold for NaN handling.

    Returns
    -------
    np.ndarray
        The regridded data block.
    """
    original_shape = data_block.shape
    # Core dimensions are at the end
    n_source_dims = len(dims_source)
    spatial_shape = original_shape[len(original_shape) - n_source_dims :]
    other_dims_shape = original_shape[: len(original_shape) - n_source_dims]
    n_spatial = int(np.prod(spatial_shape))
    n_other = int(np.prod(other_dims_shape))
    flat_data = data_block.reshape(n_other, n_spatial)

    if skipna:
        mask = np.isnan(flat_data)
        has_nans = np.any(mask)

        if not has_nans:
            # Fast path: No NaNs in this data block
            # Optimized CSR application: (matrix @ data.T).T is faster than data @ matrix.T
            result = (weights_matrix @ flat_data.T).T
            if total_weights is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = result / total_weights
        else:
            # Slow path: Handle NaNs by re-normalizing weights
            safe_data = np.where(mask, 0.0, flat_data)
            result = (weights_matrix @ safe_data.T).T
            # Sum weights of valid (non-NaN) points
            weights_sum = (weights_matrix @ (~mask).astype(np.float32).T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                final_result = result / weights_sum
                if total_weights is not None:
                    fraction_valid = weights_sum / total_weights
                    final_result = np.where(
                        fraction_valid >= (1.0 - na_thres - 1e-6),
                        final_result,
                        np.nan,
                    )
            result = final_result
    else:
        # Standard path (skipna=False): Just apply weights
        # Optimized CSR application: (matrix @ data.T).T is faster than data @ matrix.T
        result = (weights_matrix @ flat_data.T).T

    new_shape = other_dims_shape + shape_target
    return result.reshape(new_shape)


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
        if esmpy is None:
            raise ImportError(
                "ESMPy is required for Regridder. "
                "Please install it via conda: `conda install -c conda-forge esmpy`"
            )

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
        if mpi:
            # Use MULTI logkind for MPI parallelization (Aero Protocol)
            self._manager = esmpy.Manager(logkind=esmpy.LogKind.MULTI, debug=False)
        else:
            self._manager = esmpy.Manager(debug=False)

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

        # Internal state
        self._shape_source: Optional[Tuple[int, ...]] = None
        self._shape_target: Optional[Tuple[int, ...]] = None
        self._dims_source: Optional[Tuple[str, ...]] = None
        self._dims_target: Optional[Tuple[str, ...]] = None
        self._is_unstructured_src: bool = False
        self._is_unstructured_tgt: bool = False
        self._total_weights: Optional[np.ndarray] = None
        self._weights_matrix: Optional[coo_matrix] = None
        self._loaded_method: Optional[str] = None
        self._loaded_periodic: Optional[bool] = None
        self._loaded_extrap: Optional[str] = None
        self.generation_time: Optional[float] = None
        self._dask_futures: Optional[list] = None
        self._dask_client: Optional[Any] = None
        self._dask_start_time: Optional[float] = None

        if reuse_weights and os.path.exists(filename):
            self._load_weights()
            # Validate loaded weights against provided grids and parameters
            self._validate_weights()
        else:
            self._generate_weights()
            if reuse_weights:
                self._save_weights()

    def _validate_weights(self) -> None:
        """
        Validate loaded weights against the provided source and target grids.

        Ensures that shapes, dimension names, regridding method, and periodicity
        match the requested configuration to maintain scientific integrity.
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

    def _get_mesh_info(
        self, ds: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
        """Detect grid type and extract coordinates and shape."""
        return _get_mesh_info(ds)

    def _bounds_to_vertices(self, b: xr.DataArray) -> np.ndarray:
        """Convert bounds to vertices for ESMF."""
        return _bounds_to_vertices(b)

    def _get_grid_bounds(
        self, ds: xr.Dataset
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract grid cell boundaries from a dataset."""
        return _get_grid_bounds(ds)

    def _create_esmf_object(
        self, ds: xr.Dataset, is_source: bool = True
    ) -> Union[esmpy.Grid, esmpy.LocStream]:
        """
        Creates an ESMF Grid or LocStream.
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
        )

    def _generate_weights(self) -> None:
        """Generate regridding weights using ESMPy."""
        if self.parallel:
            self._generate_weights_dask(compute=self.compute_on_init)
            return

        start_time = time.perf_counter()
        src_obj = self._create_esmf_object(self.source_grid_ds, is_source=True)
        dst_obj = self._create_esmf_object(self.target_grid_ds, is_source=False)

        src_field = esmpy.Field(src_obj, name="src")
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

        if not self._is_unstructured_src and not self._is_unstructured_tgt:
            if self.mask_var and self.mask_var in self.source_grid_ds:
                regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

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
                    rows = np.concatenate([w["row_dst"] for w in all_weights]) - 1
                    cols = np.concatenate([w["col_src"] for w in all_weights]) - 1
                    data = np.concatenate([w["weights"] for w in all_weights])
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
            rows = weights["row_dst"] - 1
            cols = weights["col_src"] - 1
            data = weights["weights"]

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        self._weights_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

        self.generation_time = time.perf_counter() - start_time

    def _generate_weights_dask(self, compute: bool = True) -> None:
        """Generate regridding weights using Dask parallel workers."""
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

        if is_unstructured_dst:
            raise NotImplementedError(
                "Dask parallelization not yet optimized/verified for unstructured target grids."
            )

        # Get client
        try:
            client = dask.distributed.get_client()
        except ValueError:
            # Create a local cluster if none exists
            cluster = dask.distributed.LocalCluster()
            client = dask.distributed.Client(cluster)

        self._dask_client = client

        # Split target grid along available spatial dimensions
        dim0 = dst_dims[0]
        dim1 = dst_dims[1] if len(dst_dims) > 1 else None

        size0 = self.target_grid_ds.sizes[dim0]
        size1 = self.target_grid_ds.sizes[dim1] if dim1 else 1

        # Determine number of chunks. Use number of workers * 2 usually good heuristic
        n_workers = len(client.scheduler_info()["workers"])
        n_chunks_total = max(1, n_workers * 2)

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

        # Pre-compute global indices for the target grid to handle non-contiguous chunks
        global_indices = np.arange(size0 * size1).reshape(size0, size1)

        futures = []

        # Scatter source grid info (minimal)
        source_grid_only = self.source_grid_ds.copy(deep=False)
        for var in list(source_grid_only.data_vars):
            if var != self.mask_var:
                source_grid_only = source_grid_only.drop_vars(var)
        src_ds_future = client.scatter(source_grid_only, broadcast=True)

        for idx0 in indices0:
            if len(idx0) == 0:
                continue
            for idx1 in indices1:
                if len(idx1) == 0:
                    continue

                # Slice target grid
                sel_dict = {dim0: slice(idx0[0], idx0[-1] + 1)}
                if dim1:
                    sel_dict[dim1] = slice(idx1[0], idx1[-1] + 1)

                chunk_ds = self.target_grid_ds.isel(sel_dict)

                # Extract global indices for this chunk
                chunk_global_indices = global_indices[
                    idx0[0] : idx0[-1] + 1, idx1[0] : idx1[-1] + 1
                ].flatten()

                future = client.submit(
                    _compute_chunk_weights,
                    src_ds_future,
                    chunk_ds,
                    self.method,
                    chunk_global_indices,
                    self.extrap_method,
                    self.extrap_dist_exponent,
                    self.mask_var,
                    self.periodic,
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
        """
        if not self.parallel or self._weights_matrix is not None:
            return

        if not self._dask_futures:
            # This means compute=False was not used, or something went wrong?
            # Or maybe parallel=True but _generate_weights_dask wasn't called yet?
            # But __init__ calls _generate_weights.
            return

        # Gather results
        results = self._dask_client.gather(self._dask_futures)

        all_rows = []
        all_cols = []
        all_data = []

        for i, (r, c, d, err) in enumerate(results):
            if err:
                raise RuntimeError(f"Dask worker {i} failed: {err}")
            all_rows.append(r)
            all_cols.append(c)
            all_data.append(d)

        full_rows = np.concatenate(all_rows)
        full_cols = np.concatenate(all_cols)
        full_data = np.concatenate(all_data)

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        self._weights_matrix = coo_matrix(
            (full_data, (full_rows, full_cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

        if self._dask_start_time:
            self.generation_time = time.perf_counter() - self._dask_start_time

        # Clear futures to free memory
        self._dask_futures = None

    def _save_weights(self) -> None:
        """Save weights to a NetCDF file."""
        if esmpy.local_pet() != 0:
            return  # Only rank 0 saves weights

        if self._weights_matrix is None:
            raise RuntimeError("Weights have not been generated yet.")

        # Convert to COO to access row and col attributes
        weights_coo = self._weights_matrix.tocoo()

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
        """Load weights from a NetCDF file."""
        with xr.open_dataset(self.filename) as ds_weights:
            ds_weights.load()
            rows = ds_weights["row"].values - 1
            cols = ds_weights["col"].values - 1
            data = ds_weights["S"].values
            n_src = ds_weights.attrs["n_src"]
            n_dst = ds_weights.attrs["n_dst"]

            def _to_tuple(attr: Any) -> Tuple[Any, ...]:
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
            self.generation_time = ds_weights.attrs.get("generation_time")

        self._weights_matrix = coo_matrix(
            (data, (rows, cols)), shape=(n_dst, n_src)
        ).tocsr()

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

    def __repr__(self) -> str:
        """
        String representation of the Regridder.

        Returns
        -------
        str
            Summary of the regridder configuration.
        """
        return (
            f"Regridder(method={self.method}, "
            f"src_shape={self._shape_source}, "
            f"dst_shape={self._shape_target}, "
            f"periodic={self.periodic})"
        )

    def __call__(
        self, obj: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply regridding to an input DataArray or Dataset.

        Parameters
        ----------
        obj : xarray.DataArray or xarray.Dataset
            The input data to regrid.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The regridded data.
        """
        if self.parallel and self._weights_matrix is None:
            self.compute()

        if isinstance(obj, xr.Dataset):
            return self._regrid_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            return self._regrid_dataarray(obj)
        else:
            raise TypeError("Input must be an xarray.DataArray or xarray.Dataset.")

    def _regrid_dataarray(
        self, da_in: xr.DataArray, update_history_attr: bool = True
    ) -> xr.DataArray:
        """
        Regrid a single DataArray.

        Parameters
        ----------
        da_in : xarray.DataArray
            The input DataArray.
        update_history_attr : bool, default True
            Whether to update the history attribute.

        Returns
        -------
        xarray.DataArray
            The regridded DataArray.
        """

        input_core_dims = list(self._dims_source)
        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        # Use allow_rechunk=True to support chunked core dimensions
        # and move output_sizes to dask_gufunc_kwargs for future compatibility
        # vectorize=False because _apply_weights_core handles non-core dimensions
        out = xr.apply_ufunc(
            _apply_weights_core,
            da_in,
            kwargs={
                "weights_matrix": self._weights_matrix,
                "dims_source": self._dims_source,
                "shape_target": self._shape_target,
                "skipna": self.skipna,
                "total_weights": self._total_weights,
                "na_thres": self.na_thres,
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

        # Assign coordinates from target grid
        out = out.assign_coords(
            {
                c: self.target_grid_ds.coords[c]
                for c in self.target_grid_ds.coords
                if set(self.target_grid_ds.coords[c].dims).issubset(
                    set(self._dims_target)
                )
            }
        )

        # Update history for provenance
        if update_history_attr:
            history_msg = (
                f"Regridded using Regridder (method={self.method}, "
                f"periodic={self.periodic}, skipna={self.skipna})"
            )
            if self.generation_time:
                history_msg += f". Weight generation time: {self.generation_time:.4f}s"
            update_history(out, history_msg)

        return out

    def _regrid_dataset(self, ds_in: xr.Dataset) -> xr.Dataset:
        """
        Regrid all data variables in a Dataset.

        Parameters
        ----------
        ds_in : xarray.Dataset
            The input Dataset.

        Returns
        -------
        xarray.Dataset
            The regridded Dataset.
        """
        regridded_vars = {}
        for name, da in ds_in.data_vars.items():
            if all(dim in da.dims for dim in self._dims_source):
                regridded_vars[name] = self._regrid_dataarray(
                    da, update_history_attr=False
                )
            else:
                regridded_vars[name] = da

        out = xr.Dataset(regridded_vars, attrs=ds_in.attrs)

        # Scientific Hygiene: Preserve coordinates that are not spatial dimensions
        # and were not already aligned by the Dataset constructor.
        for c in ds_in.coords:
            if c not in out.coords and not any(
                d in self._dims_source for d in ds_in.coords[c].dims
            ):
                out = out.assign_coords({c: ds_in.coords[c]})

        # Update history for provenance
        history_msg = (
            f"Regridded Dataset using Regridder (method={self.method}, "
            f"periodic={self.periodic}, skipna={self.skipna})"
        )
        if self.generation_time:
            history_msg += f". Weight generation time: {self.generation_time:.4f}s"

        update_history(out, history_msg)

        return out
