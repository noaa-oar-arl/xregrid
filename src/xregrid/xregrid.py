from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import esmpy
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

from .utils import update_history

if TYPE_CHECKING:
    pass


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
        """
        # Initialize ESMF Manager (required for some environments)
        if mpi:
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

        self.method_map = {
            "bilinear": esmpy.RegridMethod.BILINEAR,
            "conservative": esmpy.RegridMethod.CONSERVE,
            "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
            "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
            "patch": esmpy.RegridMethod.PATCH,
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

    def _get_mesh_info(
        self, ds: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], Tuple[str, ...], bool]:
        """
        Detect grid type and extract coordinates and shape.

        Uses cf-xarray for automatic coordinate detection if standard
        names 'lat' and 'lon' are not present.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to extract mesh info from.

        Returns
        -------
        lon : numpy.ndarray
            Longitude values.
        lat : numpy.ndarray
            Latitude values.
        shape : tuple of int
            Grid shape.
        dims : tuple of str
            Coordinate dimensions.
        is_unstructured : bool
            Whether the grid is unstructured.
        """
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

        # ESMF requires numpy arrays for grid definition.
        # While the Aero Protocol prioritizes laziness, the underlying ESMF C++ library
        # requires concrete memory buffers for grid construction. We therefore
        # compute coordinate values here if they are dask-backed.
        lon_vals = lon.values
        lat_vals = lat.values

        if lat.ndim == 2:
            # Curvilinear
            return lon_vals, lat_vals, lat.shape, lat.dims, False
        elif lat.ndim == 1:
            if lat.dims == lon.dims:
                # Unstructured (e.g. MPAS)
                return lon_vals, lat_vals, lat.shape, lat.dims, True
            else:
                # Rectilinear
                lon_mesh, lat_mesh = np.meshgrid(lon_vals, lat_vals)
                return (
                    lon_mesh,
                    lat_mesh,
                    (lat.size, lon.size),
                    (lat.dims[0], lon.dims[0]),
                    False,
                )
        else:
            raise ValueError("Latitude and longitude must be 1D or 2D.")

    def _create_esmf_object(
        self, ds: xr.Dataset, is_source: bool = True
    ) -> Union[esmpy.Grid, esmpy.LocStream]:
        """
        Creates an ESMF Grid or LocStream.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to create ESMF object from.
        is_source : bool, default True
            Whether this is the source grid.

        Returns
        -------
        Union[esmpy.Grid, esmpy.LocStream]
            The ESMF object.
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

        if is_unstructured:
            if self.method not in ["nearest_s2d", "nearest_d2s"]:
                raise NotImplementedError(
                    f"Method '{self.method}' is not yet supported for unstructured grids. "
                    "Currently only 'nearest_s2d' and 'nearest_d2s' are supported for "
                    "unstructured grids via LocStream."
                )

            locstream = esmpy.LocStream(shape[0], coord_sys=esmpy.CoordSys.SPH_DEG)
            locstream["ESMF:Lon"] = lon.astype(np.float64)
            locstream["ESMF:Lat"] = lat.astype(np.float64)
            return locstream
        else:
            # Transpose to (lon, lat) order for ESMF (SPH_DEG convention)
            lon_f = lon.T
            lat_f = lat.T
            shape_f = lon_f.shape  # (nlon, nlat)

            # Periodicity configuration
            num_peri_dims = 1 if self.periodic else None
            periodic_dim = 0 if self.periodic else None
            pole_dim = 1 if self.periodic else None

            # Attempt to find bounds using cf-xarray or standard names
            lat_b = None
            lon_b = None
            try:
                lat_b_da = ds.cf.get_bounds("latitude")
                lon_b_da = ds.cf.get_bounds("longitude")

                def _bounds_to_vertices(b: xr.DataArray) -> np.ndarray:
                    """Convert (N, 2) bounds to (N+1,) vertices."""
                    if b.ndim == 2 and b.shape[-1] == 2:
                        # Take the first column and the last element of the second column
                        # This assumes the bounds are contiguous
                        return np.concatenate([b[:, 0].values, b[-1:, 1].values])
                    return b.values

                lat_b = _bounds_to_vertices(lat_b_da)
                lon_b = _bounds_to_vertices(lon_b_da)
            except (KeyError, AttributeError):
                pass

            if lat_b is None or lon_b is None:
                if "lat_b" in ds and "lon_b" in ds:
                    lat_b = ds["lat_b"]
                    lon_b = ds["lon_b"]

            has_bounds = lat_b is not None and lon_b is not None
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

            grid_lon = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
            grid_lat = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
            grid_lon[...] = lon_f.astype(np.float64)
            grid_lat[...] = lat_f.astype(np.float64)

            if has_bounds:
                # lat_b and lon_b are already set from above (as numpy arrays or DataArrays)
                lat_b_val = lat_b.values if hasattr(lat_b, "values") else lat_b
                lon_b_val = lon_b.values if hasattr(lon_b, "values") else lon_b

                if lon_b_val.ndim == 1 and lat_b_val.ndim == 1:
                    lon_b_vals, lat_b_vals = np.meshgrid(lon_b_val, lat_b_val)
                else:
                    lon_b_vals, lat_b_vals = lon_b_val, lat_b_val

                grid_lon_b = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)
                grid_lat_b = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)

                lon_b_vals_f = lon_b_vals.T
                lat_b_vals_f = lat_b_vals.T

                if self.periodic:
                    # Remove the redundant corner in the periodic dimension
                    lon_b_vals_f = lon_b_vals_f[:-1, :]
                    lat_b_vals_f = lat_b_vals_f[:-1, :]

                grid_lon_b[...] = lon_b_vals_f.astype(np.float64)
                grid_lat_b[...] = lat_b_vals_f.astype(np.float64)

            if is_source and self.mask_var and self.mask_var in ds:
                grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
                mask_ptr = grid.get_item(
                    esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER
                )
                mask_f = ds[self.mask_var].values.T
                mask_ptr[...] = mask_f.astype(np.int32)
            return grid

    def _generate_weights(self) -> None:
        """Generate regridding weights using ESMPy."""
        src_obj = self._create_esmf_object(self.source_grid_ds, is_source=True)
        dst_obj = self._create_esmf_object(self.target_grid_ds, is_source=False)

        src_field = esmpy.Field(src_obj, name="src")
        dst_field = esmpy.Field(dst_obj, name="dst")

        regrid_kwargs = {
            "regrid_method": self.method_map[self.method],
            "unmapped_action": esmpy.UnmappedAction.IGNORE,
            "factors": True,
        }

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
                "shape_src": self._shape_source,
                "shape_dst": self._shape_target,
                "dims_src": self._dims_source,
                "dims_target": self._dims_target,
                "is_unstructured_src": int(self._is_unstructured_src),
                "is_unstructured_tgt": int(self._is_unstructured_tgt),
                "method": self.method,
                "periodic": int(self.periodic),
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
            self._shape_source = tuple(ds_weights.attrs["shape_src"])
            self._shape_target = tuple(ds_weights.attrs["shape_dst"])
            self._dims_source = tuple(ds_weights.attrs["dims_src"])
            self._dims_target = tuple(ds_weights.attrs["dims_target"])
            self._is_unstructured_src = bool(ds_weights.attrs["is_unstructured_src"])
            self._is_unstructured_tgt = bool(ds_weights.attrs["is_unstructured_tgt"])
            self._loaded_periodic = bool(ds_weights.attrs.get("periodic", False))
            self._loaded_method = ds_weights.attrs.get("method")

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

        def _apply_weights(data_block: np.ndarray) -> np.ndarray:
            """Internal function to apply weight matrix to a data block."""
            original_shape = data_block.shape
            # Core dimensions are at the end
            spatial_shape = original_shape[
                len(original_shape) - len(self._dims_source) :
            ]
            other_dims_shape = original_shape[
                : len(original_shape) - len(self._dims_source)
            ]
            n_spatial = int(np.prod(spatial_shape))
            n_other = int(np.prod(other_dims_shape))
            flat_data = data_block.reshape(n_other, n_spatial)

            if self.skipna:
                mask = np.isnan(flat_data)
                safe_data = np.where(mask, 0.0, flat_data)
                # Optimized CSR application: (matrix @ data.T).T is faster than data @ matrix.T
                result = (self._weights_matrix @ safe_data.T).T
                weights_sum = (self._weights_matrix @ (~mask).astype(float).T).T
                with np.errstate(divide="ignore", invalid="ignore"):
                    final_result = result / weights_sum
                    if self._total_weights is not None:
                        fraction_valid = weights_sum / self._total_weights
                        final_result = np.where(
                            fraction_valid >= (1.0 - self.na_thres - 1e-6),
                            final_result,
                            np.nan,
                        )
                result = final_result
            else:
                # Optimized CSR application: (matrix @ data.T).T is faster than data @ matrix.T
                result = (self._weights_matrix @ flat_data.T).T

            new_shape = other_dims_shape + self._shape_target
            return result.reshape(new_shape)

        input_core_dims = list(self._dims_source)
        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        # Use allow_rechunk=True to support chunked core dimensions
        # and move output_sizes to dask_gufunc_kwargs for future compatibility
        # vectorize=False because _apply_weights handles non-core dimensions
        out = xr.apply_ufunc(
            _apply_weights,
            da_in,
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
            update_history(
                out,
                f"Regridded using Regridder (method={self.method}, "
                f"periodic={self.periodic}, skipna={self.skipna})",
            )

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

        # Update history for provenance
        update_history(
            out,
            f"Regridded Dataset using Regridder (method={self.method}, "
            f"periodic={self.periodic}, skipna={self.skipna})",
        )

        return out
