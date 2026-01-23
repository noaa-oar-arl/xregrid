import xarray as xr
import numpy as np
import esmpy
import dask.array as da
import os
from scipy.sparse import coo_matrix

class ESMPyRegridder:
    def __init__(self, source_grid_ds, target_grid_ds, method='bilinear',
                 mask_var=None, reuse_weights=False, filename='weights.nc',
                 skipna=False, na_thres=1.0, periodic=False):
        """
        Improved ESMPyRegridder that correctly handles ESMF dimensions, indexing,
        and supports unstructured grids (e.g., MPAS) and periodicity.

        Parameters
        ----------
        source_grid_ds, target_grid_ds : xr.Dataset
            Contain 'lat' and 'lon'.
        method : str
            Regridding method (bilinear, conservative, nearest_s2d, nearest_d2s, patch).
        mask_var : str, optional
            Variable name for mask (1=valid, 0=masked).
        reuse_weights : bool
            Load weights from filename if it exists.
        filename : str
            Path to weights file.
        skipna : bool
            Handle NaNs in input data by re-normalizing weights.
        na_thres : float
            Threshold for NaN handling.
        periodic : bool
            Whether the grid is periodic in longitude.
        """
        # Initialize ESMF Manager (required for some environments)
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
            'bilinear': esmpy.RegridMethod.BILINEAR,
            'conservative': esmpy.RegridMethod.CONSERVE,
            'nearest_s2d': esmpy.RegridMethod.NEAREST_STOD,
            'nearest_d2s': esmpy.RegridMethod.NEAREST_DTOS,
            'patch': esmpy.RegridMethod.PATCH
        }

        # Internal state
        self._shape_source = None
        self._shape_target = None
        self._dims_source = None
        self._dims_target = None
        self._is_unstructured_src = False
        self._is_unstructured_tgt = False
        self._total_weights = None

        if reuse_weights and os.path.exists(filename):
            self._load_weights()
        else:
            self._generate_weights()
            if reuse_weights:
                self._save_weights()

    def _get_mesh_info(self, ds):
        """Detects grid type and extracts coordinates and shape."""
        lat = ds['lat']
        lon = ds['lon']

        if lat.ndim == 2:
            # Curvilinear
            return lon.values, lat.values, lat.shape, lat.dims, False
        elif lat.ndim == 1:
            if lat.dims == lon.dims:
                # Unstructured (MPAS)
                return lon.values, lat.values, lat.shape, lat.dims, True
            else:
                # Rectilinear
                lon_vals, lat_vals = np.meshgrid(lon.values, lat.values)
                return lon_vals, lat_vals, (lat.size, lon.size), (lat.dims[0], lon.dims[0]), False
        else:
            raise ValueError("lat/lon must be 1D or 2D")

    def _create_esmf_object(self, ds, is_source=True):
        """Creates an ESMF Grid or LocStream."""
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
            locstream = esmpy.LocStream(shape[0], coord_sys=esmpy.CoordSys.SPH_DEG)
            locstream["ESMF:Lon"] = lon.astype(np.float64)
            locstream["ESMF:Lat"] = lat.astype(np.float64)
            return locstream
        else:
            # Transpose to (lon, lat) order for ESMF (SPH_DEG convention)
            lon_f = lon.T
            lat_f = lat.T
            shape_f = lon_f.shape # (nlon, nlat)

            # Periodicity configuration
            num_peri_dims = 1 if self.periodic else None
            periodic_dim = 0 if self.periodic else None
            pole_dim = 1 if self.periodic else None

            grid = esmpy.Grid(
                np.array(shape_f),
                staggerloc=esmpy.StaggerLoc.CENTER,
                coord_sys=esmpy.CoordSys.SPH_DEG,
                num_peri_dims=num_peri_dims,
                periodic_dim=periodic_dim,
                pole_dim=pole_dim
            )

            grid_lon = grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)
            grid_lat = grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)
            grid_lon[...] = lon_f.astype(np.float64)
            grid_lat[...] = lat_f.astype(np.float64)

            if is_source and self.mask_var and self.mask_var in ds:
                grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
                mask_ptr = grid.get_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
                mask_f = ds[self.mask_var].values.T
                mask_ptr[...] = mask_f.astype(np.int32)
            return grid

    def _generate_weights(self):
        src_obj = self._create_esmf_object(self.source_grid_ds, is_source=True)
        dst_obj = self._create_esmf_object(self.target_grid_ds, is_source=False)

        src_field = esmpy.Field(src_obj, name='src')
        dst_field = esmpy.Field(dst_obj, name='dst')

        regrid_kwargs = {
            'regrid_method': self.method_map[self.method],
            'unmapped_action': esmpy.UnmappedAction.IGNORE,
            'factors': True
        }

        if not self._is_unstructured_src and not self._is_unstructured_tgt:
            if self.mask_var and self.mask_var in self.source_grid_ds:
                regrid_kwargs['src_mask_values'] = np.array([0], dtype=np.int32)

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

        rows = weights['row_dst'] - 1
        cols = weights['col_src'] - 1
        data = weights['weights']

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        self._weights_matrix = coo_matrix((data, (rows, cols)), shape=(n_dst, n_src))

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

    def _save_weights(self):
        ds_weights = xr.Dataset(
            data_vars={
                'row': (['n_s'], self._weights_matrix.row + 1),
                'col': (['n_s'], self._weights_matrix.col + 1),
                'S': (['n_s'], self._weights_matrix.data)
            },
            attrs={'n_src': self._weights_matrix.shape[1], 'n_dst': self._weights_matrix.shape[0],
                   'shape_src': self._shape_source, 'shape_dst': self._shape_target,
                   'dims_src': self._dims_source, 'dims_target': self._dims_target,
                   'is_unstructured_src': int(self._is_unstructured_src),
                   'is_unstructured_tgt': int(self._is_unstructured_tgt),
                   'periodic': int(self.periodic)}
        )
        ds_weights.to_netcdf(self.filename)

    def _load_weights(self):
        ds_weights = xr.open_dataset(self.filename)
        rows = ds_weights['row'].values - 1
        cols = ds_weights['col'].values - 1
        data = ds_weights['S'].values
        n_src = ds_weights.attrs['n_src']
        n_dst = ds_weights.attrs['n_dst']
        self._shape_source = tuple(ds_weights.attrs['shape_src'])
        self._shape_target = tuple(ds_weights.attrs['shape_dst'])
        self._dims_source = tuple(ds_weights.attrs['dims_src'])
        self._dims_target = tuple(ds_weights.attrs['dims_target'])
        self._is_unstructured_src = bool(ds_weights.attrs['is_unstructured_src'])
        self._is_unstructured_tgt = bool(ds_weights.attrs['is_unstructured_tgt'])
        self.periodic = bool(ds_weights.attrs.get('periodic', False))
        self._weights_matrix = coo_matrix((data, (rows, cols)), shape=(n_dst, n_src))

        if self.skipna:
            self._total_weights = np.ones((1, n_src)) @ self._weights_matrix.T

    def __call__(self, da_in):
        def _apply_weights(data_block):
            original_shape = data_block.shape
            spatial_shape = original_shape[len(original_shape)-len(self._dims_source):]
            other_dims_shape = original_shape[:len(original_shape)-len(self._dims_source)]
            n_spatial = int(np.prod(spatial_shape))
            n_other = int(np.prod(other_dims_shape))
            flat_data = data_block.reshape(n_other, n_spatial)

            if self.skipna:
                mask = np.isnan(flat_data)
                safe_data = np.where(mask, 0.0, flat_data)
                result = safe_data @ self._weights_matrix.T
                weights_sum = (~mask).astype(float) @ self._weights_matrix.T
                with np.errstate(divide='ignore', invalid='ignore'):
                    final_result = result / weights_sum
                    fraction_valid = weights_sum / self._total_weights
                final_result = np.where(fraction_valid >= (1.0 - self.na_thres - 1e-6), final_result, np.nan)
                result = final_result
            else:
                result = flat_data @ self._weights_matrix.T

            new_shape = other_dims_shape + self._shape_target
            return result.reshape(new_shape)

        input_core_dims = list(self._dims_source)
        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        out = xr.apply_ufunc(
            _apply_weights,
            da_in,
            input_core_dims=[input_core_dims],
            output_core_dims=[temp_output_core_dims],
            output_sizes={d: s for d, s in zip(temp_output_core_dims, self._shape_target)},
            dask='parallelized',
            vectorize=True,
            output_dtypes=[da_in.dtype]
        )

        out = out.rename({temp: orig for temp, orig in zip(temp_output_core_dims, self._dims_target)})

        out = out.assign_coords({c: self.target_grid_ds.coords[c] for c in self.target_grid_ds.coords
                                 if set(self.target_grid_ds.coords[c].dims).issubset(set(self._dims_target))})
        return out
