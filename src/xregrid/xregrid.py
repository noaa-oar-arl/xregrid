from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

import esmpy

from .utils import update_history


if TYPE_CHECKING:
    pass


# Global cache for workers to reuse ESMF source objects and weight matrices
_WORKER_CACHE: dict = {}

# Global cache for the driver to store distributed futures
_DRIVER_CACHE: dict = {}


def _setup_worker_cache(key: str, value: Any) -> None:
    """Setup a value in the worker-local cache."""
    global _WORKER_CACHE
    _WORKER_CACHE[key] = value


def _assemble_weights_task(
    results: list[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]],
    n_src: int,
    n_dst: int,
) -> Any:
    """
    Internal worker task to assemble weights from multiple chunks.

    Parameters
    ----------
    results : list of tuples
        List of (rows, cols, data, error) from worker tasks.
    n_src : int
        Total number of source points.
    n_dst : int
        Total number of destination points.

    Returns
    -------
    scipy.sparse.csr_matrix
        The concatenated sparse weight matrix.
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    all_rows, all_cols, all_data = [], [], []
    for r, c, d, err in results:
        if err:
            raise RuntimeError(f"Weight generation error: {err}")
        if len(r) > 0:
            all_rows.append(r)
            all_cols.append(c)
            all_data.append(d)

    if not all_rows:
        return coo_matrix((n_dst, n_src)).tocsr()

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)
    return coo_matrix((data, (rows, cols)), shape=(n_dst, n_src)).tocsr()


def _get_weights_sum_task(matrix: Any) -> np.ndarray:
    """
    Internal worker task to compute the sum of weights for normalization.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The sparse weight matrix.

    Returns
    -------
    np.ndarray
        The sum of weights per destination point.
    """
    import numpy as np

    # Return flattened 1D array (n_dst,) for easier reshaping and broadcasting
    return np.array(matrix.sum(axis=1)).flatten()


def _populate_cache_task(value: Any, key: str) -> None:
    """
    Internal worker task to populate the worker-local cache with a value.

    Parameters
    ----------
    value : Any
        The value to cache (e.g., weight matrix).
    key : str
        The cache key.
    """
    _setup_worker_cache(key, value)


def _get_mesh_info(
    ds: xr.Dataset,
) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
    """
    Detect grid type and extract coordinates and shape from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing spatial coordinates.

    Returns
    -------
    lon : xr.DataArray
        Longitude coordinate array.
    lat : xr.DataArray
        Latitude coordinate array.
    shape : tuple of int
        The spatial shape of the grid.
    dims : tuple of str
        The names of the spatial dimensions.
    is_unstructured : bool
        Whether the grid is unstructured.

    Raises
    ------
    KeyError
        If latitude or longitude coordinates cannot be found.
    ValueError
        If coordinates have invalid dimensionality.
    """
    # Handle uxarray objects
    if hasattr(ds, "uxgrid"):
        uxgrid = getattr(ds, "uxgrid")
        # Prefer coordinates that match data variables if present
        try:
            # Check if data variable is on faces
            use_faces = False
            if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
                first_var = list(ds.data_vars.values())[0]
                if "n_face" in first_var.dims or "nFaces" in first_var.dims:
                    use_faces = True

            if (
                use_faces
                and hasattr(uxgrid, "face_lat")
                and hasattr(uxgrid, "face_lon")
            ):
                lat = uxgrid.face_lat
                lon = uxgrid.face_lon
            else:
                lat = uxgrid.node_lat
                lon = uxgrid.node_lon

            # If they share same dim, it's unstructured
            if lat.dims == lon.dims:
                return lon, lat, lat.shape, lat.dims, True
        except (AttributeError, KeyError):
            pass

    try:
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]
    except (KeyError, AttributeError):
        if "lat" in ds and "lon" in ds:
            lat = ds["lat"]
            lon = ds["lon"]
        elif "latCell" in ds and "lonCell" in ds:
            lat = ds["latCell"]
            lon = ds["lonCell"]
        elif "lat_node" in ds and "lon_node" in ds:
            lat = ds["lat_node"]
            lon = ds["lon_node"]
        elif "latitude" in ds and "longitude" in ds:
            lat = ds["latitude"]
            lon = ds["longitude"]
        else:
            raise KeyError(
                "Could not find latitude/longitude coordinates. "
                "Ensure they are named 'lat'/'lon', 'latCell'/'lonCell', "
                "'lat_node'/'lon_node', or have CF attributes."
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

            # Transpose to standard (lat, lon) order if needed
            # For 1D lat/lon, they are broadcasted to (lat, lon) or (lon, lat)
            # based on order in xr.broadcast. We enforce (lat, lon) here.
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
    """
    Convert cell boundary coordinates (bounds) to vertex coordinates for ESMF.

    Supports both 1D and 2D bounds.

    Parameters
    ----------
    b : xr.DataArray
        The input boundary coordinate array.

    Returns
    -------
    np.ndarray
        The vertex coordinate array.
    """
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
    """
    Extract grid cell boundaries from a dataset using cf-xarray or standard names.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.

    Returns
    -------
    lat_b : np.ndarray or None
        Latitude boundary coordinates.
    lon_b : np.ndarray or None
        Longitude boundary coordinates.
    """
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


def _to_degrees(da: xr.DataArray) -> xr.DataArray:
    """Convert coordinates to degrees if they are in radians."""
    if da.attrs.get("units") in ["radian", "radians", "rad"]:
        return da * 180.0 / np.pi
    return da


def _get_unstructured_mesh_info(
    ds: xr.Dataset,
) -> Tuple[
    np.ndarray,  # node_lon
    np.ndarray,  # node_lat
    np.ndarray,  # element_conn
    np.ndarray,  # element_types
    np.ndarray,  # element_ids
    Optional[np.ndarray],  # orig_cell_index
]:
    """
    Extract unstructured mesh connectivity and vertex info for ESMF Mesh.

    Supports MPAS and UGRID conventions.
    """
    # 0. Detect uxarray
    if hasattr(ds, "uxgrid"):
        uxgrid = getattr(ds, "uxgrid")
        try:
            node_lat = _to_degrees(uxgrid.node_lat).values
            node_lon = _to_degrees(uxgrid.node_lon).values
            conn_raw = uxgrid.face_node_connectivity.values
            start_index = uxgrid.face_node_connectivity.attrs.get("start_index", 0)
            fill_value = uxgrid.face_node_connectivity.attrs.get(
                "_FillValue", -9223372036854775808
            )

            element_conn = []
            element_types = []
            element_ids = []
            orig_cell_index = []

            # Vectorized triangulation (Aero Protocol: Performance)
            n_cells, max_edges = conn_raw.shape
            n_edges = np.sum(conn_raw != fill_value, axis=1)
            max_tris = max_edges - 2

            j = np.arange(1, max_tris + 1)
            mask = j[None, :] < (n_edges[:, None] - 1)

            # Extract v0, v1, v2 for all possible triangles
            v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1) - start_index
            v1 = conn_raw[:, 1:-1] - start_index
            v2 = conn_raw[:, 2:] - start_index

            element_conn = np.stack([v0[mask], v1[mask], v2[mask]], axis=1).flatten()
            orig_cell_index = np.repeat(np.arange(n_cells), max_tris)[mask.flatten()]

            n_tris = len(element_conn) // 3
            element_types = np.full(n_tris, esmpy.MeshElemType.TRI, dtype=np.int32)
            element_ids = np.arange(1, n_tris + 1, dtype=np.int32)

            return (
                node_lon,
                node_lat,
                element_conn.astype(np.int32),
                element_types,
                element_ids,
                orig_cell_index.astype(np.int32),
            )
        except (AttributeError, KeyError):
            pass

    # 1. Detect MPAS
    if "verticesOnCell" in ds and "latVertex" in ds and "lonVertex" in ds:
        node_lat = _to_degrees(ds["latVertex"]).values
        node_lon = _to_degrees(ds["lonVertex"]).values
        conn_raw = ds["verticesOnCell"].values
        n_edges = (
            ds["nEdgesOnCell"].values
            if "nEdgesOnCell" in ds
            else np.full(ds.sizes["nCells"], conn_raw.shape[1])
        )

        element_conn = []
        element_types = []
        element_ids = []
        orig_cell_index = []

        # Vectorized triangulation for MPAS (Aero Protocol: Performance)
        # MPAS is 1-based indexing for vertex IDs
        n_cells, max_edges = conn_raw.shape
        max_tris = max_edges - 2

        j = np.arange(1, max_tris + 1)
        mask = j[None, :] < (n_edges[:, None] - 1)

        v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1) - 1
        v1 = conn_raw[:, 1:-1] - 1
        v2 = conn_raw[:, 2:] - 1

        element_conn = np.stack([v0[mask], v1[mask], v2[mask]], axis=1).flatten()
        orig_cell_index = np.repeat(np.arange(n_cells), max_tris)[mask.flatten()]

        n_tris = len(element_conn) // 3
        element_types = np.full(n_tris, esmpy.MeshElemType.TRI, dtype=np.int32)
        element_ids = np.arange(1, n_tris + 1, dtype=np.int32)

        return (
            node_lon,
            node_lat,
            element_conn.astype(np.int32),
            element_types,
            element_ids,
            orig_cell_index.astype(np.int32),
        )

    # 2. Detect UGRID
    # Look for face_node_connectivity
    conn_var = None
    for var in ds.data_vars:
        if ds[var].attrs.get("cf_role") == "face_node_connectivity":
            conn_var = var
            break

    if not conn_var:
        # Fallback to standard names
        if "face_node_connectivity" in ds:
            conn_var = "face_node_connectivity"

    if conn_var:
        mesh_name = ds[conn_var].attrs.get("mesh", "")
        # Try to find node coordinates
        node_lon_var = None
        node_lat_var = None
        if mesh_name and mesh_name in ds:
            node_coords_attr = ds[mesh_name].attrs.get("node_coordinates", "").split()
            if len(node_coords_attr) >= 2:
                node_lon_var = node_coords_attr[0]
                node_lat_var = node_coords_attr[1]

        if not node_lon_var:
            # Fallback
            node_lon_var = "node_lon" if "node_lon" in ds else "lon_node"
            node_lat_var = "node_lat" if "node_lat" in ds else "lat_node"

        if node_lon_var in ds and node_lat_var in ds:
            node_lon = _to_degrees(ds[node_lon_var]).values
            node_lat = _to_degrees(ds[node_lat_var]).values
            conn_raw = ds[conn_var].values
            start_index = ds[conn_var].attrs.get("start_index", 0)
            fill_value = ds[conn_var].attrs.get("_FillValue", -1)

            element_conn = []
            element_types = []
            element_ids = []
            orig_cell_index = []

            # Vectorized triangulation for UGRID (Aero Protocol: Performance)
            n_cells, max_edges = conn_raw.shape
            n_edges = np.sum(conn_raw != fill_value, axis=1)
            max_tris = max_edges - 2

            j = np.arange(1, max_tris + 1)
            mask = j[None, :] < (n_edges[:, None] - 1)

            v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1) - start_index
            v1 = conn_raw[:, 1:-1] - start_index
            v2 = conn_raw[:, 2:] - start_index

            element_conn = np.stack([v0[mask], v1[mask], v2[mask]], axis=1).flatten()
            orig_cell_index = np.repeat(np.arange(n_cells), max_tris)[mask.flatten()]

            n_tris = len(element_conn) // 3
            element_types = np.full(n_tris, esmpy.MeshElemType.TRI, dtype=np.int32)
            element_ids = np.arange(1, n_tris + 1, dtype=np.int32)

            return (
                node_lon,
                node_lat,
                element_conn.astype(np.int32),
                element_types,
                element_ids,
                orig_cell_index.astype(np.int32),
            )

    raise ValueError(
        "Could not find unstructured mesh connectivity (MPAS or UGRID) for conservative regridding."
    )


def _create_esmf_grid(
    ds: xr.Dataset,
    method: str,
    periodic: bool = False,
    mask_var: Optional[str] = None,
    coord_sys: Optional[esmpy.CoordSys] = None,
) -> Tuple[
    Union[esmpy.Grid, esmpy.LocStream, esmpy.Mesh], list[str], Optional[np.ndarray]
]:
    """
    Create an ESMF Grid or LocStream from an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    method : str
        The regridding method.
    periodic : bool, default False
        Whether the grid is periodic in longitude.
    mask_var : str, optional
        Variable name for the mask.

    Returns
    -------
    grid : esmpy.Grid or esmpy.LocStream
        The created ESMF object.
    provenance : list of str
        A list of provenance messages describing automatic transformations.
    """
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    provenance = []
    orig_idx = None

    if is_unstructured:
        if coord_sys is None:
            coord_sys = esmpy.CoordSys.SPH_DEG if periodic else esmpy.CoordSys.CART

        if method == "conservative":
            # Use Mesh for conservative regridding
            node_lon, node_lat, element_conn, element_types, element_ids, orig_idx = (
                _get_unstructured_mesh_info(ds)
            )

            mesh = esmpy.Mesh(
                parametric_dim=2,
                spatial_dim=2,
                coord_sys=coord_sys,
            )

            node_count = len(node_lon)
            # ESMF/ESMPy typically expects 32-bit integers for IDs and connectivity
            node_ids = np.arange(1, node_count + 1, dtype=np.int32)
            node_coords = np.column_stack([node_lon, node_lat]).flatten()
            node_owners = np.zeros(node_count, dtype=np.int32)  # Single processor

            mesh.add_nodes(
                node_count,
                node_ids.reshape(-1, 1),
                node_coords,  # node_coords must be 1D
                node_owners.reshape(-1, 1),
            )

            mask_arg = None
            if mask_var and mask_var in ds:
                # Map original cell mask to triangulated elements
                mask_val = ds[mask_var].values
                element_mask = mask_val[orig_idx].astype(np.int32)
                mask_arg = element_mask.reshape(-1, 1)

            mesh.add_elements(
                len(element_ids),
                np.array(element_ids, dtype=np.int32).reshape(-1, 1),
                np.array(element_types, dtype=np.int32).reshape(-1, 1),
                np.array(element_conn, dtype=np.int32),
                element_mask=mask_arg,
            )

            return mesh, provenance, orig_idx

        if method not in ["nearest_s2d", "nearest_d2s"]:
            raise NotImplementedError(
                f"Method '{method}' is not yet supported for unstructured grids. "
                "Use 'nearest_s2d', 'nearest_d2s' or 'conservative' (requires mesh info)."
            )
        locstream = esmpy.LocStream(shape[0], coord_sys=coord_sys)
        if coord_sys == esmpy.CoordSys.CART:
            locstream["ESMF:X"] = _to_degrees(lon).values.astype(np.float64)
            locstream["ESMF:Y"] = _to_degrees(lat).values.astype(np.float64)
        else:
            locstream["ESMF:Lon"] = _to_degrees(lon).values.astype(np.float64)
            locstream["ESMF:Lat"] = _to_degrees(lat).values.astype(np.float64)

        if mask_var and mask_var in ds:
            locstream["ESMF:Mask"] = ds[mask_var].values.astype(np.int32)

        return locstream, provenance, None
    else:
        lon_f = _to_degrees(lon).values.T
        lat_f = _to_degrees(lat).values.T
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
                    provenance.append(
                        f"Automatically generated cell boundaries for {method} regridding."
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

        if coord_sys is None:
            # Use CART for regional grids to avoid boundary chord issues (Aero Protocol: Robustness)
            # SPH_DEG is used only for periodic/global grids or unstructured meshes.
            coord_sys = esmpy.CoordSys.SPH_DEG if periodic else esmpy.CoordSys.CART

        grid = esmpy.Grid(
            np.array(shape_f),
            staggerloc=staggerlocs,
            coord_sys=coord_sys,
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
        return grid, provenance, None


def _compute_chunk_weights(
    source_ds: xr.Dataset,
    chunk_ds: xr.Dataset,
    method: str,
    dest_slice_info: Union[np.ndarray, Tuple[int, int, int, int, int]],
    extrap_method: Optional[str] = None,
    extrap_dist_exponent: float = 2.0,
    mask_var: Optional[str] = None,
    periodic: bool = False,
    coord_sys: Optional[esmpy.CoordSys] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """
    Worker function to compute weights for a specific chunk of the target grid.

    Uses worker-local caching for source ESMF objects to avoid redundant initialization.

    Parameters
    ----------
    source_ds : xr.Dataset
        The source grid dataset.
    chunk_ds : xr.Dataset
        A chunk of the target grid dataset.
    method : str
        The regridding method.
    dest_slice_info : np.ndarray or tuple
        If tuple, contains (i0_start, i0_end, i1_start, i1_end, total_size1) to
        reconstruct global indices locally. If ndarray, used directly.
    extrap_method : str, optional
        The extrapolation method.
    extrap_dist_exponent : float, default 2.0
        The IDW extrapolation exponent.
    mask_var : str, optional
        Variable name for the mask.
    periodic : bool, default False
        Whether the grid is periodic in longitude.

    Returns
    -------
    rows : np.ndarray
        Destination indices (0-based, global).
    cols : np.ndarray
        Source indices (0-based).
    data : np.ndarray
        Regridding weights.
    error : str or None
        Error message if weight generation failed.
    """
    try:
        # Initialize Manager if not already done in this process
        esmpy.Manager(debug=False)

        # 1. Get or create source ESMF Field
        # We use id(source_ds) as a key. Dask ensures the same object is reused on a worker.
        src_cache_key = (id(source_ds), method, periodic, mask_var, coord_sys)
        if src_cache_key in _WORKER_CACHE:
            src_field, src_orig_idx = _WORKER_CACHE[src_cache_key]
        else:
            src_obj, _, src_orig_idx = _create_esmf_grid(
                source_ds, method, periodic, mask_var, coord_sys=coord_sys
            )
            if isinstance(src_obj, esmpy.Mesh) and method == "conservative":
                src_field = esmpy.Field(
                    src_obj, name="src", meshloc=esmpy.MeshLoc.ELEMENT
                )
            else:
                src_field = esmpy.Field(src_obj, name="src")
            _WORKER_CACHE[src_cache_key] = (src_field, src_orig_idx)

        # 2. Create target ESMF object (chunk is small, no need to cache)
        dst_obj, _, dst_orig_idx = _create_esmf_grid(
            chunk_ds, method, periodic=False, mask_var=None, coord_sys=coord_sys
        )
        if isinstance(dst_obj, esmpy.Mesh) and method == "conservative":
            dst_field = esmpy.Field(dst_obj, name="dst", meshloc=esmpy.MeshLoc.ELEMENT)
        else:
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

        if mask_var:
            regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

        if method == "conservative":
            regrid_kwargs["norm_type"] = esmpy.NormType.FRACAREA

        # 4. Generate weights
        regrid = esmpy.Regrid(src_field, dst_field, **regrid_kwargs)
        weights = regrid.get_weights_dict(deep_copy=True)

        # 5. Map local destination indices to global grid indices
        if isinstance(dest_slice_info, np.ndarray):
            # Backward compatibility or direct index passing
            global_indices = dest_slice_info
        else:
            # Reconstruct global indices locally to save driver memory (Aero Protocol)
            i0_start, i0_end, i1_start, i1_end, total_size1 = dest_slice_info
            if total_size1 == 0:
                # Unstructured target (1D)
                global_indices = np.arange(i0_start, i0_end)
            else:
                # Structured target (2D)
                n0, n1 = i0_end - i0_start, i1_end - i1_start
                global_indices = (
                    (np.arange(n0)[:, None] + i0_start) * total_size1
                    + (np.arange(n1) + i1_start)
                ).flatten()

        row_dst = weights["row_dst"] - 1
        col_src = weights["col_src"] - 1

        if dst_orig_idx is not None:
            row_dst = dst_orig_idx[row_dst]

        if src_orig_idx is not None:
            col_src = src_orig_idx[col_src]

        rows = global_indices[row_dst]
        cols = col_src
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


def _matmul(matrix: Any, data: np.ndarray) -> np.ndarray:
    """
    Backend-agnostic matrix multiplication (matrix @ data.T).T.

    Handles both NumPy and CuPy backends and ensures a NumPy array is returned.

    Parameters
    ----------
    matrix : Any
        The sparse weight matrix.
    data : np.ndarray
        The dense data array (2D: other x spatial).

    Returns
    -------
    np.ndarray
        The result of (matrix @ data.T).T as a NumPy array.
    """
    res = (matrix @ data.T).T
    if hasattr(res, "get"):
        return res.get()
    return res


def _apply_weights_core(
    data_block: np.ndarray,
    weights_matrix: Any,
    dims_source: Tuple[str, ...],
    shape_target: Tuple[int, ...],
    skipna: bool = False,
    total_weights: Optional[np.ndarray] = None,
    na_thres: float = 1.0,
    weights_key: Optional[str] = None,
) -> np.ndarray:
    """
    Apply regridding weights to a data block (NumPy array).

    Parameters
    ----------
    data_block : np.ndarray
        The input data block. Core dimensions must be at the end.
    weights_matrix : scipy.sparse.csr_matrix or str
        The sparse weight matrix or a string key for worker-local cache.
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
    # Worker-local cache retrieval (backward compatibility or explicit key)
    weights_matrix_key = weights_key
    if isinstance(weights_matrix, str):
        weights_matrix_key = weights_matrix
        weights_matrix = _WORKER_CACHE.get(weights_matrix_key)
        if weights_matrix is None:
            raise RuntimeError(
                f"Weights key '{weights_matrix_key}' not found in worker cache."
            )

    if isinstance(total_weights, str):
        total_weights_key = total_weights
        total_weights = _WORKER_CACHE.get(total_weights_key)
        if total_weights is None:
            raise RuntimeError(
                f"Total weights key '{total_weights_key}' not found in worker cache."
            )

    original_shape = data_block.shape
    # Core dimensions are at the end
    n_source_dims = len(dims_source)
    spatial_shape = original_shape[len(original_shape) - n_source_dims :]
    other_dims_shape = original_shape[: len(original_shape) - n_source_dims]
    n_spatial = int(np.prod(spatial_shape))
    n_other = int(np.prod(other_dims_shape))

    # Optimization: avoid reshape if already 2D and spatial is flat
    if len(original_shape) == 2 and n_other == original_shape[0]:
        flat_data = data_block
    else:
        flat_data = data_block.reshape(n_other, n_spatial)

    if skipna:
        # Use a more memory-efficient NaN detection (Aero Protocol: Performance)
        mask = np.isnan(flat_data)
        has_nans = np.any(mask)

        if not has_nans:
            # Fast path: No NaNs in this data block
            result = _matmul(weights_matrix, flat_data)
            if total_weights is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result /= total_weights
        else:
            # Slow path: Handle NaNs by re-normalizing weights
            is_mask_stationary = True
            if n_other > 1:
                # Optimized stationary mask detection using heuristic early exit
                # (Aero Protocol: Speedup for large grids)
                mask0 = mask[0]
                sample_size = min(1000, n_spatial)
                # Check first sample points across all time steps first
                if not np.all(mask[:, :sample_size] == mask0[:sample_size]):
                    is_mask_stationary = False
                else:
                    for i in range(1, n_other):
                        if not np.array_equal(mask[i], mask0):
                            is_mask_stationary = False
                            break

            zero = flat_data.dtype.type(0)
            if is_mask_stationary:
                # Memory win: Use broadcasting for the stationary mask
                mask = mask[0:1]
                safe_data = np.where(mask, zero, flat_data)
            else:
                safe_data = np.where(mask, zero, flat_data)

            result = _matmul(weights_matrix, safe_data)

            if is_mask_stationary:
                # Optimization: Cache the stationary weights_sum to avoid redundant sparse matmuls
                # across multiple Dask chunks (e.g. different time segments).
                weights_sum = None
                if weights_matrix_key:
                    ws_cache_key = f"ws_{weights_matrix_key}"
                    mask_cache_key = f"mask_{weights_matrix_key}"

                    if ws_cache_key in _WORKER_CACHE:
                        # Validate that the mask is identical to the cached one
                        cached_mask = _WORKER_CACHE.get(mask_cache_key)
                        if np.array_equal(mask[0:1], cached_mask):
                            weights_sum = _WORKER_CACHE[ws_cache_key]

                if weights_sum is None:
                    # Compute normalization only for the first (representative) mask
                    # Use float32 for normalization weights to save memory on large grids
                    valid_mask_single = np.logical_not(mask[0:1]).astype(np.float32)
                    weights_sum = _matmul(weights_matrix, valid_mask_single)

                    if weights_matrix_key:
                        _WORKER_CACHE[ws_cache_key] = weights_sum
                        _WORKER_CACHE[mask_cache_key] = mask[0:1].copy()
            else:
                # Sum weights of valid (non-NaN) points for each slice
                # We use float32 to keep peak memory down for ~1km grids
                valid_mask = np.logical_not(mask).astype(np.float32)
                weights_sum = _matmul(weights_matrix, valid_mask)

            with np.errstate(divide="ignore", invalid="ignore"):
                result /= weights_sum
                if total_weights is not None:
                    fraction_valid = weights_sum / total_weights
                    # Masking of low-confidence points
                    # Ensure NaN value doesn't force promotion to float64
                    nan_val = result.dtype.type(np.nan)
                    result = np.where(
                        fraction_valid < (1.0 - na_thres - 1e-6), nan_val, result
                    )
    else:
        # Standard path (skipna=False): Just apply weights
        result = _matmul(weights_matrix, flat_data)

    new_shape = other_dims_shape + shape_target
    return result.reshape(new_shape).astype(data_block.dtype, copy=False)


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

        # Determine coordinate system for consistency (Aero Protocol: Robustness)
        self._coord_sys = esmpy.CoordSys.SPH_DEG if periodic else esmpy.CoordSys.CART

        # Robust coordinate handling: internally sort coordinates to be ascending
        # to ensure ESMF weight generation is stable and avoid boundary issues.
        # (Aero Protocol: User doesn't have to worry about monotonicity)
        self.source_grid_ds, self._src_was_sorted = self._normalize_grid(source_grid_ds)
        self.target_grid_ds, self._tgt_was_sorted = self._normalize_grid(target_grid_ds)

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
        cls,
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
        """Internally sort rectilinear coordinates to be ascending."""
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
                    lat_vals = ds[lat_dim].values
                    lon_vals = ds[lon_dim].values

                    # Check if already ascending
                    is_lat_asc = np.all(np.diff(lat_vals) > 0)
                    is_lon_asc = np.all(np.diff(lon_vals) > 0)

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
        """
        return _get_mesh_info(ds)

    def _bounds_to_vertices(self, b: xr.DataArray) -> np.ndarray:
        """
        Instance-level wrapper for _bounds_to_vertices.
        """
        return _bounds_to_vertices(b)

    def _get_grid_bounds(
        self, ds: xr.Dataset
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Instance-level wrapper for _get_grid_bounds.
        """
        return _get_grid_bounds(ds)

    def _create_esmf_object(
        self, ds: xr.Dataset, is_source: bool = True
    ) -> Tuple[
        Union[esmpy.Grid, esmpy.LocStream, esmpy.Mesh], list[str], Optional[np.ndarray]
    ]:
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

        if isinstance(src_obj, esmpy.Mesh) and self.method == "conservative":
            src_field = esmpy.Field(src_obj, name="src", meshloc=esmpy.MeshLoc.ELEMENT)
        else:
            src_field = esmpy.Field(src_obj, name="src")

        if isinstance(dst_obj, esmpy.Mesh) and self.method == "conservative":
            dst_field = esmpy.Field(dst_obj, name="dst", meshloc=esmpy.MeshLoc.ELEMENT)
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

        # Map to original indices if Mesh was triangulated
        row_dst = weights["row_dst"] - 1
        col_src = weights["col_src"] - 1

        if dst_orig_idx is not None:
            row_dst = dst_orig_idx[row_dst]
        if src_orig_idx is not None:
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
        if esmpy.local_pet() != 0:
            return  # Only rank 0 saves weights

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
    def weights(self) -> Any:
        """
        Get the sparse weight matrix, gathering it from the cluster if necessary.

        Returns
        -------
        scipy.sparse.csr_matrix
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

        ds = xr.Dataset(
            data_vars={
                "weight_sum": (self._dims_target, weights_sum_2d),
                "unmapped_mask": (self._dims_target, unmapped_2d),
            },
            coords={
                c: self.target_grid_ds.coords[c]
                for c in self.target_grid_ds.coords
                if set(self.target_grid_ds.coords[c].dims).issubset(
                    set(self._dims_target)
                )
            },
        )
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

        # If remote, we must gather for a report (which is typically eager)
        # unless skip_heavy is True and we can get metadata from the Future
        if hasattr(self._weights_matrix, "key") and not skip_heavy:
            # Force gather to compute metrics
            self.weights

        n_src = int(np.prod(self._shape_source))
        n_dst = int(np.prod(self._shape_target))

        # Check if still remote after potential gather
        is_remote = hasattr(self._weights_matrix, "key")

        report = {
            "n_src": n_src,
            "n_dst": n_dst,
            "n_weights": int(self._weights_matrix.nnz) if not is_remote else -1,
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
        _processed_coords: Optional[set[str]] = None,
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
        _processed_coords : set of str, optional
            Set of coordinate names already being processed to avoid infinite recursion.
        skipna : bool, optional
            Whether to handle NaNs. If None, uses initialization default.
        na_thres : float, optional
            NaN threshold. If None, uses initialization default.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        if _processed_coords is None:
            _processed_coords = set()

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
        if da_in.name is not None:
            _processed_coords.add(str(da_in.name))

        for c_name, c_da in da_in.coords.items():
            # Avoid infinite recursion
            if c_name in _processed_coords:
                continue

            if c_name not in da_in.dims and all(
                d in c_da.dims for d in self._dims_source
            ):
                # This is an auxiliary spatial coordinate
                aux_coords_to_regrid[c_name] = self._regrid_dataarray(
                    c_da,
                    update_history_attr=False,
                    _processed_coords=_processed_coords,
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
                    weights_key_arg = f"w_{self._weights_matrix.key}"
                else:
                    weights_key_arg = f"w_{id(self._weights_matrix)}"

                # Check if workers already have this matrix in their cache
                if (client, weights_key_arg) not in _DRIVER_CACHE:
                    if hasattr(self._weights_matrix, "key"):
                        # Truly Distributed: matrix is already a Future on the cluster.
                        # We ensure weights are available on ALL workers (Aero Protocol: Scalability)
                        client.replicate(self._weights_matrix)

                        # Explicitly target all workers to ensure cache consistency
                        workers = list(client.scheduler_info()["workers"].keys())
                        if workers:
                            client.gather(
                                client.map(
                                    _populate_cache_task,
                                    [self._weights_matrix] * len(workers),
                                    [weights_key_arg] * len(workers),
                                    workers=workers,
                                )
                            )
                    else:
                        # Eager matrix: use run (will gather if not careful, but it's local)
                        client.run(
                            _setup_worker_cache, weights_key_arg, self._weights_matrix
                        )
                    _DRIVER_CACHE[(client, weights_key_arg)] = True
                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    if hasattr(self._total_weights, "key"):
                        tw_key = f"tw_{self._total_weights.key}"
                    else:
                        tw_key = f"tw_{id(self._total_weights)}"

                    if (client, tw_key) not in _DRIVER_CACHE:
                        if hasattr(self._total_weights, "key"):
                            client.replicate(self._total_weights)
                            workers = list(client.scheduler_info()["workers"].keys())
                            if workers:
                                client.gather(
                                    client.map(
                                        _populate_cache_task,
                                        [self._total_weights] * len(workers),
                                        [tw_key] * len(workers),
                                        workers=workers,
                                    )
                                )
                        else:
                            client.run(_setup_worker_cache, tw_key, self._total_weights)
                        _DRIVER_CACHE[(client, tw_key)] = True
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

        # Re-attach regridded auxiliary coordinates
        if aux_coords_to_regrid:
            out = out.assign_coords(aux_coords_to_regrid)

        # Update history for provenance
        if update_history_attr:
            esmpy_version = getattr(esmpy, "__version__", "unknown")
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
                # Initialize _processed_coords with the name of the current variable
                # to prevent it from trying to regrid itself if it appears as a coordinate.
                regridded_items[name] = self._regrid_dataarray(
                    da,
                    update_history_attr=False,
                    _processed_coords={name},
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
                                _processed_coords={c},
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

        # Update history for provenance
        esmpy_version = getattr(esmpy, "__version__", "unknown")
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


@xr.register_dataarray_accessor("regrid")
class RegridDataArrayAccessor:
    """
    Xarray Accessor for regridding DataArrays.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def to(self, target_grid: xr.Dataset, **kwargs: Any) -> xr.DataArray:
        """
        Regrid the DataArray to a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        # Convert DataArray to Dataset to ensure compatibility with Regridder
        source_ds = self._obj.to_dataset(name="_tmp_data")
        regridder = Regridder(source_ds, target_grid, **kwargs)
        return regridder(self._obj)


@xr.register_dataset_accessor("regrid")
class RegridDatasetAccessor:
    """
    Xarray Accessor for regridding Datasets.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def to(self, target_grid: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """
        Regrid the Dataset to a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        xr.Dataset
            The regridded Dataset.
        """
        regridder = Regridder(self._obj, target_grid, **kwargs)
        return regridder(self._obj)
