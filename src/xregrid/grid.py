from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from .utils import _find_coord


def _get_non_spatial_dims(ds: Union[xr.Dataset, xr.DataArray]) -> set[str]:
    """
    Identify dimensions that are likely not spatial (Time, Vertical).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The object to inspect.

    Returns
    -------
    set of str
        Names of non-spatial dimensions.
    """
    non_spatial_dims = set()

    # 1. Use cf-xarray axes
    try:
        # Time axis
        if "T" in ds.cf.axes:
            non_spatial_dims.update(ds.cf.axes["T"])
        # Vertical axis
        if "Z" in ds.cf.axes:
            non_spatial_dims.update(ds.cf.axes["Z"])
    except (KeyError, AttributeError):
        pass

    # 2. Heuristics based on dimension names
    time_names = ["time", "t", "tden", "time_counter", "t_step"]
    vert_names = [
        "lev",
        "level",
        "depth",
        "pressure",
        "sigma",
        "pres",
        "height",
        "altitude",
        "z",
    ]

    for dim in ds.dims:
        dim_lower = str(dim).lower()
        if dim_lower in time_names or dim_lower in vert_names:
            non_spatial_dims.add(str(dim))

        # 3. Dtype check for time if it's a coordinate
        if hasattr(ds, "coords") and dim in ds.coords:
            dtype = ds.coords[dim].dtype
            if np.issubdtype(dtype, np.datetime64) or np.issubdtype(
                dtype, np.timedelta64
            ):
                non_spatial_dims.add(str(dim))

    return non_spatial_dims


def _get_mesh_info(
    ds: xr.Dataset,
    method: Optional[str] = None,
    is_source: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray, Tuple[int, ...], Tuple[str, ...], bool]:
    """
    Detect grid type and extract coordinates and shape from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing spatial coordinates.
    method : str, optional
        Regridding method.
    is_source : bool, default True
        Whether this is the source grid.

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
    """
    # Identify and filter out non-spatial dimensions (Time, Z)
    non_spatial_dims = _get_non_spatial_dims(ds)

    # Handle uxarray objects
    if hasattr(ds, "uxgrid"):
        uxgrid = getattr(ds, "uxgrid")
        try:
            # Check if data variable is on faces
            use_faces = False
            if method == "conservative":
                use_faces = True
            elif hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
                # Find the first data variable that is not a coordinate or topology
                first_var = None
                for v in ds.data_vars.values():
                    if v.attrs.get("cf_role") not in [
                        "mesh_topology",
                        "face_node_connectivity",
                    ]:
                        first_var = v
                        break
                if first_var is not None:
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
                # Apply filtering before returning
                lat_isel = {d: 0 for d in non_spatial_dims if d in lat.dims}
                lon_isel = {d: 0 for d in non_spatial_dims if d in lon.dims}
                if lat_isel:
                    lat = lat.isel(lat_isel, drop=True)
                if lon_isel:
                    lon = lon.isel(lon_isel, drop=True)
                return lon, lat, lat.shape, lat.dims, True
        except (AttributeError, KeyError):
            pass

    # UGRID/unstructured coordinate prioritization
    lat = None
    lon = None

    # 1. First priority: coordinates that match the first data variable's dimensions
    if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
        first_var = None
        for v in ds.data_vars.values():
            if v.attrs.get("cf_role") not in [
                "mesh_topology",
                "face_node_connectivity",
            ]:
                first_var = v
                break
        if first_var is not None:
            for c_name in ds.coords:
                c = ds[c_name]
                if (
                    c.attrs.get("standard_name") == "latitude"
                    or "lat" in c_name.lower()
                ) and set(c.dims).issubset(set(first_var.dims)):
                    lat = c
                    break

    # 2. Second priority: method-based defaults
    if lat is None:
        if method == "conservative":
            # Prefer elements/faces
            for v in ["lat_face", "latCell", "lat_element", "lat"]:
                if v in ds and ("face" in v.lower() or "cell" in v.lower()):
                    lat = ds[v]
                    break
        elif method in ["bilinear", "patch"]:
            # Prefer nodes/vertices for both source and target
            for v in ["lat_node", "lat_vertex", "lat"]:
                if v in ds and ("node" in v.lower() or "vertex" in v.lower()):
                    lat = ds[v]
                    break

    if lat is None:
        lat = _find_coord(ds, "latitude")

    if lat is None:
        if "lat" in ds:
            lat = ds["lat"]
        elif "latCell" in ds:
            lat = ds["latCell"]
        elif "lat_face" in ds:
            lat = ds["lat_face"]
        elif "lat_node" in ds:
            lat = ds["lat_node"]
        elif "latitude" in ds:
            lat = ds["latitude"]

    if lat is None:
        raise KeyError(
            "Could not find latitude coordinates. "
            "Ensure they are named 'lat'/'lon', 'latCell'/'lonCell', "
            "'lat_node'/'lon_node', or have CF attributes."
        )

    # Find matching longitude
    lon_name = lat.name.replace("lat", "lon").replace("LAT", "LON")
    if lon_name in ds:
        lon = ds[lon_name]
    else:
        lon = _find_coord(ds, "longitude")

    if lon is None:
        if "lon" in ds:
            lon = ds["lon"]
        elif "lonCell" in ds:
            lon = ds["lonCell"]
        elif "lon_face" in ds:
            lon = ds["lon_face"]
        elif "lon_node" in ds:
            lon = ds["lon_node"]
        elif "longitude" in ds:
            lon = ds["longitude"]

    if lon is None:
        raise KeyError("Could not find longitude coordinates matching latitude.")

    # Filter out non-spatial dimensions if they are present in lat/lon
    lat_isel = {d: 0 for d in non_spatial_dims if d in lat.dims}
    lon_isel = {d: 0 for d in non_spatial_dims if d in lon.dims}
    if lat_isel:
        lat = lat.isel(lat_isel, drop=True)
    if lon_isel:
        lon = lon.isel(lon_isel, drop=True)

    # UGRID: Check for 'mesh' and 'location' attributes or topology to confirm unstructured
    is_unstructured_fmt = False
    unstructured_dims = {
        "ncol",
        "grid_size",
        "nCells",
        "nVertices",
        "nNodes",
        "nFaces",
        "nEdges",
        "n_node",
        "n_face",
        "n_edge",
        "n_cells",
        "n_vertices",
        "node",
        "face",
        "vertex",
        "cell",
        "n_pts",
    }

    if "mesh" in lat.attrs and "location" in lat.attrs:
        is_unstructured_fmt = True
    elif "mesh" in lon.attrs and "location" in lon.attrs:
        is_unstructured_fmt = True
    elif any(d in lat.dims for d in unstructured_dims) or any(
        d in ds.dims for d in unstructured_dims
    ):
        is_unstructured_fmt = True
    else:
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "mesh_topology":
                is_unstructured_fmt = True
                break

    if lat.ndim == 2:
        # Curvilinear
        if lon.ndim == 2 and lon.dims != lat.dims and set(lon.dims) == set(lat.dims):
            lon = lon.transpose(*lat.dims)
        return lon, lat, lat.shape, lat.dims, False
    elif lat.ndim == 1:
        # Shared 1D dimension => Unstructured
        if lat.dims == lon.dims:
            return lon, lat, lat.shape, lat.dims, True
        elif is_unstructured_fmt:
            # Check if it's really unstructured or just missing one dimension of a rectilinear grid
            if len(lat.dims) == 1 and len(lon.dims) == 1 and lat.dims != lon.dims:
                # Rectilinear path
                lon_mesh, lat_mesh = xr.broadcast(lon, lat)
                lon_mesh = lon_mesh.transpose(lat.dims[0], lon.dims[0])
                lat_mesh = lat_mesh.transpose(lat.dims[0], lon.dims[0])
                return (
                    lon_mesh,
                    lat_mesh,
                    (lat.size, lon.size),
                    (lat.dims[0], lon.dims[0]),
                    False,
                )
            return lon, lat, lat.shape, lat.dims, True
        else:
            # Rectilinear
            lon_mesh, lat_mesh = xr.broadcast(lon, lat)

            # Transpose to (lat, lon) order
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


def _bounds_to_vertices(b: xr.DataArray) -> Union[xr.DataArray, np.ndarray]:
    """
    Convert cell boundary coordinates (bounds) to vertex coordinates for ESMF.

    Supports both 1D and 2D bounds, and 3D curvilinear bounds.
    Backend-agnostic : stays lazy if input is a Dask array.

    Parameters
    ----------
    b : xr.DataArray
        The input boundary coordinate array.

    Returns
    -------
    xr.DataArray or np.ndarray
        The vertex coordinate array.
    """
    if b.ndim == 2 and b.shape[-1] == 2:
        # 1D coordinates with bounds (N, 2) -> (N+1,) vertices
        return xr.concat(
            [
                b.isel({b.dims[-1]: 0}),
                b.isel({b.dims[-1]: 1}).isel({b.dims[0]: slice(-1, None)}),
            ],
            dim=b.dims[0],
        )
    elif b.ndim == 3 and b.shape[-1] == 4:
        # 2D curvilinear bounds (Y, X, 4) -> (Y+1, X+1) vertices
        v0 = b.isel({b.dims[-1]: 0})  # (y, x)
        v1_last_col = b.isel({b.dims[-1]: 1}).isel(
            {b.dims[1]: slice(-1, None)}
        )  # (y, 1)

        row_block = xr.concat([v0, v1_last_col], dim=b.dims[1])  # (y, x+1)

        v3_last_row = b.isel({b.dims[-1]: 3}).isel(
            {b.dims[0]: slice(-1, None)}
        )  # (1, x)
        v2_last_corner = b.isel({b.dims[-1]: 2}).isel(
            {b.dims[0]: slice(-1, None), b.dims[1]: slice(-1, None)}
        )  # (1, 1)

        last_row_block = xr.concat(
            [v3_last_row, v2_last_corner], dim=b.dims[1]
        )  # (1, x+1)

        return xr.concat([row_block, last_row_block], dim=b.dims[0])  # (y+1, x+1)

    return b


def _get_grid_bounds(
    ds: xr.Dataset,
) -> Tuple[
    Optional[Union[xr.DataArray, np.ndarray]], Optional[Union[xr.DataArray, np.ndarray]]
]:
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
    non_spatial_dims = _get_non_spatial_dims(ds)
    try:
        lat_b_da = ds.cf.get_bounds("latitude")
        lon_b_da = ds.cf.get_bounds("longitude")

        # Filter out non-spatial dimensions
        for da in [lat_b_da, lon_b_da]:
            isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
            if isel_dict:
                if da is lat_b_da:
                    lat_b_da = lat_b_da.isel(isel_dict, drop=True)
                elif da is lon_b_da:
                    lon_b_da = lon_b_da.isel(isel_dict, drop=True)

        return _bounds_to_vertices(lat_b_da), _bounds_to_vertices(lon_b_da)
    except (KeyError, AttributeError, ValueError):
        if "lat_b" in ds and "lon_b" in ds:
            return ds["lat_b"], ds["lon_b"]
    return None, None


def _to_degrees(da: xr.DataArray) -> xr.DataArray:
    """
    Convert radians to degrees if necessary.

    Parameters
    ----------
    da : xr.DataArray
        The input coordinate data.

    Returns
    -------
    xr.DataArray
        Data in degrees.
    """
    if da.attrs.get("units") in ["radian", "radians", "rad"]:
        return da * 180.0 / np.pi
    return da


def _clip_latitudes(da: xr.DataArray) -> xr.DataArray:
    """
    Clip latitude values to exactly [-90, 90] to avoid ESMF errors.

    Parameters
    ----------
    da : xr.DataArray
        Latitude coordinate data.

    Returns
    -------
    xr.DataArray
        Clipped latitude data.
    """
    # Use xarray's clip to maintain laziness if dask-backed
    return da.clip(-90.0, 90.0)


def _normalize_longitudes(da: xr.DataArray, lon0: float = 0.0) -> xr.DataArray:
    """
    Normalize longitude values to a specific range (default [0, 360]).

    Parameters
    ----------
    da : xr.DataArray
        Longitude coordinate data.
    lon0 : float, default 0.0
        The start of the 360-degree range.

    Returns
    -------
    xr.DataArray
        Normalized longitude data.
    """
    return (da - lon0) % 360 + lon0


def _get_unstructured_mesh_info(
    ds: xr.Dataset,
    method: str = "conservative",
    is_source: bool = True,
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

    Supports MPAS, UGRID, and SCRIP conventions.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing mesh information.
    method : str, default 'conservative'
        The regridding method.
    is_source : bool, default True
        Whether this is the source grid.

    Returns
    -------
    node_lon : np.ndarray
        Longitude of nodes in degrees [0, 360].
    node_lat : np.ndarray
        Latitude of nodes in degrees [-90, 90].
    element_conn : np.ndarray
        Connectivity of elements (0-based indices).
    element_types : np.ndarray
        Types of elements.
    element_ids : np.ndarray
        Unique IDs for each element.
    orig_cell_index : np.ndarray or None
        Mapping from triangulated elements back to original cell indices.
    """
    import esmpy

    non_spatial_dims = _get_non_spatial_dims(ds)

    # 0. Detect uxarray
    if hasattr(ds, "uxgrid"):
        uxgrid = getattr(ds, "uxgrid")
        try:
            node_lat = _clip_latitudes(_to_degrees(uxgrid.node_lat)).values
            node_lon = _normalize_longitudes(_to_degrees(uxgrid.node_lon)).values
            conn_raw = uxgrid.face_node_connectivity.values
            start_index = uxgrid.face_node_connectivity.attrs.get("start_index", 0)
            fill_value = uxgrid.face_node_connectivity.attrs.get(
                "_FillValue", -9223372036854775808
            )

            # Vectorized triangulation
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
        except (AttributeError, KeyError):
            pass

    # 1. Detect MPAS
    if "verticesOnCell" in ds and "latVertex" in ds and "lonVertex" in ds:
        v_lat = ds["latVertex"]
        v_lon = ds["lonVertex"]
        v_conn = ds["verticesOnCell"]

        for da in [v_lat, v_lon, v_conn]:
            isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
            if isel_dict:
                if da is v_lat:
                    v_lat = v_lat.isel(isel_dict, drop=True)
                elif da is v_lon:
                    v_lon = v_lon.isel(isel_dict, drop=True)
                elif da is v_conn:
                    v_conn = v_conn.isel(isel_dict, drop=True)

        node_lat = _clip_latitudes(_to_degrees(v_lat)).values
        node_lon = _normalize_longitudes(_to_degrees(v_lon)).values
        conn_raw = v_conn.values
        n_edges = (
            ds["nEdgesOnCell"].values
            if "nEdgesOnCell" in ds
            else np.full(ds.sizes["nCells"], conn_raw.shape[1])
        )

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
    mesh_var = None
    for var in ds.variables:
        if ds[var].attrs.get("cf_role") == "mesh_topology":
            mesh_var = var
            break

    conn_var = None
    if mesh_var:
        conn_var = ds[mesh_var].attrs.get("face_node_connectivity")

    if not conn_var:
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "face_node_connectivity":
                conn_var = var
                break

    if not conn_var and "face_node_connectivity" in ds:
        conn_var = "face_node_connectivity"

    if conn_var:
        mesh_name = mesh_var or ds[conn_var].attrs.get("mesh", "")
        node_lon_var = None
        node_lat_var = None

        if hasattr(ds[conn_var], "attrs") and "node_coordinates" in ds[conn_var].attrs:
            node_coords_attr = ds[conn_var].attrs.get("node_coordinates", "").split()
            if len(node_coords_attr) >= 2:
                node_lon_var = node_coords_attr[0]
                node_lat_var = node_coords_attr[1]

        if not node_lon_var and mesh_name and mesh_name in ds:
            node_coords_attr = ds[mesh_name].attrs.get("node_coordinates", "").split()
            if len(node_coords_attr) >= 2:
                node_lon_var = node_coords_attr[0]
                node_lat_var = node_coords_attr[1]

        if not node_lon_var:
            for v in ds.variables:
                if v == conn_var:
                    continue
                attrs = ds[v].attrs
                if attrs.get("standard_name") == "longitude":
                    node_lon_var = v
                if attrs.get("standard_name") == "latitude":
                    node_lat_var = v

        if not node_lon_var:
            if "node_lon" in ds:
                node_lon_var = "node_lon"
            elif "lon_node" in ds:
                node_lon_var = "lon_node"
            if "node_lat" in ds:
                node_lat_var = "node_lat"
            elif "lat_node" in ds:
                node_lat_var = "lat_node"

        if node_lon_var and node_lat_var and node_lon_var in ds and node_lat_var in ds:
            v_lon = ds[node_lon_var]
            v_lat = ds[node_lat_var]
            v_conn = ds[conn_var]

            for da in [v_lon, v_lat, v_conn]:
                isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
                if isel_dict:
                    if da is v_lon:
                        v_lon = v_lon.isel(isel_dict, drop=True)
                    elif da is v_lat:
                        v_lat = v_lat.isel(isel_dict, drop=True)
                    elif da is v_conn:
                        v_conn = v_conn.isel(isel_dict, drop=True)

            node_lon = _normalize_longitudes(_to_degrees(v_lon)).values
            node_lat = _clip_latitudes(_to_degrees(v_lat)).values
            conn_raw = v_conn.values
            start_index = ds[conn_var].attrs.get("start_index", 0)
            fill_value = ds[conn_var].attrs.get("_FillValue", -1)

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

    # 3. Detect SCRIP
    if "lat_b" in ds and "lon_b" in ds and ds["lat_b"].ndim == 2:
        v_lat_b = ds["lat_b"]
        v_lon_b = ds["lon_b"]
        for da in [v_lat_b, v_lon_b]:
            isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
            if isel_dict:
                if da is v_lat_b:
                    v_lat_b = v_lat_b.isel(isel_dict, drop=True)
                elif da is v_lon_b:
                    v_lon_b = v_lon_b.isel(isel_dict, drop=True)

        n_cells, n_corners = v_lat_b.shape
        flat_lat = _clip_latitudes(_to_degrees(v_lat_b)).values.flatten()
        flat_lon = _normalize_longitudes(_to_degrees(v_lon_b)).values.flatten()

        coords = np.column_stack([flat_lon, flat_lat])
        coords_rounded = np.round(coords, 8)
        _, unique_indices, inverse_indices = np.unique(
            coords_rounded, axis=0, return_index=True, return_inverse=True
        )

        node_lon = flat_lon[unique_indices]
        node_lat = flat_lat[unique_indices]
        conn_raw = inverse_indices.reshape(n_cells, n_corners)

        max_tris = n_corners - 2
        v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1)
        v1 = conn_raw[:, 1:-1]
        v2 = conn_raw[:, 2:]

        element_conn = np.stack([v0, v1, v2], axis=2).reshape(-1, 3)
        orig_cell_index = np.repeat(np.arange(n_cells), max_tris)

        n_tris = len(element_conn)
        element_types = np.full(n_tris, esmpy.MeshElemType.TRI, dtype=np.int32)
        element_ids = np.arange(1, n_tris + 1, dtype=np.int32)

        return (
            node_lon,
            node_lat,
            element_conn.flatten().astype(np.int32),
            element_types,
            element_ids,
            orig_cell_index.astype(np.int32),
        )

    # 4. Fallback for LocStreams
    if method in ["nearest_s2d", "nearest_d2s"] or (
        method in ["bilinear", "patch"] and not is_source
    ):
        v_lat = _find_coord(ds, "latitude")
        v_lon = _find_coord(ds, "longitude")

        if v_lat is not None and v_lon is not None:
            if v_lat.dims == v_lon.dims and len(v_lat.dims) == 1:
                for da in [v_lat, v_lon]:
                    isel_dict = {d: 0 for d in non_spatial_dims if d in da.dims}
                    if isel_dict:
                        if da is v_lat:
                            v_lat = v_lat.isel(isel_dict, drop=True)
                        elif da is v_lon:
                            v_lon = v_lon.isel(isel_dict, drop=True)

                node_lat = _clip_latitudes(_to_degrees(v_lat)).values
                node_lon = _normalize_longitudes(_to_degrees(v_lon)).values
                return (
                    node_lon,
                    node_lat,
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    None,
                )

    raise ValueError(
        f"Could not find unstructured mesh connectivity (MPAS or UGRID) for {method} regridding."
    )


def _create_esmf_grid(
    ds: xr.Dataset,
    method: str,
    periodic: bool = False,
    mask_var: Optional[str] = None,
    coord_sys: Any = None,
    is_source: bool = True,
) -> Tuple[Any, list[str], Optional[np.ndarray]]:
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
    coord_sys : Any, optional
        The coordinate system (esmpy.CoordSys).
    is_source : bool, default True
        Whether this is the source grid.

    Returns
    -------
    grid : esmpy.Grid or esmpy.LocStream or esmpy.Mesh
        The created ESMF object.
    provenance : list of str
        Provenance messages.
    orig_idx : np.ndarray or None
        Original cell indices if triangulation was performed.
    """
    import esmpy

    non_spatial_dims = _get_non_spatial_dims(ds)
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(
        ds, method=method, is_source=is_source
    )
    provenance = []
    orig_idx = None

    if is_unstructured:
        if coord_sys is None:
            coord_sys = esmpy.CoordSys.SPH_DEG

        use_mesh = False
        if method == "conservative":
            use_mesh = True
        elif is_source and method in ["bilinear", "patch"]:
            use_mesh = True

        if use_mesh:
            try:
                (
                    node_lon,
                    node_lat,
                    element_conn,
                    element_types,
                    element_ids,
                    orig_idx,
                ) = _get_unstructured_mesh_info(ds, method=method, is_source=is_source)

                if len(element_ids) > 0:
                    if "lat_b" in ds and "lon_b" in ds and ds["lat_b"].ndim == 2:
                        provenance.append(
                            "Derived unstructured mesh connectivity from SCRIP-style bounds."
                        )

                    mesh = esmpy.Mesh(
                        parametric_dim=2,
                        spatial_dim=2,
                        coord_sys=coord_sys,
                    )

                    node_count = len(node_lon)
                    node_ids = np.arange(1, node_count + 1, dtype=np.int32)
                    node_coords = np.column_stack([node_lon, node_lat]).flatten()
                    node_owners = np.zeros(node_count, dtype=np.int32)

                    mesh.add_nodes(
                        node_count,
                        node_ids,
                        node_coords,
                        node_owners,
                    )

                    mask_arg = None
                    if mask_var and mask_var in ds:
                        if method == "conservative":
                            mask_val = ds[mask_var].values
                            element_mask = mask_val[orig_idx].astype(np.int32)
                            mask_arg = element_mask

                    mesh.add_elements(
                        len(element_ids),
                        np.array(element_ids, dtype=np.int32),
                        np.array(element_types, dtype=np.int32),
                        np.array(element_conn, dtype=np.int32),
                        element_mask=mask_arg if method == "conservative" else None,
                    )

                    return mesh, provenance, orig_idx
                else:
                    raise ValueError("No elements found")
            except ValueError:
                if method == "conservative":
                    raise
                if is_source and method in ["bilinear", "patch"]:
                    raise

        if method not in ["nearest_s2d", "nearest_d2s"] and is_source:
            raise NotImplementedError(
                f"Method '{method}' requires connectivity information for unstructured grids. "
            )
        locstream = esmpy.LocStream(shape[0], coord_sys=coord_sys)
        if coord_sys == esmpy.CoordSys.CART:
            locstream["ESMF:X"] = _normalize_longitudes(_to_degrees(lon)).values.astype(
                np.float64
            )
            locstream["ESMF:Y"] = _clip_latitudes(_to_degrees(lat)).values.astype(
                np.float64
            )
        else:
            locstream["ESMF:Lon"] = _normalize_longitudes(
                _to_degrees(lon)
            ).values.astype(np.float64)
            locstream["ESMF:Lat"] = _clip_latitudes(_to_degrees(lat)).values.astype(
                np.float64
            )

        if mask_var and mask_var in ds:
            v_mask = ds[mask_var]
            mask_isel = {d: 0 for d in non_spatial_dims if d in v_mask.dims}
            if mask_isel:
                v_mask = v_mask.isel(mask_isel, drop=True)
            locstream["ESMF:Mask"] = v_mask.values.astype(np.int32)

        return locstream, provenance, None
    else:
        lon_f = _normalize_longitudes(_to_degrees(lon)).values.T
        lat_f = _clip_latitudes(_to_degrees(lat)).values.T
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
                "Conservative regridding requires cell boundaries (bounds)."
            )

        staggerlocs = [esmpy.StaggerLoc.CENTER]
        if has_bounds:
            staggerlocs.append(esmpy.StaggerLoc.CORNER)

        if coord_sys is None:
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
                if not isinstance(lon_b, xr.DataArray):
                    lon_b = xr.DataArray(lon_b)
                if not isinstance(lat_b, xr.DataArray):
                    lat_b = xr.DataArray(lat_b)

                lon_b_vals, lat_b_vals = np.meshgrid(
                    _normalize_longitudes(_to_degrees(lon_b)).values,
                    _clip_latitudes(_to_degrees(lat_b)).values,
                )
            else:
                lon_b_vals = _normalize_longitudes(_to_degrees(lon_b)).values
                lat_b_vals = _clip_latitudes(_to_degrees(lat_b)).values

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
            v_mask = ds[mask_var]
            mask_isel = {d: 0 for d in non_spatial_dims if d in v_mask.dims}
            if mask_isel:
                v_mask = v_mask.isel(mask_isel, drop=True)

            grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
            grid.get_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)[
                ...
            ] = v_mask.values.T.astype(np.int32)
        return grid, provenance, None
