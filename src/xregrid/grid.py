from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr


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


def _bounds_to_vertices(b: xr.DataArray) -> Union[xr.DataArray, np.ndarray]:
    """
    Convert cell boundary coordinates (bounds) to vertex coordinates for ESMF.

    Supports both 1D and 2D bounds, and 3D curvilinear bounds.
    Backend-agnostic (Aero Protocol): stays lazy if input is a Dask array.

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
    try:
        lat_b_da = ds.cf.get_bounds("latitude")
        lon_b_da = ds.cf.get_bounds("longitude")
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

    ESMF can fail with ESMF_RC_VAL_OUTOFRANGE if latitudes are even slightly
    beyond 90 degrees due to floating point precision.

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

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing mesh information.
    method : str
        The regridding method.
    mask_var : str, optional
        The variable name for the mask.

    Returns
    -------
    node_lon : np.ndarray
        Longitude of nodes.
    node_lat : np.ndarray
        Latitude of nodes.
    element_conn : np.ndarray
        Connectivity of elements.
    element_types : np.ndarray
        Types of elements (e.g. TRI).
    element_ids : np.ndarray
        IDs of elements.
    orig_cell_index : np.ndarray, optional
        Mapping back to original cell indices.
    """
    import esmpy

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
        node_lat = _clip_latitudes(_to_degrees(ds["latVertex"])).values
        node_lon = _normalize_longitudes(_to_degrees(ds["lonVertex"])).values
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
            node_lon = _normalize_longitudes(_to_degrees(ds[node_lon_var])).values
            node_lat = _clip_latitudes(_to_degrees(ds[node_lat_var])).values
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
    coord_sys: Any = None,
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

    Returns
    -------
    grid : esmpy.Grid or esmpy.LocStream or esmpy.Mesh
        The created ESMF object.
    provenance : list of str
        A list of provenance messages describing automatic transformations.
    orig_idx : np.ndarray or None
        Original cell indices if triangulation was performed.
    """
    import esmpy

    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    provenance = []
    orig_idx = None

    if is_unstructured:
        if coord_sys is None:
            coord_sys = esmpy.CoordSys.SPH_DEG if periodic else esmpy.CoordSys.CART

        # Attempt Mesh creation for methods that support it on unstructured grids
        if method in ["conservative", "bilinear", "patch"]:
            try:
                (
                    node_lon,
                    node_lat,
                    element_conn,
                    element_types,
                    element_ids,
                    orig_idx,
                ) = _get_unstructured_mesh_info(ds)

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
                    node_ids,
                    node_coords,  # node_coords must be 1D
                    node_owners,
                )

                mask_arg = None
                if mask_var and mask_var in ds:
                    if method == "conservative":
                        # Map original cell mask to triangulated elements
                        mask_val = ds[mask_var].values
                        element_mask = mask_val[orig_idx].astype(np.int32)
                        mask_arg = element_mask
                    else:
                        # For bilinear/patch, mask is on nodes (handled via Field later)
                        pass

                mesh.add_elements(
                    len(element_ids),
                    np.array(element_ids, dtype=np.int32),
                    np.array(element_types, dtype=np.int32),
                    np.array(element_conn, dtype=np.int32),
                    element_mask=mask_arg if method == "conservative" else None,
                )

                return mesh, provenance, orig_idx
            except ValueError:
                if method == "conservative":
                    raise
                # Fall through to LocStream or NotImplementedError

        if method not in ["nearest_s2d", "nearest_d2s", "bilinear", "patch"]:
            raise NotImplementedError(
                f"Method '{method}' is not yet supported for unstructured grids without connectivity info. "
                "Use 'nearest_s2d', 'nearest_d2s', 'bilinear' or 'patch' (as target) or ensure your dataset has UGRID/MPAS mesh info for 'conservative'."
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
            locstream["ESMF:Mask"] = ds[mask_var].values.astype(np.int32)

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
                # Need to convert to DataArray if they are just numpy arrays
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
            grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)
            grid.get_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER)[
                ...
            ] = ds[mask_var].values.T.astype(np.int32)
        return grid, provenance, None
