from __future__ import annotations

import datetime
import os
import socket
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    import dask.array as da
except ImportError:
    da = None

import xarray as xr


def _create_rectilinear_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
    history_msg: str = "",
    crs: str = "EPSG:4326",
) -> xr.Dataset:
    """
    Internal helper to create rectilinear grids with consistent metadata.

    Parameters
    ----------
    lat_range : tuple of float
        (min_lat, max_lat).
    lon_range : tuple of float
        (min_lon, max_lon).
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
    history_msg : str, optional
        Message to add to the history attribute.

    Returns
    -------
    xr.Dataset
        The generated grid dataset.
    """
    if chunks is not None and da is not None:
        # Convert chunks to integer if it's a dict for 1D arrays
        lat_chunks = chunks.get("lat", -1) if isinstance(chunks, dict) else chunks
        lon_chunks = chunks.get("lon", -1) if isinstance(chunks, dict) else chunks

        lat = da.arange(
            lat_range[0] + res_lat / 2, lat_range[1], res_lat, chunks=lat_chunks
        )
        lon = da.arange(
            lon_range[0] + res_lon / 2, lon_range[1], res_lon, chunks=lon_chunks
        )
    else:
        lat = np.arange(lat_range[0] + res_lat / 2, lat_range[1], res_lat)
        lon = np.arange(lon_range[0] + res_lon / 2, lon_range[1], res_lon)

    ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    if add_bounds:
        # Use CF-compliant (N, 2) bounds.
        if chunks is not None and da is not None:
            lat_b_1d = da.arange(
                lat_range[0], lat_range[1] + res_lat, res_lat, chunks=lat_chunks
            )
            lon_b_1d = da.arange(
                lon_range[0], lon_range[1] + res_lon, res_lon, chunks=lon_chunks
            )

            # Handle potential floating point overshoot from da.arange
            # In dask, we use slicing which is lazy
            lat_b_1d = lat_b_1d[: lat.size + 1]
            lon_b_1d = lon_b_1d[: lon.size + 1]

            lat_b_2d = da.stack([lat_b_1d[:-1], lat_b_1d[1:]], axis=1)
            lon_b_2d = da.stack([lon_b_1d[:-1], lon_b_1d[1:]], axis=1)
        else:
            lat_b_1d = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
            lon_b_1d = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)

            # Handle potential floating point overshoot from np.arange
            if len(lat_b_1d) > len(lat) + 1:
                lat_b_1d = lat_b_1d[: len(lat) + 1]
            if len(lon_b_1d) > len(lon) + 1:
                lon_b_1d = lon_b_1d[: len(lon) + 1]

            lat_b_2d = np.stack([lat_b_1d[:-1], lat_b_1d[1:]], axis=1)
            lon_b_2d = np.stack([lon_b_1d[:-1], lon_b_1d[1:]], axis=1)

        ds.coords["lat_b"] = (
            ["lat", "nv"],
            lat_b_2d,
            {"units": "degrees_north", "standard_name": "latitude_bounds"},
        )
        ds.coords["lon_b"] = (
            ["lon", "nv"],
            lon_b_2d,
            {"units": "degrees_east", "standard_name": "longitude_bounds"},
        )

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    # Aero Protocol: Explicit CRS attribution
    ds.attrs["crs"] = crs

    if history_msg:
        update_history(ds, history_msg)

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


def create_global_grid(
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Create a global rectilinear grid dataset.

    Parameters
    ----------
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
        If None (default), returns an eager NumPy-backed dataset.

    Returns
    -------
    xr.Dataset
        The global grid dataset containing 'lat' and 'lon'.
    """
    return _create_rectilinear_grid(
        lat_range=(-90, 90),
        lon_range=(0, 360),
        res_lat=res_lat,
        res_lon=res_lon,
        add_bounds=add_bounds,
        chunks=chunks,
        history_msg=f"Created global grid ({res_lat}x{res_lon}) using xregrid.",
    )


def create_regional_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Create a regional rectilinear grid dataset.

    Parameters
    ----------
    lat_range : tuple of float
        (min_lat, max_lat).
    lon_range : tuple of float
        (min_lon, max_lon).
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
        If None (default), returns an eager NumPy-backed dataset.

    Returns
    -------
    xr.Dataset
        The regional grid dataset containing 'lat' and 'lon'.
    """
    return _create_rectilinear_grid(
        lat_range=lat_range,
        lon_range=lon_range,
        res_lat=res_lat,
        res_lon=res_lon,
        add_bounds=add_bounds,
        chunks=chunks,
        history_msg=f"Created regional grid ({res_lat}x{res_lon}) using xregrid.",
    )


def load_esmf_file(filepath: str) -> xr.Dataset:
    """
    Load an ESMF mesh, mosaic, or grid file into an xarray Dataset.

    Automatically recognizes SCRIP/ESMF standard variable names and renames
    them to 'lat', 'lon', 'lat_b', 'lon_b' while adding CF attributes.

    Parameters
    ----------
    filepath : str
        Path to the ESMF file.

    Returns
    -------
    xr.Dataset
        The dataset representation of the ESMF file.
    """
    ds = xr.open_dataset(filepath)

    # Recognize SCRIP/ESMF standard names
    rename_map = {
        "grid_center_lat": "lat",
        "grid_center_lon": "lon",
        "grid_corner_lat": "lat_b",
        "grid_corner_lon": "lon_b",
        "grid_imask": "mask",
    }

    found_renames = {k: v for k, v in rename_map.items() if k in ds}

    if found_renames:
        ds = ds.rename(found_renames)
        message = f"Loaded ESMF file and renamed standard variables: {found_renames}"
    else:
        message = f"Loaded ESMF file from {filepath}."

    # Add CF attributes if missing for better cf-xarray discovery
    if "lat" in ds:
        if "units" not in ds["lat"].attrs:
            ds["lat"].attrs["units"] = "degrees_north"
        if "standard_name" not in ds["lat"].attrs:
            ds["lat"].attrs["standard_name"] = "latitude"

    if "lon" in ds:
        if "units" not in ds["lon"].attrs:
            ds["lon"].attrs["units"] = "degrees_east"
        if "standard_name" not in ds["lon"].attrs:
            ds["lon"].attrs["standard_name"] = "longitude"

    # Link bounds if present
    if "lat" in ds and "lat_b" in ds:
        ds["lat"].attrs["bounds"] = "lat_b"
    if "lon" in ds and "lon_b" in ds:
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, message)

    return ds


def get_crs_info(obj: Union[xr.DataArray, xr.Dataset]) -> Optional[Any]:
    """
    Detect CRS information from an xarray object's attributes or encoding.

    Checks for 'grid_mapping', 'crs', and utilizes cf-xarray for robust discovery.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The xarray object to inspect.

    Returns
    -------
    pyproj.CRS, optional
        The detected CRS object, or None if no CRS info is found.
    """
    if pyproj is None or obj is None:
        return None

    # Try to detect CRS from attributes and encoding
    # We prioritize 'grid_mapping' then 'crs'
    crs_info = (
        obj.attrs.get("grid_mapping")
        or obj.encoding.get("grid_mapping")
        or obj.attrs.get("crs")
        or obj.encoding.get("crs")
    )

    # Try cf-xarray for robust grid mapping discovery
    if crs_info is None or isinstance(crs_info, str):
        try:
            # Use cf-xarray to find the grid mapping variable
            # Some versions use get_grid_mapping(), others use grid_mappings property
            gm_var = None
            if hasattr(obj.cf, "get_grid_mapping"):
                gm_var = obj.cf.get_grid_mapping()
            elif hasattr(obj.cf, "grid_mappings"):
                gms = obj.cf.grid_mappings
                if gms:
                    # In newer cf-xarray, grid_mappings returns a list/tuple of GridMapping objects
                    # Each GridMapping object has an 'array' attribute (the DataArray)
                    gm_var = gms[0].array if hasattr(gms[0], "array") else gms[0]

            if gm_var is not None:
                crs_info = (
                    gm_var.attrs.get("crs_wkt")
                    or gm_var.attrs.get("spatial_ref")
                    or gm_var.attrs.get("grid_mapping_name")
                )
        except (AttributeError, KeyError, ImportError):
            pass

    if crs_info:
        try:
            return pyproj.CRS(crs_info)
        except Exception:
            pass

    return None


def _find_coord(
    obj: Union[xr.DataArray, xr.Dataset], key: str
) -> Optional[xr.DataArray]:
    """
    Find a coordinate in an xarray object by CF standard name or common name.

    Prioritizes variables that match the spatial dimensions of the object's
    data variables to resolve ambiguity.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The object to search.
    key : str
        The coordinate type ('latitude' or 'longitude').

    Returns
    -------
    xr.DataArray, optional
        The found coordinate DataArray, or None.
    """
    try:
        return obj.cf[key]
    except (KeyError, AttributeError):
        try:
            # Use cf.coordinates to handle ambiguity
            matches = obj.cf.coordinates.get(key, [])
            if not matches:
                # Also check axes
                matches = obj.cf.axes.get(key, [])

            if matches:
                # Prefer one that matches the object's dimensions (if DataArray)
                # or its data variables' dimensions (if Dataset)
                if isinstance(obj, xr.DataArray):
                    for m in matches:
                        if set(obj[m].dims).issubset(set(obj.dims)):
                            return obj[m]
                elif isinstance(obj, xr.Dataset) and len(obj.data_vars) > 0:
                    for name, da in obj.data_vars.items():
                        if da.attrs.get("cf_role") not in [
                            "mesh_topology",
                            "face_node_connectivity",
                        ]:
                            for m in matches:
                                if set(obj[m].dims).issubset(set(da.dims)):
                                    return obj[m]
                return obj[matches[0]]
        except Exception:
            pass

    # Fallback to common names
    names = {
        "latitude": ["lat", "latCell", "lat_face", "lat_node", "latitude"],
        "longitude": ["lon", "lonCell", "lon_face", "lon_node", "longitude"],
    }

    for name in names.get(key, []):
        if name in obj.coords:
            return obj.coords[name]
        if isinstance(obj, xr.Dataset) and name in obj.data_vars:
            return obj.data_vars[name]

    return None


def update_history(
    obj: Union[xr.DataArray, xr.Dataset], message: str
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Update the 'history' attribute of an xarray object with a timestamped message.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The xarray object to update.
    message : str
        The message to add to the history.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The updated xarray object.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp}: {message}"
    if "history" in obj.attrs:
        obj.attrs["history"] = f"{full_message}\n" + obj.attrs["history"]
    else:
        obj.attrs["history"] = full_message
    return obj


def _transform_coords(
    x_arr: np.ndarray, y_arr: np.ndarray, crs_in: Any, crs_out: str = "EPSG:4326"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform coordinates using pyproj.

    This function is used with xr.apply_ufunc to support both Eager and Lazy backends.

    Parameters
    ----------
    x_arr : np.ndarray
        X coordinates in input CRS.
    y_arr : np.ndarray
        Y coordinates in input CRS.
    crs_in : Any
        Input CRS.
    crs_out : str, default 'EPSG:4326'
        Output CRS.

    Returns
    -------
    lon : np.ndarray
        Longitude coordinates.
    lat : np.ndarray
        Latitude coordinates.
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    return transformer.transform(x_arr, y_arr)


def create_grid_from_crs(
    crs: Union[str, int, Any],
    extent: Tuple[float, float, float, float],
    res: Union[float, Tuple[float, float]],
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Create a structured grid dataset from a CRS and extent.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        The CRS of the grid (Proj4 string, EPSG code, WKT, or CRS object).
    extent : tuple of float
        Grid extent in CRS units: (min_x, max_x, min_y, max_y).
    res : float or tuple of float
        Grid resolution in CRS units. If float, same resolution in x and y.
        If tuple, (res_x, res_y).
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
        If None (default), returns an eager NumPy-backed dataset.

    Returns
    -------
    xr.Dataset
        The grid dataset containing 'lat', 'lon' and projected coordinates 'x', 'y'.
    """
    if isinstance(res, (int, float)):
        res_x = res_y = float(res)
    else:
        res_x, res_y = map(float, res)

    # Generate 1D coordinates in projected space
    if chunks is not None and da is not None:
        # Handle dict or int chunks for 1D arrays
        x_chunks = chunks.get("x", -1) if isinstance(chunks, dict) else chunks
        y_chunks = chunks.get("y", -1) if isinstance(chunks, dict) else chunks

        x = da.arange(extent[0] + res_x / 2, extent[1], res_x, chunks=x_chunks)
        y = da.arange(extent[2] + res_y / 2, extent[3], res_y, chunks=y_chunks)
    else:
        x = np.arange(extent[0] + res_x / 2, extent[1], res_x)
        y = np.arange(extent[2] + res_y / 2, extent[3], res_y)

    x_da = xr.DataArray(x, dims=["x"], name="x")
    y_da = xr.DataArray(y, dims=["y"], name="y")

    # Use xr.broadcast for lazy 2D arrays
    yy_da, xx_da = xr.broadcast(y_da, x_da)

    # Ensure (y, x) order
    yy_da = yy_da.transpose("y", "x")
    xx_da = xx_da.transpose("y", "x")

    # Transform to lat/lon
    if pyproj is None:
        raise ImportError(
            "pyproj is required for create_grid_from_crs. "
            "Install it with `pip install pyproj`."
        )
    crs_obj = pyproj.CRS(crs)

    # Use apply_ufunc with dask='parallelized'
    lon, lat = xr.apply_ufunc(
        _transform_coords,
        xx_da,
        yy_da,
        kwargs={"crs_in": crs_obj},
        dask="parallelized",
        output_dtypes=[float, float],
        input_core_dims=[[], []],
        output_core_dims=[[], []],
    )

    # Try to get units from CRS, default to 'm'
    try:
        units = crs_obj.axis_info[0].unit_name or "m"
    except (IndexError, AttributeError):
        units = "m"

    ds = xr.Dataset(
        coords={
            "y": (
                ["y"],
                y,
                {"units": units, "standard_name": "projection_y_coordinate"},
            ),
            "x": (
                ["x"],
                x,
                {"units": units, "standard_name": "projection_x_coordinate"},
            ),
            "lat": (
                ["y", "x"],
                lat.data,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["y", "x"],
                lon.data,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # Store CRS info
    ds.attrs["crs"] = crs_obj.to_wkt()

    if add_bounds:
        # Create CF-compliant curvilinear bounds (Y, X, 4)
        # This ensures bounds are sliced correctly with centers

        if chunks is not None and da is not None:
            x_b_raw = da.stack(
                [x - res_x / 2, x + res_x / 2, x + res_x / 2, x - res_x / 2]
            )
            y_b_raw = da.stack(
                [y - res_y / 2, y - res_y / 2, y + res_y / 2, y + res_y / 2]
            )
        else:
            x_b_raw = np.stack(
                [x - res_x / 2, x + res_x / 2, x + res_x / 2, x - res_x / 2]
            )
            y_b_raw = np.stack(
                [y - res_y / 2, y - res_y / 2, y + res_y / 2, y + res_y / 2]
            )

        x_b_da = xr.DataArray(x_b_raw, dims=["nv", "x"], name="x_b")
        y_b_da = xr.DataArray(y_b_raw, dims=["nv", "y"], name="y_b")

        # Broadcast them to (nv, y, x)
        yy_b_da, xx_b_da = xr.broadcast(y_b_da, x_b_da)

        # Transform corners lazily
        lon_b, lat_b = xr.apply_ufunc(
            _transform_coords,
            xx_b_da,
            yy_b_da,
            kwargs={"crs_in": crs_obj},
            dask="parallelized",
            output_dtypes=[float, float],
            input_core_dims=[[], []],
            output_core_dims=[[], []],
        )

        # Reshape to (y, x, nv) for CF compliance
        lat_b = lat_b.transpose("y", "x", "nv")
        lon_b = lon_b.transpose("y", "x", "nv")

        ds.coords["lat_b"] = (["y", "x", "nv"], lat_b.data, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["y", "x", "nv"], lon_b.data, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created grid from CRS {crs} using xregrid (Lazy Generation).")

    return ds


def create_grid_from_ioapi(
    metadata: Dict[str, Any],
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Create a structured grid dataset from IOAPI-compliant metadata.

    Supports GDTYP:
    - 1: Lat-Lon
    - 2: Lambert Conformal
    - 5: Polar Stereographic
    - 6: Albers Equal Area
    - 7: Mercator

    Parameters
    ----------
    metadata : dict
        IOAPI metadata containing GDTYP, P_ALP, P_BET, P_GAM, XCENT, YCENT,
        XORIG, YORIG, XCELL, YCELL, NCOLS, NROWS.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.

    Returns
    -------
    xr.Dataset
        The grid dataset.
    """
    gdtyp = metadata["GDTYP"]
    p_alp = metadata["P_ALP"]
    p_bet = metadata["P_BET"]
    xcent = metadata["XCENT"]
    ycent = metadata["YCENT"]
    xorig = metadata["XORIG"]
    yorig = metadata["YORIG"]
    xcell = metadata["XCELL"]
    ycell = metadata["YCELL"]
    ncols = metadata["NCOLS"]
    nrows = metadata["NROWS"]

    if gdtyp == 1:  # Lat-Lon
        crs = "EPSG:4326"
        # In IOAPI Lat-Lon, XORIG/YORIG are degrees, XCELL/YCELL are degrees
    elif gdtyp == 2:  # Lambert Conformal
        crs = (
            f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 5:  # Polar Stereographic
        crs = (
            f"+proj=stere +lat_0={ycent} +lat_ts={p_alp} +lon_0={xcent} "
            f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 6:  # Albers Equal Area
        crs = (
            f"+proj=aea +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 7:  # Mercator
        crs = (
            f"+proj=merc +lat_ts={p_alp} +lon_0={xcent} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    else:
        raise ValueError(f"Unsupported IOAPI GDTYP: {gdtyp}")

    extent = (xorig, xorig + ncols * xcell, yorig, yorig + nrows * ycell)
    res = (xcell, ycell)

    ds = create_grid_from_crs(crs, extent, res, add_bounds=add_bounds, chunks=chunks)

    # Attach IOAPI metadata for provenance
    for k, v in metadata.items():
        ds.attrs[f"ioapi_{k}"] = v

    update_history(ds, f"Created grid from IOAPI metadata (GDTYP={gdtyp})")

    return ds


def create_grid_like(
    obj: Union[xr.DataArray, xr.Dataset],
    res: Union[float, Tuple[float, float]],
    add_bounds: bool = True,
    chunks: Optional[Union[int, Dict[str, int]]] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    crs: Optional[Union[str, int, Any]] = None,
) -> xr.Dataset:
    """
    Create a new grid dataset with the same extent and CRS as an existing object.

    Automatically detects the CRS and spatial extent of the input object.
    Supports both geographic (lat-lon) and projected coordinate systems.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input object to use as a template.
    res : float or tuple of float
        New grid resolution in the coordinate system units.
        If tuple, (res_x, res_y) or (res_lon, res_lat).
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
    extent : tuple of float, optional
        Override the detected extent (min_x, max_x, min_y, max_y).
        Use this to avoid hidden dask.compute() if you already know the extent.
    crs : str, int, or pyproj.CRS, optional
        Override the detected CRS.

    Returns
    -------
    xr.Dataset
        The new grid dataset.
    """
    if crs is not None:
        if pyproj is not None:
            crs_obj = pyproj.CRS(crs)
        else:
            crs_obj = crs
    else:
        crs_obj = get_crs_info(obj)

    if isinstance(res, (int, float)):
        res_x = res_y = float(res)
    else:
        res_x, res_y = map(float, res)

    if extent is not None:
        if crs_obj is None or (
            hasattr(crs_obj, "is_geographic") and crs_obj.is_geographic
        ):
            # Lat-Lon
            return _create_rectilinear_grid(
                (extent[2], extent[3]),  # lat_range
                (extent[0], extent[1]),  # lon_range
                res_y,  # res_lat
                res_x,  # res_lon
                add_bounds=add_bounds,
                chunks=chunks,
                crs=crs_obj.to_wkt() if hasattr(crs_obj, "to_wkt") else "EPSG:4326",
                history_msg=(
                    f"Created grid like {obj.name if hasattr(obj, 'name') else 'input'} "
                    "using xregrid (Override Extent)."
                ),
            )
        else:
            # Projected
            return create_grid_from_crs(
                crs_obj, extent, (res_x, res_y), add_bounds=add_bounds, chunks=chunks
            )

    # 1. Try to find projected coordinates
    try:
        x_da = obj.cf["projection_x_coordinate"]
        y_da = obj.cf["projection_y_coordinate"]

        try:
            # Use bounds for exact extent if available
            x_b = obj.cf.get_bounds("projection_x_coordinate")
            y_b = obj.cf.get_bounds("projection_y_coordinate")

            # Batch compute if lazy to minimize roundtrips
            if hasattr(x_b.data, "dask") or hasattr(y_b.data, "dask"):
                try:
                    import dask

                    vals = dask.compute(x_b.min(), x_b.max(), y_b.min(), y_b.max())
                    extent = tuple(map(float, vals))
                except ImportError:
                    extent = (
                        float(x_b.min()),
                        float(x_b.max()),
                        float(y_b.min()),
                        float(y_b.max()),
                    )
            else:
                extent = (
                    float(x_b.min()),
                    float(x_b.max()),
                    float(y_b.min()),
                    float(y_b.max()),
                )
        except Exception:
            # Fallback to centers
            # Discovery logic: we need min/max and average diff for heuristic
            if hasattr(x_da.data, "dask") or hasattr(y_da.data, "dask"):
                try:
                    import dask

                    # Batch everything!
                    tasks = [x_da.min(), x_da.max(), y_da.min(), y_da.max()]
                    if x_da.size > 1:
                        tasks.append(abs(x_da.diff(x_da.dims[0]).mean()))
                    if y_da.size > 1:
                        tasks.append(abs(y_da.diff(y_da.dims[0]).mean()))

                    results = dask.compute(*tasks)
                    x_min, x_max, y_min, y_max = map(float, results[:4])

                    res_x_orig = float(results[4]) if x_da.size > 1 else 0
                    res_y_orig = (
                        float(results[5])
                        if y_da.size > 1
                        else (res_x_orig if y_da.size == 1 else 0)
                    )

                    extent = (
                        x_min - res_x_orig / 2,
                        x_max + res_x_orig / 2,
                        y_min - res_y_orig / 2,
                        y_max + res_y_orig / 2,
                    )
                except ImportError:
                    # Non-batched fallback
                    res_x_orig = (
                        abs(float(x_da.diff(x_da.dims[0]).mean()))
                        if x_da.size > 1
                        else 0
                    )
                    res_y_orig = (
                        abs(float(y_da.diff(y_da.dims[0]).mean()))
                        if y_da.size > 1
                        else res_x_orig
                    )
                    extent = (
                        float(x_da.min()) - res_x_orig / 2,
                        float(x_da.max()) + res_x_orig / 2,
                        float(y_da.min()) - res_y_orig / 2,
                        float(y_da.max()) + res_y_orig / 2,
                    )
            else:
                res_x_orig = (
                    abs(float(x_da.diff(x_da.dims[0]).mean())) if x_da.size > 1 else 0
                )
                res_y_orig = (
                    abs(float(y_da.diff(y_da.dims[0]).mean()))
                    if y_da.size > 1
                    else res_x_orig
                )
                extent = (
                    float(x_da.min()) - res_x_orig / 2,
                    float(x_da.max()) + res_x_orig / 2,
                    float(y_da.min()) - res_y_orig / 2,
                    float(y_da.max()) + res_y_orig / 2,
                )

        if crs_obj is None:
            # Fallback to generic geographic if no CRS found
            crs_obj = "EPSG:4326"

        return create_grid_from_crs(
            crs_obj, extent, (res_x, res_y), add_bounds=add_bounds, chunks=chunks
        )

    except (KeyError, AttributeError, ValueError):
        pass

    # 2. Fallback to Geographic (Lat-Lon)
    try:
        lat_da = _find_coord(obj, "latitude")
        lon_da = _find_coord(obj, "longitude")
        if lat_da is None or lon_da is None:
            raise KeyError("Coordinates not found")

        try:
            lat_b = obj.cf.get_bounds("latitude")
            lon_b = obj.cf.get_bounds("longitude")

            if hasattr(lat_b.data, "dask") or hasattr(lon_b.data, "dask"):
                try:
                    import dask

                    vals = dask.compute(
                        lat_b.min(), lat_b.max(), lon_b.min(), lon_b.max()
                    )
                    lat_range = (float(vals[0]), float(vals[1]))
                    lon_range = (float(vals[2]), float(vals[3]))
                except ImportError:
                    lat_range = (float(lat_b.min()), float(lat_b.max()))
                    lon_range = (float(lon_b.min()), float(lon_b.max()))
            else:
                lat_range = (float(lat_b.min()), float(lat_b.max()))
                lon_range = (float(lon_b.min()), float(lon_b.max()))
        except Exception:
            # Heuristic for resolution to calculate extent from centers
            if hasattr(lat_da.data, "dask") or hasattr(lon_da.data, "dask"):
                try:
                    import dask

                    tasks = [lat_da.min(), lat_da.max(), lon_da.min(), lon_da.max()]
                    if lat_da.size > 1:
                        tasks.append(abs(lat_da.diff(lat_da.dims[0]).mean()))
                    if lon_da.size > 1:
                        tasks.append(abs(lon_da.diff(lon_da.dims[-1]).mean()))

                    results = dask.compute(*tasks)
                    lat_min, lat_max, lon_min, lon_max = map(float, results[:4])
                    res_lat_orig = float(results[4]) if lat_da.size > 1 else 0
                    res_lon_orig = (
                        float(results[5])
                        if lon_da.size > 1
                        else (res_lat_orig if lon_da.size == 1 else 0)
                    )

                    lat_range = (
                        lat_min - res_lat_orig / 2,
                        lat_max + res_lat_orig / 2,
                    )
                    lon_range = (
                        lon_min - res_lon_orig / 2,
                        lon_max + res_lon_orig / 2,
                    )
                except ImportError:
                    res_lat_orig = (
                        abs(float(lat_da.diff(lat_da.dims[0]).mean()))
                        if lat_da.size > 1
                        else 0
                    )
                    res_lon_orig = (
                        abs(float(lon_da.diff(lon_da.dims[-1]).mean()))
                        if lon_da.size > 1
                        else res_lat_orig
                    )
                    lat_range = (
                        float(lat_da.min()) - res_lat_orig / 2,
                        float(lat_da.max()) + res_lat_orig / 2,
                    )
                    lon_range = (
                        float(lon_da.min()) - res_lon_orig / 2,
                        float(lon_da.max()) + res_lon_orig / 2,
                    )
            else:
                res_lat_orig = (
                    abs(float(lat_da.diff(lat_da.dims[0]).mean()))
                    if lat_da.size > 1
                    else 0
                )
                res_lon_orig = (
                    abs(float(lon_da.diff(lon_da.dims[-1]).mean()))
                    if lon_da.size > 1
                    else res_lat_orig
                )
                lat_range = (
                    float(lat_da.min()) - res_lat_orig / 2,
                    float(lat_da.max()) + res_lat_orig / 2,
                )
                lon_range = (
                    float(lon_da.min()) - res_lon_orig / 2,
                    float(lon_da.max()) + res_lon_orig / 2,
                )

        return _create_rectilinear_grid(
            lat_range,
            lon_range,
            res_y,  # res_lat
            res_x,  # res_lon
            add_bounds=add_bounds,
            chunks=chunks,
            crs=crs_obj.to_wkt() if crs_obj else "EPSG:4326",
            history_msg=(
                f"Created grid like {obj.name if hasattr(obj, 'name') else 'input'} "
                "using xregrid."
            ),
        )
    except (KeyError, AttributeError, ValueError):
        raise ValueError(
            "Could not detect spatial coordinates (latitude/longitude or "
            "projection_x/y) in input object."
        )


def create_mesh_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    crs: Union[str, int, Any],
    chunks: Optional[Union[int, Dict[str, int]]] = None,
) -> xr.Dataset:
    """
    Create an unstructured mesh dataset from coordinates and a CRS.

    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates in CRS units.
    y : np.ndarray
        1D array of y coordinates in CRS units.
    crs : str, int, or pyproj.CRS
        The CRS of the coordinates.
    chunks : int or dict, optional
        Chunk sizes for the resulting dask-backed dataset.
        If None (default), returns an eager NumPy-backed dataset.

    Returns
    -------
    xr.Dataset
        The mesh dataset containing 'lat', 'lon' as 1D arrays sharing a dimension.
    """
    if pyproj is None:
        raise ImportError(
            "pyproj is required for create_mesh_from_coords. "
            "Install it with `pip install pyproj`."
        )
    crs_obj = pyproj.CRS(crs)

    x_da = xr.DataArray(x, dims=["n_pts"], name="x")
    y_da = xr.DataArray(y, dims=["n_pts"], name="y")

    if chunks is not None:
        x_da = x_da.chunk(chunks)
        y_da = y_da.chunk(chunks)

    # Use apply_ufunc with dask='parallelized'
    lon, lat = xr.apply_ufunc(
        _transform_coords,
        x_da,
        y_da,
        kwargs={"crs_in": crs_obj},
        dask="parallelized",
        output_dtypes=[float, float],
        input_core_dims=[[], []],
        output_core_dims=[[], []],
    )

    ds = xr.Dataset(
        coords={
            "lat": (
                ["n_pts"],
                lat.data,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["n_pts"],
                lon.data,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )
    ds.attrs["crs"] = crs_obj.to_wkt()

    update_history(
        ds, f"Created mesh from coordinates and CRS {crs} using xregrid (Lazy)."
    )

    return ds


def get_rdhpcs_cluster(
    machine: Optional[str] = None,
    account: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a dask-jobqueue SLURMCluster for NOAA RDHPCS systems.

    This helper automatically detects the machine if not provided and sets up
    reasonable defaults for Hera, Jet, and Gaea.

    Parameters
    ----------
    machine : str, optional
        Machine name ('hera', 'jet', 'gaea-c5', 'gaea-c6', 'ursa').
        If None, attempts to detect based on hostname.
    account : str, optional
        SLURM account/project for charging.
    **kwargs
        Additional keyword arguments passed to SLURMCluster.

    Returns
    -------
    dask_jobqueue.SLURMCluster
        The configured cluster object.
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError:
        raise ImportError(
            "dask-jobqueue is required for get_rdhpcs_cluster. "
            "Install it with `pip install dask-jobqueue`."
        )

    hostname = socket.gethostname()
    if machine is None:
        if "ufe" in hostname or "ursa" in hostname:
            machine = "ursa"
        elif "hfe" in hostname or "heralogin" in hostname:
            machine = "hera"
        elif "fe" in hostname and "jet" in hostname:
            machine = "jet"
        elif "gaea" in hostname:
            # Hard to distinguish c5/c6 from hostname alone usually
            machine = "gaea-c5"
        else:
            raise ValueError(
                f"Could not detect NOAA RDHPCS machine from hostname '{hostname}'. "
                "Please specify 'machine' explicitly."
            )

    defaults = {
        "account": account or os.environ.get("SACCOUNT"),
        "walltime": "01:00:00",
    }

    if machine == "hera":
        defaults.update(
            {
                "queue": "hera",
                "cores": 40,
                "processes": 40,
                "memory": "160GB",
                "job_extra_directives": ["--exclusive"],
            }
        )
    elif machine == "jet":
        defaults.update(
            {
                "queue": "batch",
                "cores": 24,
                "processes": 12,
                "memory": "120GB",
            }
        )
    elif machine.startswith("gaea"):
        cluster_ver = machine.split("-")[-1] if "-" in machine else "c5"
        cores = 128 if cluster_ver == "c5" else 192
        defaults.update(
            {
                "queue": "batch",
                "cores": cores,
                "processes": 16,
                "memory": "256GB" if cluster_ver == "c5" else "384GB",
                "job_extra_directives": [f"-M {cluster_ver}"],
            }
        )
    elif machine == "ursa":
        defaults.update(
            {
                "queue": "u1-compute",
                "cores": 192,
                "processes": 32,
                "memory": "384GB",
                "job_extra_directives": ["--exclusive"],
            }
        )

    # Override defaults with user kwargs
    defaults.update(kwargs)

    if defaults["account"] is None:
        import warnings

        warnings.warn(
            "No SLURM account specified. Please provide 'account' or set SACCOUNT environment variable."
        )

    return SLURMCluster(**defaults)
