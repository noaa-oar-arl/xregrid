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

try:
    import dask
except ImportError:
    dask = None

import xarray as xr


def _lazy_arange(
    start: float, stop: float, step: float, chunks: Optional[int] = None
) -> Any:
    """Helper to create a lazy dask range or eager numpy range."""
    if chunks is not None and da is not None:
        return da.arange(start, stop, step, chunks=chunks)
    return np.arange(start, stop, step)


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
    lat_range : Tuple[float, float]
        (min_lat, max_lat).
    lon_range : Tuple[float, float]
        (min_lon, max_lon).
    res_lat : float
        Latitude resolution in degrees.
    res_lon : float
        Longitude resolution in degrees.
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or Dict[str, int], optional
        Chunk sizes for the resulting dask-backed dataset.
    history_msg : str, optional
        Message to add to the history attribute.
    crs : str, default "EPSG:4326"
        CRS identifier.

    Returns
    -------
    xr.Dataset
        The generated grid dataset.
    """
    lat_chunks = chunks.get("lat", -1) if isinstance(chunks, dict) else chunks
    lon_chunks = chunks.get("lon", -1) if isinstance(chunks, dict) else chunks

    lat_arr = _lazy_arange(
        lat_range[0] + res_lat / 2, lat_range[1], res_lat, chunks=lat_chunks
    )
    lon_arr = _lazy_arange(
        lon_range[0] + res_lon / 2, lon_range[1], res_lon, chunks=lon_chunks
    )

    ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lat_arr,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lon_arr,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    if add_bounds:
        # Use CF-compliant (N, 2) bounds.
        # Ensure identical length handling for lazy/eager
        lat_b_1d = _lazy_arange(
            lat_range[0], lat_range[1] + res_lat, res_lat, chunks=lat_chunks
        )[: lat_arr.size + 1]
        lon_b_1d = _lazy_arange(
            lon_range[0], lon_range[1] + res_lon, res_lon, chunks=lon_chunks
        )[: lon_arr.size + 1]

        if chunks is not None and da is not None:
            lat_b_2d = da.stack([lat_b_1d[:-1], lat_b_1d[1:]], axis=1)
            lon_b_2d = da.stack([lon_b_1d[:-1], lon_b_1d[1:]], axis=1)
        else:
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

    # Fallback to common names (including those used in SCRIP, CAM-SE, MUSICA, CAM-fv)
    names = {
        "latitude": [
            "lat",
            "latCell",
            "lat_face",
            "lat_node",
            "latitude",
            "yc",
            "y",
            "LAT",
            "Latitude",
            "grid_center_lat",
        ],
        "longitude": [
            "lon",
            "lonCell",
            "lon_face",
            "lon_node",
            "longitude",
            "xc",
            "x",
            "LON",
            "Longitude",
            "grid_center_lon",
        ],
    }

    # 1. Prioritize dimension coordinates that match fallback names
    for name in names.get(key, []):
        if name in obj.dims and name in obj.coords:
            return obj[name]

    # 2. Check other coordinates and data variables
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
    extent : Tuple[float, float, float, float]
        Grid extent in CRS units: (min_x, max_x, min_y, max_y).
    res : float or Tuple[float, float]
        Grid resolution in CRS units. If float, same resolution in x and y.
        If tuple, (res_x, res_y).
    add_bounds : bool, default True
        Whether to add cell boundary coordinates.
    chunks : int or Dict[str, int], optional
        Chunk sizes for the resulting dask-backed dataset.

    Returns
    -------
    xr.Dataset
        The grid dataset containing 'lat', 'lon' and projected coordinates 'x', 'y'.
    """
    if isinstance(res, (int, float)):
        res_x = res_y = float(res)
    else:
        res_x, res_y = map(float, res)

    x_chunks = chunks.get("x", -1) if isinstance(chunks, dict) else chunks
    y_chunks = chunks.get("y", -1) if isinstance(chunks, dict) else chunks

    # Generate 1D coordinates in projected space
    x = _lazy_arange(extent[0] + res_x / 2, extent[1], res_x, chunks=x_chunks)
    y = _lazy_arange(extent[2] + res_y / 2, extent[3], res_y, chunks=y_chunks)

    x_da = xr.DataArray(x, dims=["x"], name="x")
    y_da = xr.DataArray(y, dims=["y"], name="y")

    # Use xr.broadcast for lazy 2D arrays
    yy_da, xx_da = xr.broadcast(y_da, x_da)

    # Ensure (y, x) order
    yy_da = yy_da.transpose("y", "x")
    xx_da = xx_da.transpose("y", "x")

    if pyproj is None:
        raise ImportError("pyproj is required for create_grid_from_crs.")
    crs_obj = pyproj.CRS(crs)

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

    ds.attrs["crs"] = crs_obj.to_wkt()

    if add_bounds:
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

        x_b_da = xr.DataArray(x_b_raw, dims=["nv", "x"])
        y_b_da = xr.DataArray(y_b_raw, dims=["nv", "y"])

        yy_b_da, xx_b_da = xr.broadcast(y_b_da, x_b_da)

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

        ds.coords["lat_b"] = (
            ["y", "x", "nv"],
            lat_b.data.transpose(1, 2, 0),
            {"units": "degrees_north"},
        )
        ds.coords["lon_b"] = (
            ["y", "x", "nv"],
            lon_b.data.transpose(1, 2, 0),
            {"units": "degrees_east"},
        )
        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

        # Add 1D projected bounds using backend-agnostic xarray operations
        x_da_1d = xr.DataArray(x, dims=["x"])
        y_da_1d = xr.DataArray(y, dims=["y"])

        # Create (N, 2) bounds
        x_b_1d = xr.concat(
            [x_da_1d - res_x / 2, x_da_1d + res_x / 2], dim="nbounds"
        ).transpose("x", "nbounds")
        y_b_1d = xr.concat(
            [y_da_1d - res_y / 2, y_da_1d + res_y / 2], dim="nbounds"
        ).transpose("y", "nbounds")

        ds.coords["x_b"] = (["x", "nbounds"], x_b_1d.data, {"units": units})
        ds.coords["y_b"] = (["y", "nbounds"], y_b_1d.data, {"units": units})
        ds["x"].attrs["bounds"] = "x_b"
        ds["y"].attrs["bounds"] = "y_b"

    update_history(ds, f"Created grid from CRS {crs} using xregrid.")
    if chunks is not None:
        ds = ds.chunk(chunks)
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
    - 3: Mercator
    - 4: Stereographic
    - 5: UTM
    - 6: Polar Stereographic
    - 7: Equatorial Mercator
    - 8: Transverse Mercator
    - 9: Albers Equal Area
    - 10: Lambert Azimuthal Equal Area / Sinusoidal

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
    elif gdtyp == 2:  # Lambert Conformal
        crs = (
            f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 3:  # Mercator
        crs = (
            f"+proj=merc +lat_ts={p_alp} +lon_0={xcent} +lat_0={ycent} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 4:  # Stereographic
        crs = (
            f"+proj=stere +lat_ts={p_alp} +lat_0={ycent} +lon_0={xcent} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 5:  # UTM
        crs = f"+proj=utm +zone={int(p_alp)} +datum=WGS84 +units=m +no_defs"
    elif gdtyp == 6:  # Polar Stereographic
        # lat_0 determined by p_alp (1.0 for North, -1.0 for South)
        lat_0 = 90.0 if p_alp > 0 else -90.0
        crs = (
            f"+proj=stere +lat_0={lat_0} +lat_ts={p_bet} +lon_0={xcent} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 7:  # Equatorial Mercator
        crs = (
            f"+proj=merc +lat_ts={p_alp} +lon_0={xcent} +lat_0=0 "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 8:  # Transverse Mercator
        crs = (
            f"+proj=tmerc +lat_0={ycent} +k={p_bet} +lon_0={xcent} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 9:  # Albers Equal Area
        crs = (
            f"+proj=aea +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    elif gdtyp == 10:  # Lambert Azimuthal Equal Area
        crs = (
            f"+proj=laea +lat_0={ycent} +lon_0={xcent} "
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

    obj_name = getattr(obj, "name", "input")
    history_msg_base = f"Created grid like {obj_name} using xregrid."
    if hasattr(obj, "attrs") and "history" in obj.attrs:
        history_msg_base += f"\nTemplate history:\n{obj.attrs['history']}"

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
                history_msg=history_msg_base + " (Override Extent).",
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
            if dask is not None and (
                hasattr(x_b.data, "dask") or hasattr(y_b.data, "dask")
            ):
                vals = dask.compute(x_b.min(), x_b.max(), y_b.min(), y_b.max())
                extent = tuple(map(float, vals))
            elif hasattr(x_b.data, "dask") or hasattr(y_b.data, "dask"):
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
            if dask is not None and (
                hasattr(x_da.data, "dask") or hasattr(y_da.data, "dask")
            ):
                # Batch everything!
                tasks_dict = {
                    "x_min": x_da.min(),
                    "x_max": x_da.max(),
                    "y_min": y_da.min(),
                    "y_max": y_da.max(),
                }
                if x_da.size > 1:
                    tasks_dict["res_x"] = abs(x_da.diff(x_da.dims[0]).mean())
                if y_da.size > 1:
                    tasks_dict["res_y"] = abs(y_da.diff(y_da.dims[0]).mean())

                results = dask.compute(tasks_dict)[0]
                x_min, x_max, y_min, y_max = (
                    float(results["x_min"]),
                    float(results["x_max"]),
                    float(results["y_min"]),
                    float(results["y_max"]),
                )

                res_x_orig = float(results.get("res_x", 0))
                res_y_orig = float(
                    results.get("res_y", res_x_orig if res_x_orig else 0)
                )

                extent = (
                    x_min - res_x_orig / 2,
                    x_max + res_x_orig / 2,
                    y_min - res_y_orig / 2,
                    y_max + res_y_orig / 2,
                )
            elif hasattr(x_da.data, "dask") or hasattr(y_da.data, "dask"):
                # Non-batched fallback
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

            if dask is not None and (
                hasattr(lat_b.data, "dask") or hasattr(lon_b.data, "dask")
            ):
                vals = dask.compute(lat_b.min(), lat_b.max(), lon_b.min(), lon_b.max())
                lat_range = (float(vals[0]), float(vals[1]))
                lon_range = (float(vals[2]), float(vals[3]))
            elif hasattr(lat_b.data, "dask") or hasattr(lon_b.data, "dask"):
                lat_range = (float(lat_b.min()), float(lat_b.max()))
                lon_range = (float(lon_b.min()), float(lon_b.max()))
            else:
                lat_range = (float(lat_b.min()), float(lat_b.max()))
                lon_range = (float(lon_b.min()), float(lon_b.max()))
        except Exception:
            # Heuristic for resolution to calculate extent from centers
            if dask is not None and (
                hasattr(lat_da.data, "dask") or hasattr(lon_da.data, "dask")
            ):
                tasks_dict = {
                    "lat_min": lat_da.min(),
                    "lat_max": lat_da.max(),
                    "lon_min": lon_da.min(),
                    "lon_max": lon_da.max(),
                }
                if lat_da.size > 1:
                    tasks_dict["res_lat"] = abs(lat_da.diff(lat_da.dims[0]).mean())
                if lon_da.size > 1:
                    tasks_dict["res_lon"] = abs(lon_da.diff(lon_da.dims[-1]).mean())

                results = dask.compute(tasks_dict)[0]
                lat_min, lat_max, lon_min, lon_max = (
                    float(results["lat_min"]),
                    float(results["lat_max"]),
                    float(results["lon_min"]),
                    float(results["lon_max"]),
                )

                res_lat_orig = float(results.get("res_lat", 0))
                res_lon_orig = float(
                    results.get("res_lon", res_lat_orig if res_lat_orig else 0)
                )

                lat_range = (
                    lat_min - res_lat_orig / 2,
                    lat_max + res_lat_orig / 2,
                )
                lon_range = (
                    lon_min - res_lon_orig / 2,
                    lon_max + res_lon_orig / 2,
                )
            elif hasattr(lat_da.data, "dask") or hasattr(lon_da.data, "dask"):
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
            history_msg=history_msg_base,
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
        # Mesh coordinates share the 'n_pts' dimension.
        # If chunks is a dict, we filter for relevant dimensions.
        if isinstance(chunks, dict):
            x_da = x_da.chunk({k: v for k, v in chunks.items() if k in x_da.dims})
            y_da = y_da.chunk({k: v for k, v in chunks.items() if k in y_da.dims})
        else:
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


def spatial_slice(
    obj: Union[xr.DataArray, xr.Dataset],
    extent: Tuple[float, float, float, float],
    crs: Optional[Union[str, int, Any]] = None,
    buffer: float = 0.0,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Slice an xarray object to a spatial extent, handling longitude wrapping.

    This function identifies spatial dimensions via cf-xarray and performs
    a backend-agnostic slice. For geographic coordinates, it robustly
    handles longitude wrapping (e.g., slicing a 0-360 grid with a -20 to 20 extent).

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The input object to slice.
    extent : tuple of float
        Spatial extent as (min_x, max_x, min_y, max_y).
    crs : str, int, or pyproj.CRS, optional
        The CRS of the provided extent. If None, assumes the same CRS as obj.
    buffer : float, default 0.0
        Extra buffer to add around the extent in coordinate units.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The spatially sliced object.

    Notes
    -----
    For longitude wrapping, if the requested extent crosses the grid's
    discontinuity, the result will be concatenated along the longitude dimension.
    """
    # 1. Coordinate and Dimension Discovery
    lat_da = _find_coord(obj, "latitude")
    lon_da = _find_coord(obj, "longitude")

    if lat_da is None or lon_da is None:
        try:
            x_da = obj.cf["projection_x_coordinate"]
            y_da = obj.cf["projection_y_coordinate"]
            is_geographic = False
        except (KeyError, AttributeError):
            raise ValueError(
                "Could not detect spatial coordinates (lat/lon or x/y) for slicing. "
                "Ensure your data has CF-compliant coordinates."
            )
    else:
        x_da, y_da = lon_da, lat_da
        is_geographic = True

    # 2. CRS Transformation
    if crs is not None:
        if pyproj is None:
            raise ImportError(
                "pyproj is required for CRS-aware slicing. "
                "Install it with `pip install pyproj`."
            )
        target_crs = get_crs_info(obj) or pyproj.CRS("EPSG:4326")
        transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)

        # Transform bbox by checking 4 corners
        x_pts = [extent[0], extent[1], extent[1], extent[0]]
        y_pts = [extent[2], extent[2], extent[3], extent[3]]
        xx, yy = transformer.transform(x_pts, y_pts)
        extent = (min(xx), max(xx), min(yy), max(yy))

    min_x, max_x, min_y, max_y = extent
    min_x -= buffer
    max_x += buffer
    min_y -= buffer
    max_y += buffer

    # 3. Y-Slicing (Latitude or Projection Y)
    y_dim = y_da.dims[0]
    if obj.indexes[y_dim].is_monotonic_increasing:
        obj = obj.sel({y_dim: slice(min_y, max_y)})
    else:
        obj = obj.sel({y_dim: slice(max_y, min_y)})

    # 4. X-Slicing (Longitude or Projection X)
    x_dim = x_da.dims[0]
    if not is_geographic:
        # Standard slice for projected coordinates
        if obj.indexes[x_dim].is_monotonic_increasing:
            obj = obj.sel({x_dim: slice(min_x, max_x)})
        else:
            obj = obj.sel({x_dim: slice(max_x, min_x)})
        return obj

    # 5. Longitude Wrapping Logic
    # Get grid convention from eager indexes
    lon_grid = obj.indexes[x_dim]
    g_min = lon_grid.min()

    # Normalize extent to [g_min, g_min + 360]
    norm_min_x = (min_x - g_min) % 360 + g_min
    norm_max_x = (max_x - g_min) % 360 + g_min

    # Detect if we need a wrapped slice
    if norm_min_x > norm_max_x:
        # Crosses the grid boundary
        if lon_grid.is_monotonic_increasing:
            part1 = obj.sel({x_dim: slice(norm_min_x, g_min + 360)})
            part2 = obj.sel({x_dim: slice(g_min, norm_max_x)})
        else:
            part1 = obj.sel({x_dim: slice(g_min + 360, norm_min_x)})
            part2 = obj.sel({x_dim: slice(norm_max_x, g_min)})

        # Concatenate parts
        res = xr.concat([part1, part2], dim=x_dim)
    else:
        # Simple non-wrapped slice
        if lon_grid.is_monotonic_increasing:
            res = obj.sel({x_dim: slice(norm_min_x, norm_max_x)})
        else:
            res = obj.sel({x_dim: slice(norm_max_x, norm_min_x)})

    # Metadata update
    msg = f"Spatially sliced to extent {extent} (wrapped={norm_min_x > norm_max_x})"
    update_history(res, msg)

    return res


def unstructured_to_scrip(ds: xr.Dataset) -> xr.Dataset:
    """
    Canonicalize an unstructured dataset (UGRID or MPAS) to SCRIP format.

    Extracts connectivity information to build explicit boundary coordinates
    (lat_b, lon_b) on a flat 'grid_size' dimension. This enables conservative
    and bilinear regridding for unstructured grids that only provide connectivity.

    Parameters
    ----------
    ds : xr.Dataset
        The input unstructured dataset.

    Returns
    -------
    xr.Dataset
        A CF-compliant SCRIP-style dataset.
    """
    from .grid import _get_unstructured_mesh_info

    # 1. Get centers via _find_coord (robust)
    lat_c = _find_coord(ds, "latitude")
    lon_c = _find_coord(ds, "longitude")

    if lat_c is None or lon_c is None:
        raise ValueError("Could not find latitude/longitude centers in dataset.")

    # 2. Extract connectivity and vertices
    try:
        (
            node_lon,
            node_lat,
            element_conn,
            element_types,
            element_ids,
            orig_cell_index,
        ) = _get_unstructured_mesh_info(ds, method="conservative")
    except Exception as e:
        raise ValueError(f"Failed to extract unstructured connectivity: {e}")

    # 3. Reshape connectivity to SCRIP-style (N, 3 for triangles)
    # _get_unstructured_mesh_info always triangulates.
    n_tris = len(element_conn) // 3
    conn_2d = element_conn.reshape(n_tris, 3)

    # 4. Map nodes to corner coordinates
    lat_b = node_lat[conn_2d]
    lon_b = node_lon[conn_2d]

    # 5. Handle mapping back to original cell centers if we triangulated a polygon grid
    # If the original grid was polygons (MPAS, UGRID faces), we now have n_tris.
    # We should probably map the original centers to the triangles if possible,
    # or just use the triangle centers.
    # For now, we return the triangulated mesh as the primary representation.

    # 3. Ensure attributes are CF-compliant for centers
    lat_attrs = lat_c.attrs.copy()
    lon_attrs = lon_c.attrs.copy()
    if "standard_name" not in lat_attrs:
        lat_attrs["standard_name"] = "latitude"
    if "standard_name" not in lon_attrs:
        lon_attrs["standard_name"] = "longitude"
    if "units" not in lat_attrs:
        lat_attrs["units"] = "degrees_north"
    if "units" not in lon_attrs:
        lon_attrs["units"] = "degrees_east"

    scrip_ds = xr.Dataset(
        coords={
            "lat": (
                ["grid_size"],
                lat_c.data[orig_cell_index]
                if orig_cell_index is not None
                else lat_c.data,
                lat_attrs,
            ),
            "lon": (
                ["grid_size"],
                lon_c.data[orig_cell_index]
                if orig_cell_index is not None
                else lon_c.data,
                lon_attrs,
            ),
            "lat_b": (
                ["grid_size", "nv"],
                lat_b,
                {"units": "degrees_north", "standard_name": "latitude_bounds"},
            ),
            "lon_b": (
                ["grid_size", "nv"],
                lon_b,
                {"units": "degrees_east", "standard_name": "longitude_bounds"},
            ),
        },
        attrs=ds.attrs,
    )

    scrip_ds["lat"].attrs["bounds"] = "lat_b"
    scrip_ds["lon"].attrs["bounds"] = "lon_b"

    update_history(scrip_ds, "Canonicalized unstructured grid to SCRIP-style format.")

    # Scientific Hygiene: add attributes that help regridder identify it as unstructured
    scrip_ds["lat"].attrs["location"] = "face"
    scrip_ds["lon"].attrs["location"] = "face"

    return scrip_ds


def mpas_to_scrip(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert an MPAS-native dataset to a CF-compliant SCRIP-style format.

    Alias for unstructured_to_scrip with MPAS-specific validation.

    Parameters
    ----------
    ds : xr.Dataset
        The MPAS dataset.

    Returns
    -------
    xr.Dataset
        SCRIP-style dataset.
    """
    if "nCells" not in ds.dims:
        raise ValueError("Dataset does not appear to be an MPAS grid (missing nCells).")
    return unstructured_to_scrip(ds)
