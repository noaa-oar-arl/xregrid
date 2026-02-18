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
import xarray as xr


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
    lat = np.arange(-90 + res_lat / 2, 90, res_lat)
    lon = np.arange(0 + res_lon / 2, 360, res_lon)

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
        # Use CF-compliant (N, 2) bounds that share dimensions with the grid.
        # This ensures they are correctly subsetted and sorted by xarray.
        # (Aero Protocol: Dask Efficiency & Robustness)
        lat_b_1d = np.arange(-90, 90 + res_lat, res_lat)
        lon_b_1d = np.arange(0, 360 + res_lon, res_lon)

        lat_b_2d = np.stack([lat_b_1d[:-1], lat_b_1d[1:]], axis=1)
        lon_b_2d = np.stack([lon_b_1d[:-1], lon_b_1d[1:]], axis=1)

        ds.coords["lat_b"] = (["lat", "nv"], lat_b_2d, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon", "nv"], lon_b_2d, {"units": "degrees_east"})

        # Link bounds using cf-xarray convention
        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created global grid ({res_lat}x{res_lon}) using xregrid.")

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


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
        lat_b_1d = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
        lon_b_1d = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)

        lat_b_2d = np.stack([lat_b_1d[:-1], lat_b_1d[1:]], axis=1)
        lon_b_2d = np.stack([lon_b_1d[:-1], lon_b_1d[1:]], axis=1)

        ds.coords["lat_b"] = (["lat", "nv"], lat_b_2d, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon", "nv"], lon_b_2d, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created regional grid ({res_lat}x{res_lon}) using xregrid.")

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


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
    x = np.arange(extent[0] + res_x / 2, extent[1], res_x)
    y = np.arange(extent[2] + res_y / 2, extent[3], res_y)

    xx, yy = np.meshgrid(x, y)

    # Transform to lat/lon
    if pyproj is None:
        raise ImportError(
            "pyproj is required for create_grid_from_crs. "
            "Install it with `pip install pyproj`."
        )
    crs_obj = pyproj.CRS(crs)
    transformer = pyproj.Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)

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
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["y", "x"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # Store CRS info
    ds.attrs["crs"] = crs_obj.to_wkt()

    if add_bounds:
        # Create CF-compliant curvilinear bounds (Y, X, 4)
        # This ensures bounds are sliced correctly with centers (Aero Protocol: Scientific Hygiene)

        # (4, Y, X)
        yy_corners, xx_corners = np.meshgrid(y, x, indexing="ij")
        # We need to broadcast the corners to (4, Y, X)
        xx_b = np.stack(
            [
                np.broadcast_to(x - res_x / 2, (len(y), len(x))),
                np.broadcast_to(x + res_x / 2, (len(y), len(x))),
                np.broadcast_to(x + res_x / 2, (len(y), len(x))),
                np.broadcast_to(x - res_x / 2, (len(y), len(x))),
            ]
        )
        yy_b = np.stack(
            [
                np.broadcast_to(y - res_y / 2, (len(x), len(y))).T,
                np.broadcast_to(y - res_y / 2, (len(x), len(y))).T,
                np.broadcast_to(y + res_y / 2, (len(x), len(y))).T,
                np.broadcast_to(y + res_y / 2, (len(x), len(y))).T,
            ]
        )

        lon_b, lat_b = transformer.transform(xx_b, yy_b)
        # Reshape to (Y, X, 4)
        lat_b = np.moveaxis(lat_b, 0, -1)
        lon_b = np.moveaxis(lon_b, 0, -1)

        ds.coords["lat_b"] = (["y", "x", "nv"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["y", "x", "nv"], lon_b, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created grid from CRS {crs} using xregrid.")

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


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
    transformer = pyproj.Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)

    ds = xr.Dataset(
        coords={
            "lat": (
                ["n_pts"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["n_pts"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )
    ds.attrs["crs"] = crs_obj.to_wkt()

    update_history(ds, f"Created mesh from coordinates and CRS {crs} using xregrid.")

    if chunks is not None:
        ds = ds.chunk(chunks)

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
