from __future__ import annotations

import datetime
from typing import Tuple, Union

import numpy as np
import pyproj
import xarray as xr


def create_global_grid(
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
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
        lat_b = np.arange(-90, 90 + res_lat, res_lat)
        lon_b = np.arange(0, 360 + res_lon, res_lon)
        ds.coords["lat_b"] = (["lat_b"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon_b"], lon_b, {"units": "degrees_east"})

        # Link bounds using cf-xarray convention
        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created global grid ({res_lat}x{res_lon}) using xregrid.")

    return ds


def create_regional_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    res_lat: float,
    res_lon: float,
    add_bounds: bool = True,
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
        lat_b = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
        lon_b = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)
        ds.coords["lat_b"] = (["lat_b"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["lon_b"], lon_b, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created regional grid ({res_lat}x{res_lon}) using xregrid.")

    return ds


def load_esmf_file(filepath: str) -> xr.Dataset:
    """
    Load an ESMF mesh, mosaic, or grid file into an xarray Dataset.

    Parameters
    ----------
    filepath : str
        Path to the ESMF file.

    Returns
    -------
    xr.Dataset
        The dataset representation of the ESMF file.
    """
    # This is a basic implementation using xarray to open NetCDF files
    # which many ESMF files are.
    ds = xr.open_dataset(filepath)

    update_history(ds, f"Loaded ESMF file from {filepath}.")

    return ds


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
    crs: Union[str, int, pyproj.CRS],
    extent: Tuple[float, float, float, float],
    res: Union[float, Tuple[float, float]],
    add_bounds: bool = True,
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
        # Derive bounds from center points to ensure alignment
        x_b = np.concatenate([[x[0] - res_x / 2], x + res_x / 2])
        y_b = np.concatenate([[y[0] - res_y / 2], y + res_y / 2])

        xx_b, yy_b = np.meshgrid(x_b, y_b)
        lon_b, lat_b = transformer.transform(xx_b, yy_b)

        ds.coords["lat_b"] = (["y_b", "x_b"], lat_b, {"units": "degrees_north"})
        ds.coords["lon_b"] = (["y_b", "x_b"], lon_b, {"units": "degrees_east"})

        ds["lat"].attrs["bounds"] = "lat_b"
        ds["lon"].attrs["bounds"] = "lon_b"

    update_history(ds, f"Created grid from CRS {crs} using xregrid.")

    return ds


def create_mesh_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    crs: Union[str, int, pyproj.CRS],
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

    Returns
    -------
    xr.Dataset
        The mesh dataset containing 'lat', 'lon' as 1D arrays sharing a dimension.
    """
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

    return ds
