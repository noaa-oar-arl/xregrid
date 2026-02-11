import xarray as xr
from xregrid import (
    create_global_grid,
    create_grid_from_crs,
    create_mesh_from_coords,
    create_regional_grid,
    load_esmf_file,
)
import os
import numpy as np


def test_create_global_grid():
    ds = create_global_grid(res_lat=10, res_lon=20)
    assert "lat" in ds
    assert "lon" in ds
    assert ds.lat.size == 18  # 180 / 10
    assert ds.lon.size == 18  # 360 / 20
    assert "lat_b" in ds
    assert "lon_b" in ds
    # New (N, 2) bounds format
    assert ds.lat_b.shape == (18, 2)
    assert ds.lon_b.shape == (18, 2)
    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert "history" in ds.attrs


def test_create_regional_grid():
    ds = create_regional_grid(
        lat_range=(-45, 45), lon_range=(0, 90), res_lat=5, res_lon=5
    )
    assert ds.lat.size == 18  # 90 / 5
    assert ds.lon.size == 18  # 90 / 5
    assert ds.lat.min() == -42.5
    assert ds.lat.max() == 42.5
    assert "lat_b" in ds
    assert np.isclose(ds.lat_b.min(), -45)
    assert np.isclose(ds.lat_b.max(), 45)


def test_load_esmf_file(tmp_path):
    # Create a dummy NetCDF file
    filepath = os.path.join(tmp_path, "test_mesh.nc")
    ds_orig = xr.Dataset({"test": (("x",), [1, 2, 3])})
    ds_orig.to_netcdf(filepath)

    ds_loaded = load_esmf_file(filepath)
    assert "test" in ds_loaded
    assert "history" in ds_loaded.attrs
    assert "Loaded ESMF file" in ds_loaded.attrs["history"]


def test_create_grid_from_crs():
    # Test with EPSG:32633 (UTM zone 33N)
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000  # 10km
    ds = create_grid_from_crs("EPSG:32633", extent, res)

    assert "lat" in ds
    assert "lon" in ds
    assert "x" in ds
    assert "y" in ds
    assert ds.lat.ndim == 2
    assert ds.x.size == 10
    assert ds.y.size == 10

    assert "lat_b" in ds
    assert "lon_b" in ds
    # New (Y, X, 4) bounds format for curvilinear
    assert ds.lat_b.ndim == 3
    assert ds.lat_b.shape == (10, 10, 4)

    assert "crs" in ds.attrs
    assert "history" in ds.attrs


def test_create_mesh_from_coords():
    x = np.array([400000, 450000, 500000])
    y = np.array([5000000, 5050000, 5100000])
    ds = create_mesh_from_coords(x, y, "EPSG:32633")

    assert "lat" in ds
    assert "lon" in ds
    assert ds.lat.ndim == 1
    assert ds.lon.ndim == 1
    assert ds.lat.size == 3
    assert ds.lat.dims == ds.lon.dims
    assert "n_pts" in ds.lat.dims

    assert "crs" in ds.attrs
