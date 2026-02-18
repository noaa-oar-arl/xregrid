import numpy as np
import xarray as xr
from xregrid.utils import (
    create_global_grid,
    create_regional_grid,
    create_grid_from_crs,
    create_mesh_from_coords,
)


def test_create_global_grid_lazy():
    """
    Aero Protocol: Double-Check Test for create_global_grid.
    Verifies that values are identical between NumPy and Dask backends.
    """
    res_lat, res_lon = 10, 20

    # Eager (NumPy)
    ds_eager = create_global_grid(res_lat=res_lat, res_lon=res_lon, chunks=None)
    assert not ds_eager.chunks

    # Lazy (Dask)
    ds_lazy = create_global_grid(
        res_lat=res_lat, res_lon=res_lon, chunks={"lat": 9, "lon": 9}
    )
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend (lat_b is non-index so it should be chunked)
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_regional_grid_lazy():
    """
    Aero Protocol: Double-Check Test for create_regional_grid.
    """
    lat_range = (-45, 45)
    lon_range = (0, 90)
    res_lat, res_lon = 5, 5

    # Eager (NumPy)
    ds_eager = create_regional_grid(lat_range, lon_range, res_lat, res_lon, chunks=None)

    # Lazy (Dask)
    ds_lazy = create_regional_grid(lat_range, lon_range, res_lat, res_lon, chunks=5)
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_grid_from_crs_lazy():
    """
    Aero Protocol: Double-Check Test for create_grid_from_crs.
    """
    # Test with EPSG:32633 (UTM zone 33N)
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000  # 10km

    # Eager (NumPy)
    ds_eager = create_grid_from_crs("EPSG:32633", extent, res, chunks=None)

    # Lazy (Dask)
    ds_lazy = create_grid_from_crs("EPSG:32633", extent, res, chunks={"x": 5, "y": 5})
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend (lat/lon are non-index here)
    assert hasattr(ds_lazy.lat.data, "dask")


def test_create_mesh_from_coords_lazy():
    """
    Aero Protocol: Double-Check Test for create_mesh_from_coords.
    """
    x = np.array([400000, 450000, 500000])
    y = np.array([5000000, 5050000, 5100000])

    # Eager (NumPy)
    ds_eager = create_mesh_from_coords(x, y, "EPSG:32633", chunks=None)

    # Lazy (Dask)
    ds_lazy = create_mesh_from_coords(x, y, "EPSG:32633", chunks={"n_pts": 2})
    assert ds_lazy.chunks

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend
    assert hasattr(ds_lazy.lat.data, "dask")
