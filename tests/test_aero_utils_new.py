import xarray as xr
from xregrid.utils import create_grid_like, create_regional_grid, create_grid_from_crs


def test_create_grid_like_latlon():
    """
    Aero Protocol: Double-Check Test for create_grid_like (Lat-Lon).
    Verifies identity between NumPy and Dask backends and preservation of laziness.
    """
    # Create a source grid
    ds_src = create_regional_grid(
        lat_range=(10, 20),
        lon_range=(100, 110),
        res_lat=1.0,
        res_lon=1.0,
        add_bounds=True,
    )

    res_new = 0.5

    # Eager (NumPy)
    ds_eager = create_grid_like(ds_src, res_new, chunks=None)
    assert not ds_eager.chunks
    assert ds_eager.lat.size == 20
    assert ds_eager.lon.size == 20

    # Lazy (Dask)
    ds_src_lazy = ds_src.chunk({"lat": 5, "lon": 5})
    ds_lazy = create_grid_like(ds_src_lazy, res_new, chunks=5)

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify laziness: lat_b should be a dask array
    assert hasattr(ds_lazy.lat_b.data, "dask")


def test_create_grid_like_projected():
    """
    Aero Protocol: Double-Check Test for create_grid_like (Projected).
    """
    # UTM zone 33N
    crs = "EPSG:32633"
    extent = (400000, 500000, 5000000, 5100000)
    res_orig = 10000

    ds_src = create_grid_from_crs(crs, extent, res_orig, add_bounds=True)

    res_new = 5000

    # Eager
    ds_eager = create_grid_like(ds_src, res_new, chunks=None)

    # Lazy
    ds_src_lazy = ds_src.chunk({"x": 5, "y": 5})
    ds_lazy = create_grid_like(ds_src_lazy, res_new, chunks=5)

    # Assert values
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify laziness (lat/lon are non-index in projected grid)
    assert hasattr(ds_lazy.lat.data, "dask")
    assert ds_lazy.attrs["crs"] == ds_src.attrs["crs"]


def test_rectilinear_hygiene():
    """
    Verify that _create_rectilinear_grid produces high-hygiene metadata.
    """
    ds = create_regional_grid((0, 10), (0, 10), 1, 1)

    assert ds.attrs["crs"] == "EPSG:4326"

    # Test custom CRS
    # create_regional_grid currently doesn't expose crs, let's test _create_rectilinear_grid directly
    from xregrid.utils import _create_rectilinear_grid

    ds_nad83 = _create_rectilinear_grid((0, 10), (0, 10), 1, 1, crs="EPSG:4269")
    assert ds_nad83.attrs["crs"] == "EPSG:4269"

    assert ds.lat.attrs["standard_name"] == "latitude"
    assert ds.lon.attrs["standard_name"] == "longitude"
    assert ds.lat_b.attrs["standard_name"] == "latitude_bounds"
    assert ds.lon_b.attrs["standard_name"] == "longitude_bounds"
    assert "history" in ds.attrs
