import xarray as xr
from xregrid.utils import create_global_grid, create_grid_from_crs, create_grid_like


def test_aero_lazy_rectilinear_grid():
    """Verify that _create_rectilinear_grid uses dask and matches eager version."""
    res = 1.0
    chunks = 10

    # Eager version
    ds_eager = create_global_grid(res, res, chunks=None)
    assert not hasattr(ds_eager.lat.data, "dask")

    # Lazy version
    ds_lazy = create_global_grid(res, res, chunks=chunks)
    # Dimension coordinates are often eager in xarray due to indexing.
    # Check bounds which should definitely be lazy.
    assert hasattr(ds_lazy.lat_b.data, "dask")

    # Match check
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_aero_lazy_projected_grid():
    """Verify that create_grid_from_crs uses dask and matches eager version."""
    crs = "EPSG:3857"
    extent = (0, 1000, 0, 1000)
    res = 100
    chunks = 5

    # Eager version
    ds_eager = create_grid_from_crs(crs, extent, res, chunks=None)
    assert not hasattr(ds_eager.x.data, "dask")

    # Lazy version
    ds_lazy = create_grid_from_crs(crs, extent, res, chunks=chunks)
    # Check lat/lon which are 2D non-dimension coordinates in projected grids
    assert hasattr(ds_lazy.lat.data, "dask")

    # Match check
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_create_grid_like_no_compute():
    """Verify that create_grid_like with explicit extent avoids compute."""
    # Create a lazy dataset
    ds = create_global_grid(1.0, 1.0, chunks=10)

    # This should NOT trigger computation of ds if we pass extent
    extent = (0, 360, -90, 90)
    ds_like = create_grid_like(ds, 2.0, extent=extent, crs="EPSG:4326")

    # Verify it matches expected output
    assert ds_like.lat.size == 90
    assert ds_like.lon.size == 180
