import numpy as np
import pytest
from xregrid.utils import spatial_slice, create_global_grid


def test_spatial_slice_basic():
    """Test standard slicing on a NumPy-backed Dataset."""
    ds = create_global_grid(1.0, 1.0)
    ds["data"] = (["lat", "lon"], np.random.rand(180, 360))

    # Slice a region in the middle
    extent = (10, 30, 10, 30)
    sliced = spatial_slice(ds, extent)

    assert sliced.lat.min() >= 10
    assert sliced.lat.max() <= 30
    assert sliced.lon.min() >= 10
    assert sliced.lon.max() <= 30
    assert not hasattr(sliced.data.data, "dask")


def test_spatial_slice_dask():
    """Test slicing on a Dask-backed Dataset verifying laziness."""
    ds = create_global_grid(1.0, 1.0, chunks={"lat": 90, "lon": 90})
    ds["data"] = (["lat", "lon"], np.random.rand(180, 360))
    ds = ds.chunk({"lat": 90, "lon": 90})

    extent = (10, 30, 10, 30)
    sliced = spatial_slice(ds, extent)

    # Verify laziness
    assert hasattr(sliced.data.data, "dask")

    # Compute and verify values
    computed = sliced.compute()
    assert computed.lat.min() >= 10
    assert computed.lat.max() <= 30


def test_spatial_slice_wrap():
    """Test longitude wrapping (cross-meridian)."""
    # Grid [0, 360]
    ds = create_global_grid(1.0, 1.0)
    ds["data"] = (["lat", "lon"], np.random.rand(180, 360))

    # Requested extent [-20, 20] which crosses 0/360
    extent = (-20, 20, -10, 10)
    sliced = spatial_slice(ds, extent)

    # Result should have longitudes around 340-360 and 0-20
    assert sliced.lon.size > 0
    assert (sliced.lon >= 340).any()
    assert (sliced.lon <= 20).any()
    assert "wrapped=True" in sliced.attrs["history"]


def test_spatial_slice_crs():
    """Test CRS-aware slicing."""
    pytest.importorskip("pyproj")

    ds = create_global_grid(1.0, 1.0)
    ds["data"] = (["lat", "lon"], np.random.rand(180, 360))

    # Extent in Web Mercator (EPSG:3857) around (0,0)
    # roughly 10 degrees in meters
    extent_3857 = (-1113194, 1113194, -1113194, 1113194)
    sliced = spatial_slice(ds, extent_3857, crs="EPSG:3857")

    assert sliced.lat.min() < 0
    assert sliced.lat.max() > 0
    assert sliced.lon.min() < 10  # 350 in 0-360
    assert sliced.lon.max() > 0
