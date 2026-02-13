import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder

# Check for real ESMF
try:
    import esmpy

    if hasattr(esmpy, "_is_mock") or "unittest.mock" in str(type(esmpy)):
        raise ImportError
    HAS_REAL_ESMF = True
except ImportError:
    HAS_REAL_ESMF = False


def test_descending_coordinates():
    """Test that regridding works correctly with descending coordinates (like air_temperature)."""
    # Create a source grid with descending latitude
    lat = np.linspace(90, -90, 19)  # 10 degree resolution
    lon = np.linspace(0, 350, 36)
    ds_src = xr.Dataset(
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )
    # Set data to lat values
    ds_src["data"] = (
        ["lat", "lon"],
        np.broadcast_to(lat[:, None], (len(lat), len(lon))),
    )

    # Target grid with ascending latitude
    target_lat = np.linspace(-90, 90, 37)  # 5 degree resolution
    target_lon = np.linspace(0, 355, 72)  # Includes out-of-bounds lon (355)
    ds_tgt = xr.Dataset(
        coords={
            "lat": (["lat"], target_lat, {"units": "degrees_north"}),
            "lon": (["lon"], target_lon, {"units": "degrees_east"}),
        }
    )

    regridder = Regridder(ds_src, ds_tgt, method="bilinear")
    res = regridder(ds_src["data"])

    if not HAS_REAL_ESMF:
        pytest.skip("Skipping scientific correctness check for mocked ESMF")

    # Check mean. Points at lon=355 will be zero (unmapped).
    # There are 72 lon points. 71 are mapped, 1 is unmapped.
    # Expected mean: val * 71 / 72
    expected_mean = target_lat * 71 / 72
    np.testing.assert_allclose(res.mean(dim="lon"), expected_mean, atol=1e-5)

    # Verify unmapped points at the boundary
    assert np.all(res.sel(lon=355) == 0)


def test_mixed_monotonicity():
    """Test that regridding works when only one coordinate is descending."""
    lat = np.linspace(-90, 90, 19)
    lon = np.linspace(350, 0, 36)  # descending lon
    ds_src = xr.Dataset(
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )
    ds_src["data"] = (
        ["lat", "lon"],
        np.broadcast_to(lon[None, :], (len(lat), len(lon))),
    )

    target_lat = np.linspace(-90, 90, 19)
    target_lon = np.linspace(0, 350, 36)  # ascending lon
    ds_tgt = xr.Dataset(
        coords={
            "lat": (["lat"], target_lat, {"units": "degrees_north"}),
            "lon": (["lon"], target_lon, {"units": "degrees_east"}),
        }
    )

    regridder = Regridder(ds_src, ds_tgt, method="bilinear")
    res = regridder(ds_src["data"])

    # The output should have ascending lon as requested by target_grid
    assert np.all(np.diff(res.lon) > 0)

    if not HAS_REAL_ESMF:
        pytest.skip("Skipping scientific correctness check for mocked ESMF")

    # Data should match target_lon
    np.testing.assert_allclose(res.mean(dim="lat"), target_lon, atol=1e-5)


def test_output_order_preservation():
    """Test that the output preserves the coordinate order of the target grid."""
    lat_src = np.linspace(-90, 90, 10)
    lon_src = np.linspace(0, 360, 10)
    ds_src = xr.Dataset(coords={"lat": lat_src, "lon": lon_src})
    ds_src["data"] = (["lat", "lon"], np.random.rand(10, 10))

    # Target grid with DESCENDING latitude
    lat_tgt = np.linspace(90, -90, 10)
    lon_tgt = np.linspace(0, 360, 10)
    ds_tgt = xr.Dataset(coords={"lat": lat_tgt, "lon": lon_tgt})

    regridder = Regridder(ds_src, ds_tgt)
    res = regridder(ds_src["data"])

    # Result should have descending latitude as requested
    assert np.all(np.diff(res.lat) < 0)
    np.testing.assert_allclose(res.lat, lat_tgt)
