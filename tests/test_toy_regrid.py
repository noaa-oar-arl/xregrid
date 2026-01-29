import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_toy_regrid_global_05():
    """Test regridding the xarray toy dataset to a global 0.5 degree grid."""
    # 1. Load toy dataset
    # We use a subset of time to keep the test fast
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 2))

    # 2. Create target global 0.5 degree grid
    # A 0.5 degree global grid has 360 latitude points and 720 longitude points
    target_res = 0.5
    target_grid = create_global_grid(res_lat=target_res, res_lon=target_res)

    expected_lat_size = int(180 / target_res)
    expected_lon_size = int(360 / target_res)

    assert target_grid.lat.size == expected_lat_size
    assert target_grid.lon.size == expected_lon_size

    # 3. Initialize Regridder
    # Note: In the test environment, ESMF is mocked, so weight generation is synthetic
    regridder = Regridder(ds, target_grid, method="bilinear", periodic=True)

    # 4. Regrid the DataArray
    air_regridded = regridder(ds.air)

    # 5. Verify DataArray output
    assert isinstance(air_regridded, xr.DataArray)
    assert air_regridded.shape == (ds.time.size, expected_lat_size, expected_lon_size)
    assert "lat" in air_regridded.coords
    assert "lon" in air_regridded.coords
    assert "time" in air_regridded.coords

    # Check that coordinates match the target grid
    np.testing.assert_allclose(air_regridded.lat, target_grid.lat)
    np.testing.assert_allclose(air_regridded.lon, target_grid.lon)

    # 6. Regrid the Dataset
    ds_regridded = regridder(ds)

    # 7. Verify Dataset output
    assert isinstance(ds_regridded, xr.Dataset)
    assert "air" in ds_regridded.data_vars
    assert ds_regridded.air.shape == (
        ds.time.size,
        expected_lat_size,
        expected_lon_size,
    )

    # Verify provenance
    assert "history" in ds_regridded.attrs
    assert "Regridder" in ds_regridded.attrs["history"]
    assert "bilinear" in ds_regridded.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
