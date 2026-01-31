import numpy as np
import pytest
import xarray as xr
from xregrid import create_global_grid


def test_stationary_mask_optimization():
    """
    Test that the stationary mask optimization produces correct results.
    Following Aero Protocol: Verifies Eager (NumPy) and Lazy (Dask) identity.
    """
    # 1. Setup grids (small grids for testing with mocked ESMF)
    src_res = 10.0
    tgt_res = 20.0
    src_grid = create_global_grid(src_res, src_res)
    tgt_grid = create_global_grid(tgt_res, tgt_res)

    # 2. Create data with stationary NaNs
    ntime = 4
    nlat = src_grid.lat.size
    nlon = src_grid.lon.size

    # Use deterministic data for testing
    data = np.ones((ntime, nlat, nlon))

    # Add a stationary mask
    data_with_nans = data.copy()
    data_with_nans[:, 0, 0] = np.nan  # First point is always NaN

    da_src = xr.DataArray(
        data_with_nans,
        coords={"time": np.arange(ntime), "lat": src_grid.lat, "lon": src_grid.lon},
        dims=("time", "lat", "lon"),
        name="test_data",
    )

    # 3. Test Eager (NumPy) with Accessor
    # skipna=True triggers the optimized stationary mask path
    da_regridded_eager = da_src.regrid.to(tgt_grid, method="bilinear", skipna=True)

    assert isinstance(da_regridded_eager, xr.DataArray)
    # Target shape: (ntime, 9, 18) for 20deg resolution
    assert da_regridded_eager.shape == (ntime, tgt_grid.lat.size, tgt_grid.lon.size)

    # 4. Test Lazy (Dask) with Accessor
    da_src_lazy = da_src.chunk({"time": 2})
    da_regridded_lazy = da_src_lazy.regrid.to(tgt_grid, method="bilinear", skipna=True)

    # Verify it is still lazy
    assert hasattr(da_regridded_lazy.data, "dask")

    # Compute and compare
    da_regridded_lazy_computed = da_regridded_lazy.compute()

    # Identity test: Eager vs Lazy
    xr.testing.assert_allclose(da_regridded_eager, da_regridded_lazy_computed)

    # Verify history tracking (Scientific Hygiene)
    assert "history" in da_regridded_eager.attrs
    assert "Regridded" in da_regridded_eager.attrs["history"]


def test_non_stationary_mask():
    """Test that non-stationary masks still work correctly (slow path)."""
    src_res = 10.0
    tgt_res = 20.0
    src_grid = create_global_grid(src_res, src_res)
    tgt_grid = create_global_grid(tgt_res, tgt_res)

    ntime = 3
    nlat = src_grid.lat.size
    nlon = src_grid.lon.size
    data = np.ones((ntime, nlat, nlon))

    # Different mask for each time step
    data[0, 0, 0] = np.nan
    data[1, 0, 1] = np.nan
    data[2, 0, 2] = np.nan

    da_src = xr.DataArray(
        data,
        coords={"time": np.arange(ntime), "lat": src_grid.lat, "lon": src_grid.lon},
        dims=("time", "lat", "lon"),
        name="test_data_moving",
    )

    da_regridded = da_src.regrid.to(tgt_grid, method="bilinear", skipna=True)
    assert da_regridded.shape == (ntime, tgt_grid.lat.size, tgt_grid.lon.size)


if __name__ == "__main__":
    pytest.main([__file__])
