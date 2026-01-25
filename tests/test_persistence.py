import os
import numpy as np
import xarray as xr
from xregrid import ESMPyRegridder, create_global_grid


def create_sample_data(dask=False, seed=None):
    """Create sample data for testing."""
    if seed is not None:
        np.random.seed(seed)
    ds = create_global_grid(res_lat=10.0, res_lon=10.0)
    data = np.random.rand(len(ds.lat), len(ds.lon))
    da = xr.DataArray(
        data,
        coords={"lat": ds.lat, "lon": ds.lon},
        dims=["lat", "lon"],
        name="test_data",
    )
    if dask:
        da = da.chunk({"lat": 9, "lon": 18})
    return da, ds


def test_weight_persistence_eager_lazy(tmp_path):
    """
    Test that weights can be saved and reloaded, and results are identical
    for both eager and lazy data.
    """
    filename = str(tmp_path / "weights.nc")

    # 1. Create source and target grids
    source_da, source_grid = create_sample_data(dask=False)
    target_grid = create_global_grid(res_lat=5.0, res_lon=5.0)

    # 2. Generate and save weights
    regridder_save = ESMPyRegridder(
        source_grid,
        target_grid,
        method="bilinear",
        reuse_weights=True,
        filename=filename,
    )

    res_original = regridder_save(source_da)
    assert os.path.exists(filename)

    # 3. Load weights in a new regridder
    regridder_load = ESMPyRegridder(
        source_grid,
        target_grid,
        method="bilinear",
        reuse_weights=True,
        filename=filename,
    )

    # Verify eager result identity
    res_eager = regridder_load(source_da)
    xr.testing.assert_allclose(res_original, res_eager)

    # 4. Verify lazy result identity (Aero Protocol: Double-Check Test)
    # Re-create source_da as dask with same values
    source_da_lazy = source_da.chunk({"lat": 9, "lon": 18})
    res_lazy = regridder_load(source_da_lazy).compute()

    # Remove history from comparison as timestamps will differ
    res_eager_no_hist = res_eager.copy()
    res_lazy_no_hist = res_lazy.copy()
    res_eager_no_hist.attrs.pop("history", None)
    res_lazy_no_hist.attrs.pop("history", None)

    xr.testing.assert_allclose(res_eager_no_hist, res_lazy_no_hist)

    # Also verify that the loaded result matches the original fresh result
    res_original_no_hist = res_original.copy()
    res_original_no_hist.attrs.pop("history", None)
    xr.testing.assert_allclose(res_original_no_hist, res_eager_no_hist)


def test_weight_persistence_skipna(tmp_path):
    """Test persistence with skipna=True."""
    filename = str(tmp_path / "weights_skipna.nc")

    source_da, source_grid = create_sample_data(dask=False)
    # Add some NaNs
    source_da.values[0, 0] = np.nan

    target_grid = create_global_grid(res_lat=5.0, res_lon=5.0)

    regridder_save = ESMPyRegridder(
        source_grid,
        target_grid,
        method="bilinear",
        reuse_weights=True,
        filename=filename,
        skipna=True,
    )

    res_save = regridder_save(source_da)

    regridder_load = ESMPyRegridder(
        source_grid,
        target_grid,
        method="bilinear",
        reuse_weights=True,
        filename=filename,
        skipna=True,
    )

    res_load = regridder_load(source_da)

    xr.testing.assert_allclose(res_save, res_load)
