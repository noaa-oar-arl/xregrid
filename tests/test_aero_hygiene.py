import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_regional_grid


def test_eager_lazy_identity():
    """Verify that Eager (NumPy) and Lazy (Dask) data produce identical results."""
    # Create small regional grid
    ds = create_regional_grid((10, 20), (100, 110), 1.0, 1.0)
    ds["data"] = (("lat", "lon"), np.random.rand(10, 10))

    target_grid = create_regional_grid((10, 20), (100, 110), 0.5, 0.5)

    # Initialize Regridder
    regridder = Regridder(ds, target_grid, method="bilinear")

    # Eager regridding
    da_eager = ds.data
    out_eager = regridder(da_eager)

    # Lazy regridding
    da_lazy = da_eager.chunk({"lat": 5, "lon": 5})
    out_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(out_eager, out_lazy)


def test_scientific_hygiene_no_inplace():
    """Verify that input datasets are not modified in-place during regridding."""
    # Create grid without bounds initially
    ds = create_regional_grid((10, 20), (100, 110), 1.0, 1.0, add_bounds=False)
    # Add a variable but no bounds yet
    ds["data"] = (("lat", "lon"), np.random.rand(10, 10))

    # Ensure it has a history attribute to check
    if "history" not in ds.attrs:
        ds.attrs["history"] = "Original history"
    orig_history = ds.attrs.get("history", "")

    target_grid = create_regional_grid((10, 20), (100, 110), 0.5, 0.5)

    # Conservative regridding will trigger bounds generation
    regridder = Regridder(ds, target_grid, method="conservative")

    # Check that ds was not modified in-place
    assert ds.attrs.get("history", "") == orig_history
    assert "lat_b" not in ds.coords

    # Check that output HAS the provenance
    out = regridder(ds.data)
    assert "Automatically generated cell boundaries" in out.attrs["history"]


def test_plot_comparison_with_regridder():
    """Verify that plot_comparison works with a Regridder instance."""
    import matplotlib.pyplot as plt

    ds = create_regional_grid((10, 20), (100, 110), 1.0, 1.0)
    ds["data"] = (("lat", "lon"), np.random.rand(10, 10))
    target_grid = create_regional_grid((10, 20), (100, 110), 0.5, 0.5)

    regridder = Regridder(ds, target_grid, method="bilinear")
    da_tgt = regridder(ds.data)

    # This should run without error
    from xregrid.viz import plot_comparison

    # Use a mock or just run it (it will use the mocked plt in tests)
    fig = plot_comparison(ds.data, da_tgt, regridder=regridder)
    assert fig is not None
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
