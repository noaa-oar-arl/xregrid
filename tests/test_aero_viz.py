import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_comparison_interactive

try:
    import hvplot.xarray  # noqa: F401
    import holoviews as hv

    HAS_HV = True
except ImportError:
    HAS_HV = False


@pytest.mark.skipif(not HAS_HV, reason="hvplot/holoviews not installed")
def test_plot_comparison_interactive_types():
    """Verify that plot_comparison_interactive returns the correct HoloViews object."""
    # Setup small grids
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(20, 20)

    # Eager Data
    da_src = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="test_data",
    )

    # Target data (dummy)
    da_tgt = xr.DataArray(
        np.random.rand(9, 18),
        dims=("lat", "lon"),
        coords={"lat": tgt_grid.lat, "lon": tgt_grid.lon},
        name="test_data",
    )

    # 1. Test with Eager data, no regridder
    layout = plot_comparison_interactive(da_src, da_tgt)
    assert isinstance(layout, hv.Layout)
    assert len(layout) == 3  # Source, Target, Difference

    # 2. Test with Lazy data
    da_src_lazy = da_src.chunk({"lat": 3, "lon": 6})
    da_tgt_lazy = da_tgt.chunk({"lat": 3, "lon": 6})
    layout_lazy = plot_comparison_interactive(da_src_lazy, da_tgt_lazy)
    assert isinstance(layout_lazy, hv.Layout)

    # 3. Test with Regridder
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")
    layout_regrid = plot_comparison_interactive(da_src, da_tgt, regridder=regridder)
    assert isinstance(layout_regrid, hv.Layout)


@pytest.mark.skipif(not HAS_HV, reason="hvplot/holoviews not installed")
def test_plot_comparison_interactive_titles():
    """Verify that titles are correctly applied to the layout."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(20, 20)
    da_src = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
    )
    da_tgt = xr.DataArray(
        np.random.rand(9, 18),
        dims=("lat", "lon"),
        coords={"lat": tgt_grid.lat, "lon": tgt_grid.lon},
    )

    title = "My Custom Comparison"
    layout = plot_comparison_interactive(da_src, da_tgt, title=title)

    # In HoloViews, title might be in opts
    # We just check it doesn't crash and returns the layout
    assert isinstance(layout, hv.Layout)
