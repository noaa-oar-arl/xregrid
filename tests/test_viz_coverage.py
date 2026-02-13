import pytest
import xarray as xr
import numpy as np
from xregrid.viz import plot_static, plot_comparison, plot_interactive, plot_diagnostics
from xregrid import Regridder, create_global_grid
from unittest.mock import patch, MagicMock


def test_plot_static_crs_detection():
    """Verify CRS detection from attributes in plot_static."""
    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-90, 90, 18), "lon": np.linspace(0, 360, 36)},
        name="test_data",
    )

    # Test with grid_mapping attribute
    da.attrs["grid_mapping"] = "crs"
    crs_var = xr.DataArray(0, name="crs")
    crs_var.attrs["crs_wkt"] = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    ds = da.to_dataset().assign(crs=crs_var)

    # Call plot_static
    with patch("matplotlib.pyplot.axes") as mock_axes:
        mock_ax = MagicMock()
        mock_axes.return_value = mock_ax
        plot_static(ds.test_data)
        assert mock_axes.called


def test_plot_static_fallback_no_cartopy():
    """Verify fallback to standard matplotlib when cartopy is missing."""
    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-90, 90, 18), "lon": np.linspace(0, 360, 36)},
    )

    with patch("xregrid.viz.ccrs", None):
        with patch("matplotlib.pyplot.gca") as mock_gca:
            plot_static(da)
            assert mock_gca.called


def test_plot_static_robust_slicing():
    """Verify robust slicing of extra dimensions in plot_static."""
    da = xr.DataArray(
        np.random.rand(3, 18, 36),
        dims=("time", "lat", "lon"),
        coords={
            "time": [1, 2, 3],
            "lat": np.linspace(-90, 90, 18),
            "lon": np.linspace(0, 360, 36),
        },
        name="test_data",
    )

    with patch("matplotlib.pyplot.axes"):
        with pytest.warns(UserWarning, match="Automatically selecting the first slice"):
            plot_static(da)


def test_plot_comparison_smoke():
    """Smoke test for plot_comparison."""
    src_da = xr.DataArray(np.random.rand(18, 36), dims=("lat", "lon"), name="src")
    tgt_da = xr.DataArray(np.random.rand(36, 72), dims=("lat", "lon"), name="tgt")

    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(3)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_comparison(src_da, tgt_da)
        assert mock_subplots.called


def test_plot_diagnostics_smoke():
    """Smoke test for plot_diagnostics."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)

    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(2)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_diagnostics(regridder)
        assert mock_subplots.called


def test_plot_interactive_smoke():
    """Smoke test for plot_interactive."""
    from xregrid.viz import hvplot as has_hvplot

    if not has_hvplot:
        pytest.skip("hvplot missing")
    da = xr.DataArray(np.random.rand(18, 36), dims=("lat", "lon"), name="test")
    plot_interactive(da)
