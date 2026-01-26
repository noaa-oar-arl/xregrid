import pytest
import xarray as xr
import numpy as np
from xregrid import Regridder, create_global_grid


def test_weight_method_mismatch(tmp_path):
    """Verify that a ValueError is raised when method doesn't match loaded weights."""
    filename = str(tmp_path / "weights.nc")
    src = create_global_grid(10, 20)
    tgt = create_global_grid(5, 10)

    # 1. Save with bilinear
    Regridder(src, tgt, method="bilinear", reuse_weights=True, filename=filename)

    # 2. Try to load with patch
    with pytest.raises(
        ValueError,
        match="Requested method 'patch' does not match loaded weights method 'bilinear'",
    ):
        Regridder(src, tgt, method="patch", reuse_weights=True, filename=filename)


def test_weight_periodic_mismatch(tmp_path):
    """Verify that a ValueError is raised when periodic flag doesn't match loaded weights."""
    filename = str(tmp_path / "weights.nc")
    src = create_global_grid(10, 20)
    tgt = create_global_grid(5, 10)

    # 1. Save with periodic=False
    Regridder(
        src,
        tgt,
        method="bilinear",
        periodic=False,
        reuse_weights=True,
        filename=filename,
    )

    # 2. Try to load with periodic=True
    with pytest.raises(
        ValueError,
        match="Requested periodic=True does not match loaded weights periodic=False",
    ):
        Regridder(
            src,
            tgt,
            method="bilinear",
            periodic=True,
            reuse_weights=True,
            filename=filename,
        )


def test_validation_double_check(tmp_path):
    """Double-Check Test: Verify validation works for both Eager and Lazy data paths."""
    filename = str(tmp_path / "weights.nc")
    src = create_global_grid(10, 20)
    tgt = create_global_grid(5, 10)

    # Generate weights
    regridder = Regridder(
        src, tgt, method="bilinear", reuse_weights=True, filename=filename
    )

    # Eager data
    da_eager = xr.DataArray(
        np.random.rand(src.lat.size, src.lon.size),
        dims=("lat", "lon"),
        coords={"lat": src.lat, "lon": src.lon},
    )
    res_eager = regridder(da_eager)

    # Lazy data
    da_lazy = da_eager.chunk({"lat": 9, "lon": 18})
    res_lazy = regridder(da_lazy).compute()

    # Identity check (Scientific Hygiene)
    xr.testing.assert_allclose(
        res_eager.drop_vars("history", errors="ignore"),
        res_lazy.drop_vars("history", errors="ignore"),
    )

    # Now verify that a NEW regridder with the SAME filename but DIFFERENT method still fails
    # regardless of data backend (it fails at init)
    with pytest.raises(ValueError, match="Requested method"):
        Regridder(
            src, tgt, method="conservative", reuse_weights=True, filename=filename
        )


def test_plot_static_ax_validation():
    """Verify that plot_static correctly validates GeoAxes and doesn't warn unnecessarily."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        pytest.skip("matplotlib or cartopy not installed")

    from xregrid.viz import plot_static
    import warnings

    da = xr.DataArray(
        np.random.rand(10, 10),
        dims=("lat", "lon"),
        coords={"lat": np.arange(10), "lon": np.arange(10)},
    )

    # 1. Test with proper GeoAxes (should NOT warn)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plot_static(da, ax=ax)
        geo_warnings = [
            str(warning.message) for warning in w if "GeoAxes" in str(warning.message)
        ]
        assert len(geo_warnings) == 0
    plt.close(fig)

    # 2. Test with regular Axes (SHOULD warn)
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            plot_static(da, ax=ax)
        except ValueError:
            # Cartopy raises ValueError when plotting with transform on non-GeoAxes
            pass
        geo_warnings = [
            str(warning.message) for warning in w if "GeoAxes" in str(warning.message)
        ]
        assert len(geo_warnings) > 0
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
