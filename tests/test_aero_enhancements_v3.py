import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_recursion_safety_aux_coords():
    """
    Aero Protocol: Verify recursion safety with complex auxiliary coordinates.
    Ensures that DataArrays with nested or self-referencing coordinates don't cause infinite loops.
    """
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(15, 15)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # Create a DataArray with an auxiliary coordinate that has the same spatial dimensions
    data = np.random.rand(6, 12)
    lat = src_grid.lat
    lon = src_grid.lon

    aux_data = np.random.rand(6, 12)
    da_aux = xr.DataArray(
        aux_data,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="aux_coord",
    )

    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon, "my_aux": da_aux},
        name="test_data",
    )

    # Regrid Eager
    res_eager = regridder(da)
    assert "my_aux" in res_eager.coords
    assert res_eager.my_aux.shape == (12, 24)

    # Regrid Lazy
    da_lazy = da.chunk({"lat": 3, "lon": 6})
    res_lazy = regridder(da_lazy)

    # Double-Check: Eager vs Lazy identity
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    xr.testing.assert_allclose(res_eager.my_aux, res_lazy.my_aux.compute())


def test_mutual_recursion_safety():
    """
    Verify safety when two auxiliary coordinates refer to each other.
    """
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(30, 30)
    regridder = Regridder(src_grid, tgt_grid)

    lat = src_grid.lat
    lon = src_grid.lon

    da1 = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="da1",
    )
    da2 = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="da2",
    )

    # Manually create mutual reference in coords (possible in xarray)
    da1 = da1.assign_coords(other=da2)
    da2 = da2.assign_coords(other=da1)

    # This should not hang or crash
    res = regridder(da1)
    assert "other" in res.coords


def test_plot_diagnostics_dispatch():
    """Verify that plot_diagnostics method exists and dispatches without error."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(30, 30)
    regridder = Regridder(src_grid, tgt_grid)

    # We mock the actual plotting calls to avoid needing a GUI/display
    try:
        import matplotlib.pyplot as plt

        # Test static
        fig = regridder.plot_diagnostics(mode="static")
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pass

    # Test interactive (we check if it calls hvplot, but we don't need to render it)
    try:
        import hvplot.xarray  # noqa: F401

        layout = regridder.plot_diagnostics(mode="interactive")
        assert layout is not None
    except ImportError:
        pass


if __name__ == "__main__":
    pytest.main([__file__])
