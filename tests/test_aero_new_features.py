import os
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid, plot


def test_generation_time_provenance():
    """Verify that weight generation time is tracked and included in history."""
    src_grid = create_global_grid(10, 20)
    tgt_grid = create_global_grid(15, 30)

    filename = "test_provenance.nc"
    if os.path.exists(filename):
        os.remove(filename)

    # Test Eager (NumPy)
    data = np.random.rand(18, 18)
    # Only keep coords that share dims with the data
    valid_coords = {
        c: src_grid.coords[c]
        for c in src_grid.coords
        if set(src_grid.coords[c].dims).issubset({"lat", "lon"})
    }
    da_src = xr.DataArray(data, dims=("lat", "lon"), coords=valid_coords, name="test")

    # Set reuse_weights=True so it saves the file
    regridder = Regridder(src_grid, tgt_grid, filename=filename, reuse_weights=True)
    assert regridder.generation_time is not None
    assert regridder.generation_time > 0

    da_regridded = regridder(da_src)
    assert "Weight generation time" in da_regridded.attrs["history"]

    # Verify persistence
    regridder_reused = Regridder(
        src_grid, tgt_grid, filename=filename, reuse_weights=True
    )
    # Should be identical as it's loaded from the same file
    assert regridder_reused.generation_time == regridder.generation_time

    da_regridded_reused = regridder_reused(da_src)
    assert (
        f"Weight generation time: {regridder.generation_time:.4f}s"
        in da_regridded_reused.attrs["history"]
    )

    if os.path.exists(filename):
        os.remove(filename)


def test_unified_plot_dispatch():
    """Verify that the unified plot function dispatches correctly."""
    da_test = xr.DataArray(np.random.rand(10, 10), dims=("lat", "lon"), name="test")

    # Mock plot_static and plot_interactive
    with patch("xregrid.viz.plot_static") as mock_static:
        with patch("xregrid.viz.plot_interactive") as mock_interactive:
            # Test static dispatch
            plot(da_test, mode="static", custom_arg=True)
            mock_static.assert_called_once_with(da_test, custom_arg=True)

            # Test interactive dispatch
            plot(da_test, mode="interactive", custom_arg=False)
            mock_interactive.assert_called_once_with(da_test, custom_arg=False)

    # Test invalid mode
    with pytest.raises(ValueError, match="Unknown plotting mode"):
        plot(da_test, mode="invalid")


def test_backend_agnostic_provenance():
    """Verify provenance works for both NumPy and Dask backends."""
    src_grid = create_global_grid(10, 20)
    tgt_grid = create_global_grid(10, 20)  # Identity regridding for simplicity

    regridder = Regridder(src_grid, tgt_grid, reuse_weights=False)
    valid_coords = {
        c: src_grid.coords[c]
        for c in src_grid.coords
        if set(src_grid.coords[c].dims).issubset({"lat", "lon"})
    }

    # NumPy
    da_numpy = xr.DataArray(
        np.random.rand(18, 18), dims=("lat", "lon"), coords=valid_coords
    )
    out_numpy = regridder(da_numpy)
    assert "Weight generation time" in out_numpy.attrs["history"]

    # Dask
    da_dask = xr.DataArray(
        da.from_array(np.random.rand(18, 18), chunks=(9, 9)),
        dims=("lat", "lon"),
        coords=valid_coords,
    )
    out_dask = regridder(da_dask)
    assert "Weight generation time" in out_dask.attrs["history"]
    # Ensure it's still a dask array
    assert out_dask.chunks is not None
