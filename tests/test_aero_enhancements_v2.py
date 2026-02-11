import numpy as np
import xarray as xr
import dask.array as da
import pytest
from xregrid.xregrid import _bounds_to_vertices, Regridder
from xregrid.utils import create_global_grid


def test_bounds_to_vertices_lazy():
    """Verify _bounds_to_vertices stays lazy with Dask arrays."""
    # 1D case
    b1 = xr.DataArray(
        da.from_array(np.random.rand(10, 2), chunks=(5, 2)), dims=("x", "b")
    )
    v1 = _bounds_to_vertices(b1)
    assert hasattr(v1.data, "dask")
    assert v1.shape == (11,)

    # 3D case (curvilinear)
    b3 = xr.DataArray(
        da.from_array(np.random.rand(10, 20, 4), chunks=(5, 10, 4)),
        dims=("y", "x", "b"),
    )
    v3 = _bounds_to_vertices(b3)
    assert hasattr(v3.data, "dask")
    assert v3.shape == (11, 21)

    # Verify values match NumPy
    v1_np = _bounds_to_vertices(b1.compute())
    if isinstance(v1, xr.DataArray):
        xr.testing.assert_allclose(v1.compute(), v1_np)
    else:
        np.testing.assert_allclose(v1.compute(), v1_np)

    v3_np = _bounds_to_vertices(b3.compute())
    if isinstance(v3, xr.DataArray):
        xr.testing.assert_allclose(v3.compute(), v3_np)
    else:
        np.testing.assert_allclose(v3.compute(), v3_np)


def test_regridder_plot_weights_smoke():
    """Smoke test for plot_weights method."""
    ds_src = create_global_grid(30, 30)
    ds_tgt = create_global_grid(60, 60)

    regridder = Regridder(ds_src, ds_tgt, method="bilinear")

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    # Plot weights for the first destination point
    fig = regridder.plot_weights(0)
    assert fig is not None
    plt.close()


@pytest.mark.parametrize("method", ["bilinear", "conservative"])
def test_conservative_with_dask_bounds(method):
    """Ensure Regridder works when bounds are dask-backed."""
    ds_src = create_global_grid(30, 30)
    ds_tgt = create_global_grid(60, 60)

    # Chunk bounds (using new dimension names from normalization)
    ds_src["lat_b"] = ds_src["lat_b"].chunk({"lat": 5})
    ds_src["lon_b"] = ds_src["lon_b"].chunk({"lon": 5})

    # This should not trigger compute until absolutely necessary (inside ESMPy)
    regridder = Regridder(ds_src, ds_tgt, method=method)

    # Use only dimensions as coords to avoid CoordinateValidationError
    da_src = xr.DataArray(
        np.random.rand(6, 12),
        dims=("lat", "lon"),
        coords={c: ds_src.coords[c] for c in ["lat", "lon"]},
    )
    res = regridder(da_src)
    assert res is not None
