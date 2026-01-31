import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix
from xregrid.xregrid import _apply_weights_core


def test_apply_weights_core_eager_lazy():
    """Verify _apply_weights_core works identically for NumPy and handles Dask via apply_ufunc."""
    # Create a simple 2x2 -> 1x1 regridding (averaging)
    # Weights matrix: [0.25, 0.25, 0.25, 0.25]
    weights = csr_matrix(
        ([0.25, 0.25, 0.25, 0.25], ([0, 0, 0, 0], [0, 1, 2, 3])), shape=(1, 4)
    )

    dims_source = ("lat", "lon")
    shape_target = (1, 1)

    data = np.array([[1.0, 2.0], [3.0, 4.0]])  # mean is 2.5

    # 1. Eager check
    res_eager = _apply_weights_core(data, weights, dims_source, shape_target)
    assert res_eager.shape == (1, 1)
    assert res_eager[0, 0] == 2.5

    # 2. Check with NaNs and skipna=True
    data_nan = np.array(
        [[1.0, np.nan], [3.0, 4.0]]
    )  # mean of valid is (1+3+4)/3 = 8/3 = 2.666...
    total_weights = np.array([1.0])  # sum of all weights for the cell

    res_nan = _apply_weights_core(
        data_nan,
        weights,
        dims_source,
        shape_target,
        skipna=True,
        total_weights=total_weights,
    )
    np.testing.assert_allclose(res_nan[0, 0], 8 / 3)


def test_apply_weights_core_dask_integration():
    """Verify that _apply_weights_core can be used within xr.apply_ufunc with Dask."""
    import dask.array as da

    weights = csr_matrix(
        ([0.25, 0.25, 0.25, 0.25], ([0, 0, 0, 0], [0, 1, 2, 3])), shape=(1, 4)
    )
    dims_source = ("lat", "lon")
    shape_target = (1, 1)

    data = np.random.rand(4, 2, 2)  # (time, lat, lon)
    da_in = xr.DataArray(data, dims=("time", "lat", "lon")).chunk({"time": 2})

    out = xr.apply_ufunc(
        _apply_weights_core,
        da_in,
        kwargs={
            "weights_matrix": weights,
            "dims_source": dims_source,
            "shape_target": shape_target,
        },
        input_core_dims=[list(dims_source)],
        output_core_dims=[["lat_out", "lon_out"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"lat_out": 1, "lon_out": 1}},
    )

    assert isinstance(out.data, da.Array)
    res = out.compute()
    assert res.shape == (4, 1, 1)

    # Verify values
    for i in range(4):
        expected = np.mean(data[i])
        np.testing.assert_allclose(res.values[i, 0, 0], expected)


if __name__ == "__main__":
    pytest.main([__file__])
