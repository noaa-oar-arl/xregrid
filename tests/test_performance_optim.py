import numpy as np
import pytest
from xregrid.xregrid import _apply_weights_core, _WORKER_CACHE, _matmul
from scipy.sparse import csr_matrix
from unittest.mock import MagicMock
import dask.distributed
import xarray as xr


def test_stationary_mask_caching(mocker):
    """Verify that stationary mask normalization is cached across calls."""
    # Clear cache
    _WORKER_CACHE.clear()

    # Mock _matmul to count calls
    # Note: we patch the core module since that's where the active reference is
    mock_matmul = mocker.patch("xregrid.core._matmul", side_effect=_matmul)

    # Create small synthetic data
    # 2 time steps, 4x4 grid -> 2x2 target
    data = np.ones((2, 4, 4), dtype=np.float32)
    data[:, 0:2, 0:2] = np.nan  # Stationary mask (top left corner)

    # Mock weight matrix (16 -> 4)
    # Identity-like for the first 4 points
    weights = np.zeros((4, 16))
    for i in range(4):
        weights[i, i] = 1.0
    weights_sparse = csr_matrix(weights)

    weights_key = "test_weights_key"
    _WORKER_CACHE[weights_key] = weights_sparse

    # 1st call: should compute weights_sum and cache it
    res1 = _apply_weights_core(
        data[0:1], weights_key, ("lat", "lon"), (2, 2), skipna=True
    )

    # count should be 2: one for data, one for weights_sum
    assert mock_matmul.call_count == 2

    # Reset mock count
    mock_matmul.reset_mock()

    # 2nd call: should use cached weights_sum
    res2 = _apply_weights_core(
        data[1:2], weights_key, ("lat", "lon"), (2, 2), skipna=True
    )

    # count should be 1: only for data
    assert mock_matmul.call_count == 1

    # Verify results are identical since input data was identical
    np.testing.assert_allclose(res1, res2)

    # Verify it handles NON-stationary mask correctly (cache invalidation)
    mock_matmul.reset_mock()
    data_diff = data.copy()
    data_diff[1, 3, 3] = np.nan  # New NaN

    _ = _apply_weights_core(
        data_diff[1:2], weights_key, ("lat", "lon"), (2, 2), skipna=True
    )

    # Should recompute weights_sum because mask changed
    assert mock_matmul.call_count == 2


def test_memory_efficiency_broadcasting():
    """Verify that stationary mask uses broadcasting."""
    data = np.ones((10, 4, 4), dtype=np.float32)
    data[:, 0:2, 0:2] = np.nan

    weights = csr_matrix(np.eye(4, 16))

    # Should not crash and should be correct
    res = _apply_weights_core(data, weights, ("lat", "lon"), (2, 2), skipna=True)
    assert res.shape == (10, 2, 2)
    assert np.isnan(res[0, 0, 0])
    assert res[0, 1, 1] == 1.0


def test_dask_stationary_mask_caching(mocker):
    """Verify stationary mask caching works with Dask-backed data."""
    from xregrid.regridder import Regridder

    # Setup local cluster with processes=False so it uses our modules
    cluster = dask.distributed.LocalCluster(
        n_workers=1, threads_per_worker=1, processes=False
    )
    client = dask.distributed.Client(cluster)

    try:
        _WORKER_CACHE.clear()
        mock_matmul = mocker.patch("xregrid.core._matmul", side_effect=_matmul)

        # Identity-like weight matrix
        weights = csr_matrix(np.eye(4, 16))

        # Source data (Dask-backed)
        data = np.ones((4, 4, 4), dtype=np.float32)
        data[:, 0:2, 0:2] = np.nan
        da = xr.DataArray(data, dims=("time", "lat", "lon")).chunk({"time": 1})

        ds_src = xr.Dataset(
            {"data": da}, coords={"lat": np.arange(4), "lon": np.arange(4)}
        )
        ds_dst = xr.Dataset(coords={"lat": np.arange(2), "lon": np.arange(2)})

        # Mock Regridder internally to avoid actual ESMF calls
        mocker.patch("xregrid.regridder.Regridder._generate_weights", return_value=None)
        mocker.patch(
            "xregrid.regridder.Regridder._get_mesh_info",
            side_effect=[
                (MagicMock(), MagicMock(), (4, 4), ("lat", "lon"), False),
                (MagicMock(), MagicMock(), (2, 2), ("lat", "lon"), False),
            ],
        )

        regridder = Regridder(ds_src, ds_dst, skipna=True)
        # Inject our known state
        regridder._weights_matrix = weights
        regridder._dims_source = ("lat", "lon")
        regridder._dims_target = ("lat", "lon")
        regridder._shape_target = (2, 2)
        regridder._total_weights = np.array(weights.sum(axis=1)).T

        # Apply regridding
        out = regridder(da)

        # Trigger compute
        res = out.compute()

        # Verify results
        assert res.shape == (4, 2, 2)
        assert np.isnan(res[0, 0, 0])

        # Verify calls
        # With time chunk=1, there are 4 chunks.
        # 1st chunk: computes and caches (2 calls to _matmul)
        # 2nd, 3rd, 4th chunk: use cache (1 call each)
        # Total: 2 + 1 + 1 + 1 = 5 calls
        assert mock_matmul.call_count == 5

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    pytest.main([__file__])
