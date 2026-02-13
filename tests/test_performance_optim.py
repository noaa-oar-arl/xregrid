import numpy as np
from xregrid.core import _apply_weights_core, _WORKER_CACHE, _matmul
from scipy.sparse import csr_matrix
import dask.distributed
import xarray as xr


def test_stationary_mask_caching(mocker):
    """Verify that stationary mask normalization is cached across calls."""
    # Clear cache
    _WORKER_CACHE.clear()

    # Mock _matmul to count calls
    mock_matmul = mocker.patch("xregrid.core._matmul", side_effect=_matmul)

    # Create small synthetic data
    data = np.ones((2, 4, 4), dtype=np.float32)
    data[:, 0:2, 0:2] = np.nan  # Stationary mask

    # Mock weight matrix (16 -> 4)
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

    # Verify results are identical
    np.testing.assert_allclose(res1, res2)


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

    # Setup local cluster with processes=False
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

        # Use classes instead of MagicMock for mesh info to avoid pickling recursion
        class MockObj:
            def __init__(self, name):
                self.name = name

        # Mock Regridder internally
        mocker.patch("xregrid.regridder.Regridder._generate_weights", return_value=None)
        mocker.patch(
            "xregrid.regridder.Regridder._get_mesh_info",
            side_effect=[
                (MockObj("src"), ["src"], (4, 4), ("lat", "lon"), False),
                (MockObj("dst"), ["dst"], (2, 2), ("lat", "lon"), False),
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
        assert mock_matmul.call_count == 5

    finally:
        client.close()
        cluster.close()
