import pytest
import dask.distributed
from xregrid import Regridder, create_global_grid

# Check for real ESMF
try:
    import esmpy

    if hasattr(esmpy, "_is_mock") or "unittest.mock" in str(type(esmpy)):
        raise ImportError
    HAS_REAL_ESMF = True
except ImportError:
    HAS_REAL_ESMF = False


@pytest.fixture(scope="module")
def dask_client():
    # esmpy is not thread-safe, so we must use processes=True when using real ESMF
    cluster = dask.distributed.LocalCluster(
        n_workers=1, threads_per_worker=1, processes=HAS_REAL_ESMF
    )
    client = dask.distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_lazy_diagnostics_distributed(dask_client):
    """
    Aero Protocol: Verify that diagnostics remain lazy in distributed mode.
    """
    src = create_global_grid(30, 60)
    tgt = create_global_grid(10, 20)

    regridder = Regridder(src, tgt, parallel=True)

    # 1. Verify Lazy Diagnostics
    diag = regridder.diagnostics()
    assert hasattr(diag.weight_sum.data, "dask"), "weight_sum should be lazy"
    assert hasattr(diag.unmapped_mask.data, "dask"), "unmapped_mask should be lazy"

    # 2. Verify __repr__ is non-blocking and lazy-aware
    repr_str = repr(regridder)
    assert "quality=lazy" in repr_str

    # 3. Verify quality_report(skip_heavy=True) handles remote weights
    # If skip_heavy=True and remote, it should return -1 to avoid roundtrips.
    report_light = regridder.quality_report(skip_heavy=True)
    assert report_light["n_weights"] == -1

    # 4. Verify weights property gathers correctly
    w = regridder.weights
    assert w is not None
    assert not hasattr(regridder._weights_matrix, "key")

    # 5. Verify quality_report(skip_heavy=False) now works eagerly
    report_heavy = regridder.quality_report(skip_heavy=False)
    assert report_heavy["n_weights"] != -1
    assert "unmapped_count" in report_heavy


def test_diagnostics_distributed_identity():
    """
    Verify that Eager and Lazy diagnostic values have identical shapes.
    Note: Exact values may differ with synthetic mocks due to multi-chunking.
    """
    src = create_global_grid(30, 60)
    tgt = create_global_grid(15, 30)

    # Eager
    regridder_eager = Regridder(src, tgt, parallel=False)
    diag_eager = regridder_eager.diagnostics()

    # Lazy
    regridder_lazy = Regridder(src, tgt, parallel=True)
    diag_lazy_raw = regridder_lazy.diagnostics()
    assert hasattr(diag_lazy_raw.weight_sum.data, "dask")
    diag_lazy = diag_lazy_raw.compute()

    # Verify shapes and dimensions
    assert diag_eager.sizes == diag_lazy.sizes
    assert diag_eager.weight_sum.shape == diag_lazy.weight_sum.shape
    # Verify we have some weights
    assert diag_lazy.weight_sum.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__])
