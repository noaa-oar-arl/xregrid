import numpy as np
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


def test_aero_quality_lazy_distributed(dask_client):
    """
    Aero Protocol: Double-Check Test for Quality Report.
    Verifies that quality_report(format='dataset') is lazy for Dask-backed regridders
    and matches eager results.
    """
    src = create_global_grid(30, 60)
    tgt = create_global_grid(15, 30)

    # 1. Eager (NumPy) Path
    regridder_eager = Regridder(src, tgt, parallel=False)
    report_eager = regridder_eager.quality_report(format="dataset")

    # Verify report_eager is indeed eager
    for var in report_eager.data_vars:
        assert not hasattr(
            report_eager[var].data, "dask"
        ), f"{var} should be NumPy-backed"

    # 2. Lazy (Dask) Path
    regridder_lazy = Regridder(src, tgt, parallel=True)
    report_lazy = regridder_lazy.quality_report(format="dataset")

    # Verify report_lazy preserves laziness for heavy metrics
    assert hasattr(report_lazy.n_weights.data, "dask"), "n_weights should be lazy"
    assert hasattr(
        report_lazy.unmapped_count.data, "dask"
    ), "unmapped_count should be lazy"
    assert hasattr(
        report_lazy.unmapped_fraction.data, "dask"
    ), "unmapped_fraction should be lazy"
    assert hasattr(
        report_lazy.weight_sum_min.data, "dask"
    ), "weight_sum_min should be lazy"

    # 3. Double-Check Identity
    # Computing the lazy report should yield identical results to the eager one
    report_lazy_comp = report_lazy.compute()

    for var in report_eager.data_vars:
        if var in report_lazy_comp:
            # Note: with synthetic mocks and multiple chunks, n_weights and unmapped_count
            # may differ because each mock chunk generates one weight.
            # Real ESMF would be identical.
            if not HAS_REAL_ESMF and var in [
                "n_weights",
                "unmapped_count",
                "unmapped_fraction",
                "weight_sum_max",
                "weight_sum_mean",
            ]:
                continue

            np.testing.assert_allclose(
                report_eager[var].values,
                report_lazy_comp[var].values,
                err_msg=f"Mismatch in metric {var} between Eager and Lazy backends",
            )


def test_quality_report_dict_is_eager(dask_client):
    """
    Verify that quality_report(format='dict') still returns eager values
    for immediate consumption, even for distributed regridders.
    """
    src = create_global_grid(30, 60)
    tgt = create_global_grid(20, 40)
    regridder = Regridder(src, tgt, parallel=True)

    report = regridder.quality_report(format="dict")

    assert isinstance(report, dict)
    assert isinstance(report["n_weights"], int)
    assert isinstance(report["unmapped_count"], int)
    assert isinstance(report["unmapped_fraction"], float)


if __name__ == "__main__":
    pytest.main([__file__])
