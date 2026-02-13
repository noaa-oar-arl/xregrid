import os
import pytest
import xarray as xr
import numpy as np
from xregrid import Regridder, create_global_grid
from unittest.mock import patch


def test_regridder_mpi_parallel_error():
    """Verify ValueError when both mpi and parallel are True."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    with pytest.raises(ValueError, match="Cannot use both MPI and Dask"):
        Regridder(src, tgt, mpi=True, parallel=True)


def test_regridder_missing_dask_error():
    """Verify ImportError when parallel=True but dask.distributed is missing."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="Dask distributed is required"):
            Regridder(src, tgt, parallel=True)


def test_regridder_save_load_weights(tmp_path):
    """Verify saving and loading weights."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    weight_file = str(tmp_path / "weights.nc")

    # Weights are saved automatically if reuse_weights=True and file doesn't exist
    Regridder(src, tgt, method="bilinear", filename=weight_file, reuse_weights=True)
    assert os.path.exists(weight_file)

    # Load weights
    regridder2 = Regridder.from_weights(weight_file, src, tgt)
    assert regridder2.method == "bilinear"

    # Verify validation fails with wrong parameters
    with pytest.raises(ValueError, match="does not match loaded weights method"):
        Regridder.from_weights(weight_file, src, tgt, method="conservative")


def test_regrid_dataset_coverage():
    """Verify _regrid_dataset with various variable types."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    ds = xr.Dataset(
        data_vars={
            "var1": (["lat", "lon"], np.random.rand(18, 36)),
            "var2": (["lat", "lon"], np.random.rand(18, 36)),
            "scalar": 42,
            "other": (["time"], [1, 2, 3]),
        },
        coords={"lat": src.lat, "lon": src.lon, "time": [0, 1, 2]},
    )

    regridder = Regridder(src, tgt)
    res = regridder(ds)

    assert "var1" in res.data_vars
    assert "var2" in res.data_vars
    assert "scalar" in res.data_vars
    assert "other" in res.data_vars
    # create_global_grid(5, 5) gives (36, 72)
    assert res.var1.shape == (36, 72)


def test_extrap_methods_coverage():
    """Verify different extrapolation methods."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    for method in ["nearest_s2d", "nearest_idw", "creep_fill"]:
        regridder = Regridder(src, tgt, extrap_method=method, extrap_dist_exponent=3.0)
        assert regridder.extrap_method == method


def test_regridder_repr_lazy():
    """Verify __repr__ with lazy weights."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    regridder = Regridder(src, tgt, parallel=True, compute=False)
    # The Regridder should already have a Future if parallel=True and compute=False
    # but since we are mocking, let's ensure it has the right attributes for the repr

    # Use a simple class instead of MagicMock to avoid pickling recursion
    class MockFuture:
        def __init__(self):
            self.key = "some_key"

    regridder._weights_matrix = MockFuture()

    repr_str = repr(regridder)
    assert "quality=lazy" in repr_str


def test_regridder_quality_report_coverage():
    """Verify quality_report with different options."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)

    report = regridder.quality_report(format="dataset")
    assert isinstance(report, xr.Dataset)
    assert "unmapped_fraction" in report.data_vars
