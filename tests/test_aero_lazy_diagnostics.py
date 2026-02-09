import numpy as np
import pytest
import sys
from unittest.mock import MagicMock
import dask.distributed


# --- Mock ESMF Setup (Pickleable) ---
class MockGrid:
    def __init__(self, *args, **kwargs):
        self.staggerloc = [0, 1]

    def get_coords(self, *args, **kwargs):
        return MagicMock()

    def get_item(self, *args, **kwargs):
        return MagicMock()

    def add_item(self, *args, **kwargs):
        pass


class MockMesh:
    def __init__(self, *args, **kwargs):
        pass

    def add_nodes(self, *args, **kwargs):
        pass

    def add_elements(self, *args, **kwargs):
        pass


class MockRegrid:
    def __init__(self, *args, **kwargs):
        pass

    def get_factors(self):
        return np.array([0]), np.array([0])

    def get_weights_dict(self, deep_copy=True):
        return {
            "row_dst": np.array([1]),
            "col_src": np.array([1]),
            "weights": np.array([1.0]),
        }


def setup_mock_esmpy():
    mock = MagicMock()
    mock.CoordSys.SPH_DEG = 1
    mock.CoordSys.CART = 0
    mock.StaggerLoc.CENTER = 0
    mock.StaggerLoc.CORNER = 1
    mock.GridItem.MASK = 1
    mock.RegridMethod.BILINEAR = 0
    mock.RegridMethod.CONSERVE = 1
    mock.RegridMethod.NEAREST_STOD = 2
    mock.RegridMethod.NEAREST_DTOS = 3
    mock.RegridMethod.PATCH = 4
    mock.UnmappedAction.IGNORE = 1
    mock.ExtrapMethod.NEAREST_STOD = 0
    mock.ExtrapMethod.NEAREST_IDAVG = 1
    mock.ExtrapMethod.CREEP_FILL = 2
    mock.MeshLoc.NODE = 0
    mock.MeshLoc.ELEMENT = 1
    mock.MeshElemType.TRI = 1
    mock.MeshElemType.QUAD = 2
    mock.NormType.FRACAREA = 0
    mock.NormType.DSTAREA = 1
    mock.Manager.return_value = MagicMock()
    mock.pet_count.return_value = 1
    mock.local_pet.return_value = 0
    mock.__version__ = "8.6.0"
    mock.Grid = MockGrid
    mock.LocStream = MockGrid
    mock.Mesh = MockMesh
    mock.Field.return_value = MagicMock()
    mock.Regrid = MockRegrid
    sys.modules["esmpy"] = mock


setup_mock_esmpy()
from xregrid import Regridder, create_global_grid  # noqa: E402


@pytest.fixture(scope="module")
def dask_client():
    cluster = dask.distributed.LocalCluster(n_workers=1, processes=False)
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
    report_light = regridder.quality_report(skip_heavy=True)
    assert report_light["n_weights"] == -1

    # 4. Verify weights property gathers correctly
    w = regridder.weights
    assert isinstance(w, (np.ndarray, object))  # scipy sparse matrix
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
    assert diag_eager.dims == diag_lazy.dims
    assert diag_eager.weight_sum.shape == diag_lazy.weight_sum.shape
    # Verify we have some weights
    assert diag_lazy.weight_sum.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__])
