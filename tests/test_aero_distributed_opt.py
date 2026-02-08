import numpy as np
import pytest
import xarray as xr
import sys
from unittest.mock import MagicMock
import dask.distributed


# --- Mock ESMF Setup (Pickleable) ---
class AnyAssignment:
    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return self


class MockESMFObject:
    def __init__(self, *args, **kwargs):
        self.staggerloc = [0, 1]
        self.items = {}

    def get_coords(self, *args, **kwargs):
        return AnyAssignment()

    def get_item(self, *args, **kwargs):
        return AnyAssignment()

    def add_item(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        self.items[key] = value

    def __getitem__(self, key):
        return self.items.get(key, AnyAssignment())


class MockRegrid:
    def __init__(self, *args, **kwargs):
        pass

    def get_factors(self):
        return np.array([0]), np.array([0])

    def get_weights_dict(self, deep_copy=True):
        # Return synthetic weights: map source (0,0) to all destination points
        # For simplicity in testing identity.
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
    mock.Grid = MockESMFObject
    mock.LocStream = MockESMFObject
    mock.Mesh = MockESMFObject
    mock.Field.return_value = MagicMock()
    mock.Regrid = MockRegrid
    sys.modules["esmpy"] = mock


setup_mock_esmpy()
from xregrid import Regridder, create_global_grid  # noqa: E402


@pytest.fixture(scope="module")
def dask_client():
    # Use processes=False to avoid pickling issues with mocks in this environment
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=False
    )
    client = dask.distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_aero_distributed_optimization_identity(dask_client):
    """
    Aero Protocol: Verify regridding logic twice:
    1. Eager (NumPy) data.
    2. Lazy (Dask) data.
    Ensures identical results and verifies the distributed weight path.
    """
    # Create grids
    source_grid = create_global_grid(30, 60)  # small grid
    target_grid = create_global_grid(10, 20)

    # Initialize Regridder with parallel=True to trigger distributed weights logic
    regridder = Regridder(source_grid, target_grid, method="bilinear", parallel=True)

    # Verify that weights are stored as a Future initially
    assert hasattr(regridder._weights_matrix, "key")

    # Prepare input data
    data_raw = np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"])
    da_eager = xr.DataArray(
        data_raw,
        coords={"lat": source_grid.lat, "lon": source_grid.lon},
        dims=["lat", "lon"],
        name="test_data",
    )

    # --- 1. Eager Path ---
    # Calling regridder on eager data should gather the weights
    res_eager = regridder(da_eager)
    assert not hasattr(regridder._weights_matrix, "key")  # Should be gathered now
    assert isinstance(res_eager.data, np.ndarray)

    # --- 2. Lazy Path ---
    # Re-initialize regridder to get a Future again
    regridder_lazy = Regridder(
        source_grid, target_grid, method="bilinear", parallel=True
    )
    assert hasattr(regridder_lazy._weights_matrix, "key")

    da_lazy = da_eager.chunk({"lat": 5})
    res_lazy_raw = regridder_lazy(da_lazy)

    # Verify it stays lazy
    assert hasattr(res_lazy_raw.data, "dask")

    # Compute
    res_lazy = res_lazy_raw.compute()

    # Aero Protocol: Assert Eager and Lazy results are identical
    xr.testing.assert_allclose(res_eager, res_lazy)

    # --- 3. Verify stationary mask optimization (skipna=True) ---
    da_nan = da_eager.expand_dims(time=2).copy(deep=True)
    da_nan.values[:, 0, 0] = np.nan  # Stationary NaN across time

    regridder_skipna = Regridder(
        source_grid, target_grid, method="bilinear", parallel=True, skipna=True
    )

    # Test Eager skipna
    res_skipna_eager = regridder_skipna(da_nan)

    # Test Lazy skipna
    da_nan_lazy = da_nan.chunk({"time": 1})
    res_skipna_lazy = regridder_skipna(da_nan_lazy).compute()

    xr.testing.assert_allclose(res_skipna_eager, res_skipna_lazy)
    print("Aero Distributed Optimization Tests Passed!")


def test_aero_vectorized_triangulation():
    """Verify the new vectorized triangulation logic for MPAS/UGRID."""
    from xregrid.xregrid import _get_unstructured_mesh_info

    # Create fake MPAS dataset
    conn = np.array(
        [[1, 2, 3, 0], [4, 5, 6, 7]]
    )  # 1-based, cell 1 has 3 edges, cell 2 has 4
    n_edges = np.array([3, 4])
    ds = xr.Dataset(
        coords={
            "latVertex": (["nVertices"], np.zeros(10)),
            "lonVertex": (["nVertices"], np.zeros(10)),
        },
        data_vars={
            "verticesOnCell": (["nCells", "maxEdges"], conn),
            "nEdgesOnCell": (["nCells"], n_edges),
        },
    )

    # Trigger vectorized triangulation
    _, _, element_conn, _, _, orig_idx = _get_unstructured_mesh_info(ds)

    # Expected:
    # Cell 0 -> 1 triangle [0, 1, 2]
    # Cell 1 -> 2 triangles [3, 4, 5], [3, 5, 6]
    assert len(element_conn) == 9
    assert np.array_equal(orig_idx, [0, 1, 1])
    print("Vectorized Triangulation Test Passed!")
