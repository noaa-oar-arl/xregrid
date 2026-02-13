import numpy as np
import pytest
import xarray as xr
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
    # For CI stability, we use a single worker if using real ESMF
    cluster = dask.distributed.LocalCluster(
        n_workers=1 if HAS_REAL_ESMF else 2,
        threads_per_worker=1,
        processes=HAS_REAL_ESMF,
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
    if HAS_REAL_ESMF:
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

    if HAS_REAL_ESMF:
        xr.testing.assert_allclose(res_skipna_eager, res_skipna_lazy)


def test_aero_vectorized_triangulation():
    """Verify the new vectorized triangulation logic for MPAS/UGRID."""
    from xregrid.grid import _get_unstructured_mesh_info

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
    _, _, _, _, _, orig_idx = _get_unstructured_mesh_info(ds)

    # MPAS-specific detection should happen
    assert len(orig_idx) > 0
