import xarray as xr
import numpy as np
import dask.distributed
from xregrid import Regridder, create_global_grid
from xregrid.xregrid import esmpy
from unittest.mock import MagicMock


def setup_worker_mock():
    from unittest.mock import MagicMock
    import sys
    import numpy as np

    if "esmpy" in sys.modules and not isinstance(sys.modules["esmpy"], MagicMock):
        return

    mock_esmpy = MagicMock()
    mock_esmpy.CoordSys.SPH_DEG = 1
    mock_esmpy.StaggerLoc.CENTER = 0
    mock_esmpy.StaggerLoc.CORNER = 1
    mock_esmpy.GridItem.MASK = 1
    mock_esmpy.RegridMethod.BILINEAR = 0
    mock_esmpy.RegridMethod.CONSERVE = 1
    mock_esmpy.RegridMethod.NEAREST_STOD = 2
    mock_esmpy.RegridMethod.NEAREST_DTOS = 3
    mock_esmpy.RegridMethod.PATCH = 4
    mock_esmpy.UnmappedAction.IGNORE = 1
    mock_esmpy.ExtrapMethod.NEAREST_STOD = 0
    mock_esmpy.ExtrapMethod.NEAREST_IDAVG = 1
    mock_esmpy.ExtrapMethod.CREEP_FILL = 2
    mock_esmpy.Manager.return_value = MagicMock()
    mock_esmpy.pet_count.return_value = 1
    mock_esmpy.local_pet.return_value = 0

    class MockGrid:
        def __init__(self, *args, **kwargs):
            self.get_coords = MagicMock()
            self.get_item = MagicMock()
            self.add_item = MagicMock()
            self.staggerloc = [0, 1]

    mock_esmpy.Grid = MockGrid
    mock_esmpy.Field.return_value = MagicMock()
    mock_regrid = MagicMock()
    mock_regrid.get_factors.return_value = (np.array([0]), np.array([0]))
    mock_regrid.get_weights_dict.return_value = {
        "row_dst": np.array([1]),
        "col_src": np.array([1]),
        "weights": np.array([1.0]),
    }
    mock_esmpy.Regrid.return_value = mock_regrid
    sys.modules["esmpy"] = mock_esmpy


def test_dask_parallel_regridding():
    """
    Test that running with parallel=True creates the same weights as serial execution.
    Also verifies lazy initialization.
    """
    # Create LocalCluster for testing
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True
    )
    client = dask.distributed.Client(cluster)
    client.run(setup_worker_mock)

    try:
        source_grid = create_global_grid(10, 10)
        target_grid = create_global_grid(5, 5)

        # 1. Generate weights in serial
        regridder_serial = Regridder(
            source_grid, target_grid, method="bilinear", parallel=False
        )
        w_serial = regridder_serial._weights_matrix

        # 2a. Generate weights in parallel (Eager)
        print(f"Using Dask Client: {client}")
        regridder_eager = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True, compute=True
        )
        w_eager = regridder_eager._weights_matrix

        # 3a. Compare Eager
        assert w_serial.shape == w_eager.shape
        # Note: Weights might differ in mock environment due to multiple chunks
        # returning the same mock weight. In real ESMF, these would match.
        if not isinstance(esmpy, MagicMock):
            assert w_serial.nnz == w_eager.nnz
            diff_eager = w_serial - w_eager
            assert np.abs(diff_eager.data).max() < 1e-10 if diff_eager.nnz > 0 else True

        # 2b. Generate weights in parallel (Lazy)
        regridder_lazy = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True, compute=False
        )

        # Verify persist mechanism (should just return self)
        assert regridder_lazy.persist() is regridder_lazy

        # Verify it hasn't computed yet
        assert regridder_lazy._weights_matrix is None
        assert regridder_lazy._dask_futures is not None

        # Trigger compute
        print("Triggering compute on lazy regridder...")
        regridder_lazy.compute()
        w_lazy = regridder_lazy._weights_matrix

        assert w_lazy is not None
        assert regridder_lazy._dask_futures is None  # should be cleared

        # 3b. Compare Lazy
        assert w_serial.shape == w_lazy.shape
        if not isinstance(esmpy, MagicMock):
            assert w_serial.nnz == w_lazy.nnz
            diff_lazy = w_serial - w_lazy
            assert np.abs(diff_lazy.data).max() < 1e-10 if diff_lazy.nnz > 0 else True

        print("Generic identity verification successful")

        # 4. Compare regridding result on dummy data
        data = np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"])
        da = xr.DataArray(
            data,
            coords={"lat": source_grid.lat, "lon": source_grid.lon},
            dims=["lat", "lon"],
        )

        res_serial = regridder_serial(da)
        res_parallel = regridder_lazy(da)  # Should be identical

        if not isinstance(esmpy, MagicMock):
            xr.testing.assert_allclose(res_serial, res_parallel)

        # 5. Test auto-compute on call
        regridder_auto = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True, compute=False
        )
        assert regridder_auto._weights_matrix is None
        print("Triggering auto-compute via __call__...")
        res_auto = regridder_auto(da)
        assert regridder_auto._weights_matrix is not None
        if not isinstance(esmpy, MagicMock):
            xr.testing.assert_allclose(res_serial, res_auto)

        print("Dask parallel verification successful!")

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    test_dask_parallel_regridding()


def test_dask_curvilinear_parallel():
    """
    Test parallel regridding on curvilinear grids.
    """
    from xregrid import create_grid_from_crs

    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True
    )
    client = dask.distributed.Client(cluster)
    client.run(setup_worker_mock)

    try:
        source_grid = create_grid_from_crs("EPSG:4326", (0, 10, 0, 10), 1)
        target_grid = create_grid_from_crs(
            "EPSG:3857", (0, 1000000, 0, 1000000), 100000
        )

        regridder = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True
        )
        assert regridder._weights_matrix is not None

        data = xr.DataArray(
            np.random.rand(10, 10),
            coords={"y": source_grid.y, "x": source_grid.x},
            dims=["y", "x"],
        )
        data.coords["lat"] = source_grid.lat
        data.coords["lon"] = source_grid.lon

        res = regridder(data)
        assert res.shape == (10, 10)
    finally:
        client.close()
        cluster.close()
