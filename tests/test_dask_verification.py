import xarray as xr
import numpy as np
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


def test_dask_parallel_regridding():
    """
    Test that running with parallel=True creates the same weights as serial execution.
    Also verifies lazy initialization.
    """
    # Create LocalCluster for testing
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=HAS_REAL_ESMF
    )
    client = dask.distributed.Client(cluster)

    try:
        source_grid = create_global_grid(10, 10)
        target_grid = create_global_grid(5, 5)

        # 1. Generate weights in serial
        regridder_serial = Regridder(
            source_grid, target_grid, method="bilinear", parallel=False
        )
        w_serial = regridder_serial.weights

        # 2a. Generate weights in parallel (Eager)
        try:
            regridder_eager = Regridder(
                source_grid, target_grid, method="bilinear", parallel=True, compute=True
            )
            w_eager = regridder_eager.weights
        except Exception as e:
            print(f"ERROR in Regridder creation: {e}")
            import traceback

            traceback.print_exc()
            raise

        # 3a. Compare Eager
        assert w_serial.shape == w_eager.shape
        if HAS_REAL_ESMF:
            assert w_serial.nnz == w_eager.nnz
            diff_eager = w_serial - w_eager
            assert np.abs(diff_eager.data).max() < 1e-10 if diff_eager.nnz > 0 else True

        # 2b. Generate weights in parallel (Lazy)
        regridder_lazy = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True, compute=False
        )

        # Verify persist mechanism
        assert regridder_lazy.persist() is regridder_lazy

        # Verify it hasn't computed yet
        assert regridder_lazy._weights_matrix is None
        assert regridder_lazy._dask_futures is not None

        # Trigger compute
        regridder_lazy.compute()
        w_lazy = regridder_lazy.weights

        assert w_lazy is not None
        assert regridder_lazy._dask_futures is None

        # 3b. Compare Lazy
        assert w_serial.shape == w_lazy.shape
        if HAS_REAL_ESMF:
            assert w_serial.nnz == w_lazy.nnz
            diff_lazy = w_serial - w_lazy
            assert np.abs(diff_lazy.data).max() < 1e-10 if diff_lazy.nnz > 0 else True

        # 4. Compare regridding result on dummy data
        data = np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"])
        da = xr.DataArray(
            data,
            coords={"lat": source_grid.lat, "lon": source_grid.lon},
            dims=["lat", "lon"],
        )

        res_serial = regridder_serial(da)
        res_parallel = regridder_lazy(da)

        if HAS_REAL_ESMF:
            xr.testing.assert_allclose(res_serial, res_parallel)

        # 5. Test auto-compute on call
        regridder_auto = Regridder(
            source_grid, target_grid, method="bilinear", parallel=True, compute=False
        )
        assert regridder_auto._weights_matrix is None
        res_auto = regridder_auto(da)
        assert regridder_auto._weights_matrix is not None
        if HAS_REAL_ESMF:
            xr.testing.assert_allclose(res_serial, res_auto)

    finally:
        client.close()
        cluster.close()


def test_dask_curvilinear_parallel():
    """
    Test parallel regridding on curvilinear grids.
    """
    from xregrid import create_grid_from_crs

    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=HAS_REAL_ESMF
    )
    client = dask.distributed.Client(cluster)

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
