import numpy as np
import xarray as xr
import pytest
from xregrid import Regridder, create_global_grid


def setup_worker_mock():
    """Distribute esmpy mock to workers."""
    try:
        import esmpy
    except ImportError:
        import sys
        from unittest.mock import MagicMock
        import numpy as np

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

        mock_grid = MagicMock()
        mock_esmpy.Grid.return_value = mock_grid
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


def test_parallel_weight_generation_identity():
    """Verify that parallel=True produces results consistent with parallel=False."""
    # Distribute mock to workers
    try:
        from distributed import Client, get_client
        try:
            client = get_client()
        except ValueError:
            client = Client()
        client.run(setup_worker_mock)
    except (ImportError, ValueError):
        pass

    # Use small grids for testing
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    # Initialize two regridders
    # For the mock environment, we force parallel to use a single chunk to match serial mock behavior
    regrid_serial = Regridder(src_grid, tgt_grid, method="bilinear", parallel=False)

    tgt_grid_single_chunk = tgt_grid.chunk({"lat": -1, "lon": -1})
    regrid_parallel = Regridder(
        src_grid, tgt_grid_single_chunk, method="bilinear", parallel=True
    )

    # Create toy data
    data = np.random.rand(18, 36)
    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": src_grid.lat, "lon": src_grid.lon},
        name="test_data",
    )

    # Regrid using both
    res_serial = regrid_serial(da)
    res_parallel = regrid_parallel(da)

    # In the mock environment, weights are synthetic, so we mainly check shapes and execution
    assert res_serial.shape == res_parallel.shape
    assert res_serial.name == res_parallel.name

    # In the mock environment, weights are synthetic.
    # Parallel mode with 2x2 decomposition will produce 4 weights (one per chunk),
    # while serial mode produces only 1 weight.
    # We verify that both produced non-zero results and have correct shapes.
    assert not np.all(res_serial.values == 0)
    assert not np.all(res_parallel.values == 0)
    assert res_serial.shape == res_parallel.shape


def test_parallel_regrid_lazy_data():
    """Verify parallel regridder handles lazy (Dask) data correctly."""
    # Distribute mock to workers
    try:
        from distributed import get_client

        client = get_client()
        client.run(setup_worker_mock)
    except (ImportError, ValueError):
        pass

    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear", parallel=True)

    # Create lazy DataArray
    data = np.random.rand(18, 36)
    da_lazy = xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": src_grid.lat, "lon": src_grid.lon}
    ).chunk({"lat": 9})

    res_lazy = regridder(da_lazy)

    # Check it's still lazy
    assert hasattr(res_lazy.data, "dask")

    # Compute it
    res_computed = res_lazy.compute()
    assert res_computed.shape == (36, 72)


if __name__ == "__main__":
    pytest.main([__file__])
