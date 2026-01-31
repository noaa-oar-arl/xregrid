import pytest
import xarray as xr
import numpy as np
import dask.array as da
from unittest.mock import MagicMock, patch
import sys
import os
from xregrid import Regridder, create_global_grid


def test_mpi_initialization():
    """Test that mpi=True correctly initializes ESMF Manager."""
    source_grid = create_global_grid(10, 10)
    target_grid = create_global_grid(20, 20)

    with patch("esmpy.Manager") as mock_manager:
        # We need to make sure LogKind is available in the mock if esmpy is mocked
        import esmpy

        _ = Regridder(source_grid, target_grid, mpi=True)
        mock_manager.assert_called_with(logkind=esmpy.LogKind.MULTI, debug=False)


def test_mpi_weight_gathering():
    """Test that weights are gathered correctly in an MPI environment."""
    source_grid = create_global_grid(10, 10)
    target_grid = create_global_grid(20, 20)

    # Mock weights dictionary for each rank
    weights_rank0 = {
        "row_dst": np.array([1]),
        "col_src": np.array([1]),
        "weights": np.array([0.5]),
    }
    weights_rank1 = {
        "row_dst": np.array([2]),
        "col_src": np.array([2]),
        "weights": np.array([0.5]),
    }

    mock_mpi_pkg = MagicMock()
    mock_mpi_internal = MagicMock()
    mock_mpi_pkg.MPI = mock_mpi_internal
    mock_comm = MagicMock()
    mock_mpi_internal.COMM_WORLD = mock_comm

    # Simulate rank 0
    with patch.dict(sys.modules, {"mpi4py": mock_mpi_pkg}):
        with (
            patch("esmpy.pet_count", return_value=2),
            patch("esmpy.local_pet", return_value=0),
            patch("esmpy.Regrid") as mock_regrid_class,
        ):
            mock_regrid = MagicMock()
            mock_regrid.get_factors.return_value = (np.array([1]), np.array([1]))
            mock_regrid.get_weights_dict.return_value = weights_rank0
            mock_regrid_class.return_value = mock_regrid

            # Rank 0 gather should receive both
            mock_comm.gather.return_value = [weights_rank0, weights_rank1]

            regridder = Regridder(source_grid, target_grid, mpi=True)

            # Verify gathered matrix
            matrix = regridder._weights_matrix.tocoo()
            np.testing.assert_array_equal(matrix.row, [0, 1])
            np.testing.assert_array_equal(matrix.col, [0, 1])
            np.testing.assert_array_equal(matrix.data, [0.5, 0.5])


def test_mpi_no_save_on_non_root():
    """Test that non-root ranks do not save weights."""
    source_grid = create_global_grid(10, 10)
    target_grid = create_global_grid(20, 20)

    with patch("esmpy.local_pet", return_value=1):
        with patch("xarray.Dataset.to_netcdf") as mock_to_netcdf:
            # reuse_weights=True triggers _save_weights if it doesn't exist
            if os.path.exists("test_weights.nc"):
                os.remove("test_weights.nc")
            _ = Regridder(
                source_grid,
                target_grid,
                mpi=True,
                reuse_weights=True,
                filename="test_weights.nc",
            )
            mock_to_netcdf.assert_not_called()


def test_regrid_eager_lazy_identity():
    """Verify that Eager (NumPy) and Lazy (Dask) regridding produce identical results."""
    # Standard 1.0 -> 2.0 degree regridding
    source_grid = create_global_grid(1.0, 1.0)
    target_grid = create_global_grid(2.0, 2.0)

    regridder = Regridder(source_grid, target_grid, method="bilinear")

    # Create sample data
    data = np.random.rand(180, 360)
    da_eager = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": source_grid.lat, "lon": source_grid.lon},
        name="test",
    )
    da_lazy = da_eager.chunk({"lat": 90, "lon": 180})

    # Regrid both
    res_eager = regridder(da_eager)
    res_lazy = regridder(da_lazy)

    # Assert identity
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert isinstance(res_lazy.data, da.Array)
    assert not isinstance(res_eager.data, da.Array)


if __name__ == "__main__":
    pytest.main([__file__])
