import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_weights


def test_plot_weights_eager():
    """Double-Check Test: Verify plot_weights works for Eager (NumPy) backend."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)

    # Track A: Static
    with patch("xregrid.viz.plot_static") as mock_plot:
        plot_weights(regridder, row_idx=0, mode="static")
        mock_plot.assert_called_once()
        da_weights = mock_plot.call_args[0][0]
        assert isinstance(da_weights, xr.DataArray)
        assert da_weights.shape == regridder._shape_source

    # Track B: Interactive
    with patch("xregrid.viz.plot_interactive") as mock_plot_int:
        plot_weights(regridder, row_idx=0, mode="interactive")
        mock_plot_int.assert_called_once()


def test_plot_weights_lazy_no_gather():
    """Double-Check Test: Verify plot_weights for Lazy (Dask) backend avoids full gather."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt, parallel=True, compute=False)

    # Mock weights as a remote Future
    class MockFuture:
        def __init__(self):
            self.key = "remote_weights_key"

    regridder._weights_matrix = MockFuture()
    regridder._dask_client = MagicMock()

    # The return of the remote task
    mock_row = np.zeros(regridder._shape_source).flatten()
    regridder._dask_client.submit.return_value.result.return_value = mock_row

    # Call plot_weights
    with patch("xregrid.viz.plot_static") as mock_plot:
        plot_weights(regridder, row_idx=5, mode="static")

        # VERIFY: Plot was called
        mock_plot.assert_called_once()

        # VERIFY: No full gather called on weights matrix
        # (regridder.weights would call client.gather)
        regridder._dask_client.gather.assert_not_called()

        # VERIFY: Distributed task was submitted
        regridder._dask_client.submit.assert_called_once()
        args = regridder._dask_client.submit.call_args[0]
        assert "_get_weight_row_task" in args[0].__name__
        assert args[1] is regridder._weights_matrix
        assert args[2] == 5


def test_plot_weights_invalid_mode():
    """Verify ValueError for invalid mode."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)
    with pytest.raises(ValueError, match="Unknown plotting mode"):
        plot_weights(regridder, row_idx=0, mode="invalid")
