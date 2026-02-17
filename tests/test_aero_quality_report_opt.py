import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid
from unittest.mock import MagicMock, patch, PropertyMock


def test_quality_report_no_gather_distributed(mocker):
    """
    Aero Protocol: Verify that quality_report avoids gathering the full weight matrix.
    """
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(30, 30)

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 1. Setup remote weights simulation
    mock_client = MagicMock()
    regridder._dask_client = mock_client

    remote_matrix = MagicMock()
    remote_matrix.key = "remote_weights_123"
    regridder._weights_matrix = remote_matrix

    # Mock submit for _get_nnz_task
    mock_nnz_future = MagicMock()
    mock_nnz_future.result.return_value = 1234

    def side_effect(func, *args, **kwargs):
        if "_get_nnz_task" in str(func):
            return mock_nnz_future
        # Return a mock future for anything else (like diagnostics)
        f = MagicMock()
        f.result.return_value = np.ones(int(np.prod(regridder._shape_target)))
        return f

    mock_client.submit.side_effect = side_effect

    # Mock diagnostics to avoid Dask computation issues in the test
    mock_diag = xr.Dataset(
        {
            "weight_sum": (["lat", "lon"], np.ones((6, 12))),
            "unmapped_mask": (["lat", "lon"], np.zeros((6, 12))),
        },
        coords={"lat": np.arange(6), "lon": np.arange(12)},
    )
    mocker.patch.object(regridder, "diagnostics", return_value=mock_diag)

    # 2. Call quality_report with a spy on the weights property
    # We use a simple attribute mock instead of property mock to avoid confusion
    with patch.object(Regridder, "weights", new_callable=PropertyMock) as mock_weights:
        # We need to make sure mock_weights doesn't trigger anything
        report = regridder.quality_report(skip_heavy=False)

    # 3. Verifications
    assert report["n_weights"] == 1234
    # Ensure weights property was NEVER accessed (which would mean no gather)
    assert mock_weights.call_count == 0

    # Check that diagnostics was used for other metrics
    assert report["unmapped_count"] == 0


def test_quality_report_eager_fallback():
    """Verify that quality_report still works correctly for eager NumPy weights."""
    src_grid = create_global_grid(30, 30)
    tgt_grid = create_global_grid(30, 30)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    report = regridder.quality_report()
    assert report["n_weights"] > 0
    assert "unmapped_count" in report


if __name__ == "__main__":
    pytest.main([__file__])
