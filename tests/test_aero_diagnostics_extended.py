import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_diagnostics


def test_diagnostics_content():
    """Verify that diagnostics() returns the correct variables and metadata."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear")

    ds_diag = regridder.diagnostics()

    assert isinstance(ds_diag, xr.Dataset)
    assert "weight_sum" in ds_diag
    assert "unmapped_mask" in ds_diag
    assert ds_diag.weight_sum.shape == (12, 24)
    assert ds_diag.unmapped_mask.shape == (12, 24)

    # Check scientific hygiene
    assert "history" in ds_diag.attrs
    assert "Generated spatial diagnostics" in ds_diag.attrs["history"]


def test_diagnostics_consistency_with_quality_report():
    """Verify that diagnostics() and quality_report() metrics match."""
    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear")

    ds_diag = regridder.diagnostics()
    report = regridder.quality_report()

    assert int(ds_diag.unmapped_mask.sum()) == report["unmapped_count"]
    assert float(ds_diag.weight_sum.max()) == pytest.approx(report["weight_sum_max"])


def test_plot_diagnostics_smoke():
    """Verify that plot_diagnostics runs without error (Track A)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)
    regridder = Regridder(src, tgt, method="bilinear")

    fig = plot_diagnostics(regridder)
    assert fig is not None
    plt.close(fig)


def test_diagnostics_parallel_compatibility():
    """
    Aero Protocol: Verify that diagnostics still work for a regridder
    initialized in 'parallel' mode (mocked cluster).
    """
    # Use mocks for esmpy objects if needed, but here we can just test the
    # weight construction path if we have esmpy.

    src = create_global_grid(30, 30)
    tgt = create_global_grid(15, 15)

    # We can't easily run a full Dask cluster in this environment safely without esmpy on workers,
    # but we can verify that if we have a Regridder with weights, diagnostics works.
    regridder = Regridder(src, tgt, method="bilinear")
    # Simulate being in parallel mode after compute()
    regridder.parallel = True

    ds_diag = regridder.diagnostics()
    assert "weight_sum" in ds_diag


def test_diagnostics_lazy_initialization():
    """
    Aero Protocol Double-Check: Verify that Regridder diagnostics work
    when initialized with Dask-backed coordinates.
    """
    src_res = 30
    tgt_res = 15
    src_grid = create_global_grid(src_res, src_res)
    tgt_grid = create_global_grid(tgt_res, tgt_res)

    # Convert coordinates to Dask
    src_grid = src_grid.chunk({"lat": 3, "lon": 6})
    tgt_grid = tgt_grid.chunk({"lat": 3, "lon": 6})

    # Initialize Regridder (Serial mode, but with Lazy coordinates)
    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    ds_diag = regridder.diagnostics()

    assert isinstance(ds_diag, xr.Dataset)
    assert "weight_sum" in ds_diag

    # Verify quality report (Lazy reductions)
    report = regridder.quality_report()
    assert report["n_src"] == (180 // src_res) * (360 // src_res)


if __name__ == "__main__":
    pytest.main([__file__])
