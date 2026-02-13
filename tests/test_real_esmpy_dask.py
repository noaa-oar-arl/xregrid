import pytest
import xarray as xr
import numpy as np
import dask.distributed
from xregrid import Regridder, create_global_grid
from unittest.mock import MagicMock


# Check if esmpy is mocked
def is_esmpy_mocked():
    try:
        import esmpy

        return (
            hasattr(esmpy, "_is_mock")
            or isinstance(esmpy, MagicMock)
            or "MagicMock" in str(type(esmpy))
        )
    except ImportError:
        return True


HAS_REAL_ESMF = not is_esmpy_mocked()
pytestmark = pytest.mark.skipif(not HAS_REAL_ESMF, reason="esmpy is missing or mocked")


@pytest.fixture(scope="module")
def dask_client():
    # Use processes=True for real ESMF thread-safety
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=HAS_REAL_ESMF
    )
    client = dask.distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_global_rectilinear_regrid_dask(dask_client):
    # Test global grid regridding which was reported to fail
    res_src = 1.0
    res_dst = 2.0

    source_grid = create_global_grid(res_src, res_src)
    target_grid = create_global_grid(res_dst, res_dst)

    # Add data with multiple time steps to test dask application over time
    nt = 5
    data = xr.DataArray(
        np.random.rand(nt, source_grid.sizes["lat"], source_grid.sizes["lon"]),
        coords={"time": np.arange(nt), "lat": source_grid.lat, "lon": source_grid.lon},
        dims=["time", "lat", "lon"],
        name="air",
    ).chunk({"time": 1})

    # Initialize Regridder with parallel=True
    # Testing both periodic=True and False
    for periodic in [True, False]:
        print(f"Testing periodic={periodic}...")
        regridder = Regridder(
            source_grid,
            target_grid,
            method="bilinear",
            parallel=True,
            periodic=periodic,
        )

        res = regridder(data)

        assert res.shape == (nt, target_grid.sizes["lat"], target_grid.sizes["lon"])
        # Trigger computation
        res_computed = res.compute()
        assert not np.isnan(res_computed).all()
        print(f"Periodic={periodic} success!")


def test_descending_lat_regrid_dask(dask_client):
    # Test descending latitudes which was identified as a bug
    source_grid = create_global_grid(1.0, 1.0)
    source_grid = source_grid.sortby("lat", ascending=False)

    target_grid = create_global_grid(2.0, 2.0)

    data = xr.DataArray(
        np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"]),
        coords={"lat": source_grid.lat, "lon": source_grid.lon},
        dims=["lat", "lon"],
        name="test",
    ).chunk({"lat": 45})

    regridder = Regridder(
        source_grid, target_grid, method="bilinear", parallel=True, periodic=True
    )
    res = regridder(data).compute()
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])


def test_diagnostics_dask(dask_client):
    source_grid = create_global_grid(10.0, 10.0)
    target_grid = create_global_grid(20.0, 20.0)

    regridder = Regridder(source_grid, target_grid, method="bilinear", parallel=True)

    # Test diagnostics
    diag = regridder.diagnostics()
    assert "weight_sum" in diag
    assert "unmapped_mask" in diag

    # Trigger computation of lazy diagnostics
    ws = diag.weight_sum.compute()
    assert ws.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])

    # Test quality report
    report = regridder.quality_report()
    assert "n_src" in report
    assert "unmapped_fraction" in report


if __name__ == "__main__":
    pytest.main([__file__])
