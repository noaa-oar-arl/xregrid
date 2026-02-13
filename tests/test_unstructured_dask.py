import pytest
import xarray as xr
import numpy as np
import dask.distributed
from xregrid import Regridder, create_global_grid, create_mesh_from_coords

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
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=HAS_REAL_ESMF
    )
    client = dask.distributed.Client(cluster)
    yield client
    client.close()
    cluster.close()


def test_regrid_structured_to_unstructured_dask(dask_client):
    source_grid = create_global_grid(10, 10)

    n_pts = 50
    lon = np.linspace(0, 360, n_pts)
    lat = np.linspace(-90, 90, n_pts)
    target_grid = create_mesh_from_coords(lon, lat, "EPSG:4326")

    regridder = Regridder(source_grid, target_grid, method="nearest_s2d", parallel=True)

    data = xr.DataArray(
        np.random.rand(source_grid.sizes["lat"], source_grid.sizes["lon"]),
        coords={"lat": source_grid.lat, "lon": source_grid.lon},
        dims=["lat", "lon"],
        name="test_data",
    )

    res = regridder(data)

    assert res.shape == (n_pts,)
    assert "n_pts" in res.dims
    assert "lat" in res.coords
    assert "lon" in res.coords


def test_regrid_unstructured_to_structured_dask(dask_client):
    n_pts = 50
    lon = np.linspace(0, 360, n_pts)
    lat = np.linspace(-90, 90, n_pts)
    source_grid = create_mesh_from_coords(lon, lat, "EPSG:4326")
    source_grid["test_data"] = (["n_pts"], np.random.rand(n_pts))

    target_grid = create_global_grid(10, 10)

    regridder = Regridder(source_grid, target_grid, method="nearest_s2d", parallel=True)

    da = source_grid["test_data"]
    res = regridder(da)

    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])
    assert "lat" in res.dims
    assert "lon" in res.dims


def test_regrid_unstructured_to_unstructured_dask(dask_client):
    n_pts_src = 50
    lon_src = np.linspace(0, 360, n_pts_src)
    lat_src = np.linspace(-90, 90, n_pts_src)
    source_grid = create_mesh_from_coords(lon_src, lat_src, "EPSG:4326")
    source_grid["test_data"] = (["n_pts"], np.random.rand(n_pts_src))

    n_pts_dst = 30
    lon_dst = np.linspace(0, 360, n_pts_dst)
    lat_dst = np.linspace(-90, 90, n_pts_dst)
    target_grid = create_mesh_from_coords(lon_dst, lat_dst, "EPSG:4326")

    regridder = Regridder(source_grid, target_grid, method="nearest_s2d", parallel=True)

    da = source_grid["test_data"]
    res = regridder(da)

    assert res.shape == (n_pts_dst,)
    assert "n_pts" in res.dims
