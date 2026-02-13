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


def test_unstructured_with_mask_dask(dask_client):
    n_pts = 20
    lon = np.linspace(0, 360, n_pts)
    lat = np.linspace(-90, 90, n_pts)
    source_grid = create_mesh_from_coords(lon, lat, crs="EPSG:4326")
    source_grid["mask"] = (["n_pts"], np.ones(n_pts, dtype=int))
    source_grid["mask"].values[0] = 0

    target_grid = create_global_grid(10, 10)

    regridder = Regridder(
        source_grid, target_grid, method="nearest_s2d", mask_var="mask", parallel=True
    )

    data = xr.DataArray(
        np.random.rand(n_pts),
        coords={
            "n_pts": source_grid.n_pts,
            "lat": source_grid.lat,
            "lon": source_grid.lon,
        },
        dims=["n_pts"],
        name="test_data",
    ).chunk({"n_pts": 5})

    res = regridder(data)
    assert res.shape == (18, 36)
    val = res.compute()
    assert not np.isnan(val).all()


def test_mpas_like_detection_dask(dask_client):
    nCells = 50
    ds = xr.Dataset(
        coords={
            "latCell": (["nCells"], np.linspace(-90, 90, nCells)),
            "lonCell": (["nCells"], np.linspace(0, 360, nCells)),
        }
    )
    ds["test_var"] = (["nCells"], np.random.rand(nCells))
    target_grid = create_global_grid(30, 60)
    regridder = Regridder(ds, target_grid, method="nearest_s2d", parallel=True)
    res = regridder(ds["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])


def test_unstructured_radians_dask(dask_client):
    n_pts = 20
    lon = np.linspace(0, 2 * np.pi, n_pts)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
    source_grid = xr.Dataset(
        coords={
            "lat": (["n_pts"], lat, {"units": "radians"}),
            "lon": (["n_pts"], lon, {"units": "rad"}),
        }
    )
    source_grid["test_var"] = (["n_pts"], np.random.rand(n_pts))
    target_grid = create_global_grid(10, 10)
    regridder = Regridder(source_grid, target_grid, method="nearest_s2d", parallel=True)
    res = regridder(source_grid["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])


def test_structured_to_unstructured_mask_dask(dask_client):
    source_grid = create_global_grid(10, 10)
    source_grid["mask"] = (["lat", "lon"], np.ones((18, 36), dtype=int))
    n_pts = 30
    target_grid = create_mesh_from_coords(
        np.linspace(0, 360, n_pts), np.linspace(-90, 90, n_pts), crs="EPSG:4326"
    )
    regridder = Regridder(
        source_grid, target_grid, method="nearest_s2d", mask_var="mask", parallel=True
    )
    data = xr.DataArray(
        np.random.rand(18, 36),
        coords={"lat": source_grid.lat, "lon": source_grid.lon},
        dims=["lat", "lon"],
    ).chunk({"lat": 9})
    res = regridder(data)
    assert res.shape == (n_pts,)
    val = res.compute()
    assert val.shape == (n_pts,)


def test_mpas_conservative_regrid_dask(dask_client):
    if HAS_REAL_ESMF:
        pytest.skip("MPAS conservative regridding requires valid mesh.")
    nCells, nVertices = 20, 40
    ds = xr.Dataset(
        coords={
            "latCell": (["nCells"], np.linspace(-90, 90, nCells)),
            "lonCell": (["nCells"], np.linspace(0, 360, nCells)),
            "latVertex": (["nVertices"], np.linspace(-90, 90, nVertices)),
            "lonVertex": (["nVertices"], np.linspace(0, 360, nVertices)),
            "verticesOnCell": (
                ["nCells", "maxNodes"],
                np.random.randint(1, nVertices + 1, (nCells, 6)),
            ),
            "nEdgesOnCell": (["nCells"], np.full(nCells, 6)),
        }
    )
    ds["test_var"] = (["nCells"], np.random.rand(nCells))
    target_grid = create_global_grid(10, 10)
    regridder = Regridder(ds, target_grid, method="conservative", parallel=True)
    res = regridder(ds["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])
    val = res.compute()
    assert not np.isnan(val).all()


def test_ugrid_conservative_regrid_dask(dask_client):
    if HAS_REAL_ESMF:
        pytest.skip("UGRID conservative regridding requires valid mesh.")
    nFaces, nNodes = 20, 40
    ds = xr.Dataset(
        coords={
            "lat_node": (["nNodes"], np.linspace(-90, 90, nNodes)),
            "lon_node": (["nNodes"], np.linspace(0, 360, nNodes)),
            "lat": (["nFaces"], np.linspace(-90, 90, nFaces)),
            "lon": (["nFaces"], np.linspace(0, 360, nFaces)),
            "face_node_connectivity": (
                ["nFaces", "nMaxNodes"],
                np.random.randint(0, nNodes, (nFaces, 4)),
            ),
        }
    )
    ds["face_node_connectivity"].attrs["cf_role"] = "face_node_connectivity"
    ds["face_node_connectivity"].attrs["start_index"] = 0
    ds["test_var"] = (["nFaces"], np.random.rand(nFaces))
    target_grid = create_global_grid(10, 10)
    regridder = Regridder(ds, target_grid, method="conservative", parallel=True)
    res = regridder(ds["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])
    val = res.compute()
    assert not np.isnan(val).all()
