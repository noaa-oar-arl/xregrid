import sys
from unittest.mock import MagicMock

import dask.distributed
import numpy as np
import pytest
import xarray as xr


# Setup mock for the driver process
def setup_driver_mock():
    if "esmpy" in sys.modules and isinstance(sys.modules["esmpy"], MagicMock):
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
    mock_esmpy.MeshLoc.NODE = 0
    mock_esmpy.MeshLoc.ELEMENT = 1
    mock_esmpy.MeshElemType.TRI = 1
    mock_esmpy.MeshElemType.QUAD = 2
    mock_esmpy.NormType.FRACAREA = 0
    mock_esmpy.NormType.DSTAREA = 1
    mock_esmpy.Manager.return_value = MagicMock()
    mock_esmpy.pet_count.return_value = 1
    mock_esmpy.local_pet.return_value = 0
    mock_esmpy.__version__ = "8.6.0"

    class Grid:
        def __init__(self, *args, **kwargs):
            self.get_coords = MagicMock()
            self.get_item = MagicMock()
            self.add_item = MagicMock()
            self.staggerloc = [0, 1]

    class LocStream:
        def __init__(self, *args, **kwargs):
            self.items = {}

        def __setitem__(self, key, value):
            self.items[key] = value

    mock_esmpy.Grid = Grid
    mock_esmpy.LocStream = LocStream

    class Mesh:
        def __init__(self, *args, **kwargs):
            pass

        def add_nodes(self, *args, **kwargs):
            pass

        def add_elements(self, *args, **kwargs):
            pass

    mock_esmpy.Mesh = Mesh
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


# Initialize mock only if real esmpy is not available
try:
    import esmpy

    # Check if it's already a mock from conftest
    if hasattr(esmpy, "_is_mock") or "unittest.mock" in str(type(esmpy)):
        raise ImportError
    HAS_REAL_ESMF = True
except ImportError:
    HAS_REAL_ESMF = False

if not HAS_REAL_ESMF:
    setup_driver_mock()

from xregrid import Regridder, create_global_grid, create_mesh_from_coords  # noqa: E402


def setup_worker_mock():
    """Setup esmpy mock for Dask workers."""
    import sys
    from unittest.mock import MagicMock

    import numpy as np

    if "esmpy" in sys.modules and isinstance(sys.modules["esmpy"], MagicMock):
        return

    mock_esmpy = MagicMock()
    mock_esmpy.CoordSys.SPH_DEG = 1
    mock_esmpy.StaggerLoc.CENTER = 0
    mock_esmpy.RegridMethod.BILINEAR = 0
    mock_esmpy.RegridMethod.CONSERVE = 1
    mock_esmpy.RegridMethod.NEAREST_STOD = 2
    mock_esmpy.RegridMethod.NEAREST_DTOS = 3
    mock_esmpy.RegridMethod.PATCH = 4
    mock_esmpy.UnmappedAction.IGNORE = 1
    mock_esmpy.ExtrapMethod.NEAREST_STOD = 0
    mock_esmpy.ExtrapMethod.NEAREST_IDAVG = 1
    mock_esmpy.ExtrapMethod.CREEP_FILL = 2
    mock_esmpy.MeshLoc.NODE = 0
    mock_esmpy.MeshLoc.ELEMENT = 1
    mock_esmpy.MeshElemType.TRI = 1
    mock_esmpy.MeshElemType.QUAD = 2
    mock_esmpy.NormType.FRACAREA = 0
    mock_esmpy.NormType.DSTAREA = 1
    mock_esmpy.Manager.return_value = MagicMock()
    mock_esmpy.pet_count.return_value = 1
    mock_esmpy.local_pet.return_value = 0
    mock_esmpy.__version__ = "8.6.0"

    class Grid:
        def __init__(self, *args, **kwargs):
            self.get_coords = MagicMock()
            self.staggerloc = [0]

    class LocStream:
        def __init__(self, *args, **kwargs):
            self.items = {}

        def __setitem__(self, key, value):
            self.items[key] = value

    class Mesh:
        def __init__(self, *args, **kwargs):
            pass

        def add_nodes(self, *args, **kwargs):
            pass

        def add_elements(self, *args, **kwargs):
            pass

    mock_esmpy.Grid = Grid
    mock_esmpy.LocStream = LocStream
    mock_esmpy.Mesh = Mesh
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


@pytest.fixture(scope="module")
def dask_client():
    # esmpy is not thread-safe, so we must use processes=True when using real ESMF
    cluster = dask.distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True
    )
    client = dask.distributed.Client(cluster)
    if not HAS_REAL_ESMF:
        client.run(setup_worker_mock)
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

    # parallel=True triggers Dask weight generation
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

    assert res.chunks is not None
    assert res.shape == (18, 36)

    # Trigger compute
    val = res.compute()
    assert not np.isnan(val).all()


def test_mpas_like_detection_dask(dask_client):
    nCells = 50
    ds = xr.Dataset(
        coords={
            "latCell": (
                ["nCells"],
                np.linspace(-90, 90, nCells),
                {"units": "degrees_north"},
            ),
            "lonCell": (
                ["nCells"],
                np.linspace(0, 360, nCells),
                {"units": "degrees_east"},
            ),
        }
    )
    # Mock some data
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
    assert res.chunks is not None
    val = res.compute()
    assert val.shape == (n_pts,)


def test_mpas_conservative_regrid_dask(dask_client):
    if HAS_REAL_ESMF:
        pytest.skip(
            "MPAS conservative regridding with random data fails with real ESMF. Requires valid mesh."
        )
    # Mock MPAS dataset
    nCells = 20
    nVertices = 40
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

    # Use conservative method
    regridder = Regridder(ds, target_grid, method="conservative", parallel=True)

    res = regridder(ds["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])
    val = res.compute()
    assert not np.isnan(val).all()


def test_ugrid_conservative_regrid_dask(dask_client):
    if HAS_REAL_ESMF:
        pytest.skip(
            "UGRID conservative regridding with random data fails with real ESMF. Requires valid mesh."
        )
    # Mock UGRID dataset
    nFaces = 20
    nNodes = 40
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

    # Use conservative method
    regridder = Regridder(ds, target_grid, method="conservative", parallel=True)

    res = regridder(ds["test_var"])
    assert res.shape == (target_grid.sizes["lat"], target_grid.sizes["lon"])
    val = res.compute()
    assert not np.isnan(val).all()
