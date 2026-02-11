import pytest
import xarray as xr
import numpy as np
import dask.distributed


# Setup mock for the driver process too
def setup_driver_mock():
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
