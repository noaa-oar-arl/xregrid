import sys
from unittest.mock import MagicMock

import numpy as np
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

from xregrid import Regridder, create_global_grid  # noqa: E402


class UxDatasetMock:
    def __init__(self, ds, uxgrid):
        self._ds = ds
        self.uxgrid = uxgrid

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __getitem__(self, key):
        return self._ds[key]

    @property
    def data_vars(self):
        return self._ds.data_vars

    @property
    def coords(self):
        return self._ds.coords

    @property
    def dims(self):
        return self._ds.dims

    @property
    def sizes(self):
        return self._ds.sizes


def test_uxarray_support():
    # 1. Create a mocked uxarray object
    # For real ESMF, all nodes must be used in connectivity.
    # We'll create a simple strip of 10 triangles using 12 nodes.
    n_face = 10
    n_node = 12

    mock_uxgrid = MagicMock()
    mock_uxgrid.node_lat = xr.DataArray(np.linspace(-90, 90, n_node), dims=["n_node"])
    mock_uxgrid.node_lon = xr.DataArray(np.linspace(0, 360, n_node), dims=["n_node"])
    mock_uxgrid.face_lat = xr.DataArray(np.linspace(-90, 90, n_face), dims=["n_face"])
    mock_uxgrid.face_lon = xr.DataArray(np.linspace(0, 360, n_face), dims=["n_face"])

    # Create connectivity: (0,1,2), (1,2,3), (2,3,4), ...
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, i + 1, i + 2]

    mock_uxgrid.face_node_connectivity = xr.DataArray(
        conn, dims=["n_face", "n_max_face_nodes"]
    )
    mock_uxgrid.face_node_connectivity.attrs["start_index"] = 0
    mock_uxgrid.face_node_connectivity.attrs["_FillValue"] = -1

    # Mock UxDataset
    ds_base = xr.Dataset({"test_var": (["n_face"], np.random.rand(n_face))})
    ds = UxDatasetMock(ds_base, mock_uxgrid)

    target_grid = create_global_grid(10, 10)

    # Try bilinear
    regridder = Regridder(ds, target_grid, method="nearest_s2d")
    res = regridder(ds["test_var"])

    assert res.shape == (18, 36)

    # Try conservative
    regridder_cons = Regridder(ds, target_grid, method="conservative")
    res_cons = regridder_cons(ds["test_var"])
    assert res_cons.shape == (18, 36)
