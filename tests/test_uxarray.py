import numpy as np
import xarray as xr
from xregrid import Regridder, create_global_grid
from unittest.mock import MagicMock

# Check for real ESMF
try:
    import esmpy

    if hasattr(esmpy, "_is_mock") or "unittest.mock" in str(type(esmpy)):
        raise ImportError
    HAS_REAL_ESMF = True
except ImportError:
    HAS_REAL_ESMF = False


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
    n_face = 10
    n_node = 12

    # Use simple MagicMock for uxgrid since it's not esmpy and not passed to workers usually
    mock_uxgrid = MagicMock()
    mock_uxgrid.node_lat = xr.DataArray(np.linspace(-90, 90, n_node), dims=["n_node"])
    mock_uxgrid.node_lon = xr.DataArray(np.linspace(0, 360, n_node), dims=["n_node"])
    mock_uxgrid.face_lat = xr.DataArray(np.linspace(-90, 90, n_face), dims=["n_face"])
    mock_uxgrid.face_lon = xr.DataArray(np.linspace(0, 360, n_face), dims=["n_face"])

    # Create connectivity
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

    # Try bilinear (nearest_s2d for robust fallback in mock)
    regridder = Regridder(ds, target_grid, method="nearest_s2d")
    res = regridder(ds["test_var"])

    assert res.shape == (18, 36)

    # Try conservative
    regridder_cons = Regridder(ds, target_grid, method="conservative")
    res_cons = regridder_cons(ds["test_var"])
    assert res_cons.shape == (18, 36)
