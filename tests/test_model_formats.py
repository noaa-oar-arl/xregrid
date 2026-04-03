import numpy as np
import pytest
import xarray as xr
from xregrid.grid import _get_mesh_info


def test_cam_se_discovery():
    """Verify discovery of CAM-SE (ncol) unstructured grids."""
    ds = xr.Dataset(
        data_vars={"temp": (["ncol"], np.random.rand(10))},
        coords={
            "lat": (["ncol"], np.linspace(-90, 90, 10)),
            "lon": (["ncol"], np.linspace(0, 350, 10)),
        },
    )
    # _get_mesh_info should identify this as unstructured
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    assert is_unstructured
    assert dims == ("ncol",)
    assert shape == (10,)


def test_cam_fv_discovery():
    """Verify discovery of CAM-fv (LAT/LON) rectilinear grids."""
    ds = xr.Dataset(
        data_vars={"temp": (["LAT", "LON"], np.random.rand(5, 10))},
        coords={
            "LAT": (["LAT"], np.linspace(-90, 90, 5)),
            "LON": (["LON"], np.linspace(0, 350, 10)),
        },
    )
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    assert not is_unstructured
    assert shape == (5, 10)
    assert dims == ("LAT", "LON")


def test_scrip_discovery():
    """Verify discovery of SCRIP unstructured grids."""
    ds = xr.Dataset(
        data_vars={"temp": (["grid_size"], np.random.rand(10))},
        coords={
            "grid_center_lat": (["grid_size"], np.linspace(-90, 90, 10)),
            "grid_center_lon": (["grid_size"], np.linspace(0, 350, 10)),
        },
    )
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    assert is_unstructured
    assert dims == ("grid_size",)


if __name__ == "__main__":
    pytest.main([__file__])
