import numpy as np
import pytest
import xarray as xr
from xregrid.grid import _clip_latitudes, _normalize_longitudes
from xregrid.regridder import Regridder
from xregrid.utils import create_global_grid


def test_clip_latitudes_double_check():
    """
    Double-Check Test for latitude clipping.
    """
    lat_vals = np.array([-90.1, -45.0, 0.0, 45.0, 90.1])
    lat_da = xr.DataArray(lat_vals, dims="lat", name="lat")

    # 1. Eager (NumPy)
    clipped_eager = _clip_latitudes(lat_da)
    assert np.all(clipped_eager >= -90.0)
    assert np.all(clipped_eager <= 90.0)
    assert clipped_eager[0] == -90.0
    assert clipped_eager[-1] == 90.0

    # 2. Lazy (Dask)
    lat_da_lazy = lat_da.chunk({"lat": 2})
    clipped_lazy = _clip_latitudes(lat_da_lazy)

    # Assert laziness
    assert hasattr(clipped_lazy.data, "dask")

    # Assert identity
    xr.testing.assert_allclose(clipped_eager, clipped_lazy.compute())


def test_normalize_longitudes_double_check():
    """
    Double-Check Test for longitude normalization.
    """
    lon_vals = np.array([-10.0, 0.0, 180.0, 360.0, 370.0])
    lon_da = xr.DataArray(lon_vals, dims="lon", name="lon")

    # 1. Eager (NumPy)
    norm_eager = _normalize_longitudes(lon_da)
    expected = np.array([350.0, 0.0, 180.0, 0.0, 10.0])
    np.testing.assert_allclose(norm_eager.values, expected)

    # 2. Lazy (Dask)
    lon_da_lazy = lon_da.chunk({"lon": 2})
    norm_lazy = _normalize_longitudes(lon_da_lazy)

    # Assert laziness
    assert hasattr(norm_lazy.data, "dask")

    # Assert identity
    xr.testing.assert_allclose(norm_eager, norm_lazy.compute())


def test_regridder_with_out_of_range_coords():
    """
    Verify that Regridder handles out-of-range coordinates gracefully.
    """
    # Create grid with slightly out-of-range latitude
    ds_src = create_global_grid(10, 10)
    # Manually corrupt one latitude value
    lats = ds_src["lat"].values.copy()
    lats[0] = -90.0001
    lats[-1] = 90.0001
    ds_src = ds_src.assign_coords(lat=(("lat",), lats, ds_src["lat"].attrs))

    ds_tgt = create_global_grid(5, 5)

    # This should not raise ESMF_RC_VAL_OUTOFRANGE because of clipping
    regridder = Regridder(ds_src, ds_tgt, method="bilinear")

    data = xr.DataArray(
        np.random.rand(ds_src.sizes["lat"], ds_src.sizes["lon"]),
        coords={"lat": ds_src["lat"], "lon": ds_src["lon"]},
        dims=("lat", "lon"),
        name="test_data",
    )

    # Eager regridding
    out_eager = regridder(data)
    assert not out_eager.isnull().all()

    # Lazy regridding
    data_lazy = data.chunk({"lat": 5})
    out_lazy = regridder(data_lazy)
    assert hasattr(out_lazy.data, "dask")

    xr.testing.assert_allclose(out_eager, out_lazy.compute())


if __name__ == "__main__":
    pytest.main([__file__])
