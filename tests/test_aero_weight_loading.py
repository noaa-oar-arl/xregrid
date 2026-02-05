import os
import pytest
import numpy as np
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_aero_weight_loading_roundtrip(tmp_path):
    """
    Test the from_weights factory method and enhanced validation.
    Verifies both NumPy and Dask data paths as per Aero Protocol.
    """
    filename = str(tmp_path / "weights.nc")

    # Create grids
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(5.0, 5.0)

    # 1. Generate and save weights
    # We use a context where esmpy is available (or mocked)
    Regridder(
        ds_src,
        ds_tgt,
        method="bilinear",
        filename=filename,
        reuse_weights=True,
        skipna=True,
        na_thres=0.5,
    )

    # Ensure file was created
    assert os.path.exists(filename)

    # 2. Verify from_weights factory
    regridder2 = Regridder.from_weights(
        filename, ds_src, ds_tgt, method="bilinear", skipna=True, na_thres=0.5
    )

    assert regridder2.method == "bilinear"
    assert regridder2.skipna is True
    assert regridder2.na_thres == 0.5

    # 3. Test application with Eager (NumPy) data
    data_np = np.random.rand(18, 36).astype(np.float32)
    da_np = xr.DataArray(
        data_np,
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        dims=("lat", "lon"),
        name="test",
    )
    res_np = regridder2(da_np)
    # Target shape is (36, 72) for 5 degree global grid
    assert res_np.shape == (36, 72)

    # 4. Test application with Lazy (Dask) data
    da_dask = da_np.chunk({"lat": 9, "lon": 18})
    res_dask = regridder2(da_dask)
    assert res_dask.chunks is not None

    # 5. Assert equality (Eager vs Lazy)
    # Mocked weights might return just a single point or something simple,
    # but the shape and logic should hold.
    np.testing.assert_allclose(res_np.values, res_dask.compute().values)

    # 6. Test Validation failure - Parameter mismatch
    with pytest.raises(ValueError, match="Requested method"):
        Regridder.from_weights(filename, ds_src, ds_tgt, method="nearest_s2d")

    with pytest.raises(ValueError, match="Requested skipna"):
        Regridder.from_weights(filename, ds_src, ds_tgt, skipna=False)

    with pytest.raises(ValueError, match="Requested na_thres"):
        Regridder.from_weights(filename, ds_src, ds_tgt, skipna=True, na_thres=0.9)


def test_aero_encoding_preservation():
    """Verify that encoding is preserved after regridding."""
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(5.0, 5.0)

    regridder = Regridder(ds_src, ds_tgt)

    da_in = xr.DataArray(
        np.random.rand(18, 36),
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        dims=("lat", "lon"),
        name="test",
    )
    da_in.encoding = {"_FillValue": -999.0, "dtype": "float32"}

    res = regridder(da_in)
    assert res.encoding["_FillValue"] == -999.0
    assert res.encoding["dtype"] == "float32"
