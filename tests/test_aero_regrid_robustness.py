import xarray as xr
import numpy as np
import pytest
from xregrid import Regridder
from xregrid.utils import create_global_grid


def test_aero_lazy_initialization():
    """
    Aero Protocol: Verify that Regridder initialization doesn't force
    computation of Dask-backed coordinates when possible.
    Note: Xarray always computes dimension coordinates to build indexes.
    To test laziness, we use non-dimension coordinates.
    """
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    # Create an unstructured grid with Dask-backed coordinates
    # Unstructured coordinates share the same dimension name
    n_pts = 100
    lat_vals = np.linspace(-90, 90, n_pts)
    lon_vals = np.linspace(0, 360, n_pts)

    lat = da.from_array(lat_vals, chunks=10)
    lon = da.from_array(lon_vals, chunks=10)

    ds_src = xr.Dataset(
        coords={
            "lat": (["n_pts"], lat, {"units": "degrees_north"}),
            "lon": (["n_pts"], lon, {"units": "degrees_east"}),
        }
    )

    ds_tgt = ds_src.copy()

    # Initialize in parallel mode
    regridder = Regridder(ds_src, ds_tgt, parallel=True, compute=False)

    # Check that source_grid_ds coordinates are still dask-backed
    # Unstructured grids should skip _normalize_grid, so they should remain lazy
    assert hasattr(
        regridder.source_grid_ds.lat.data, "dask"
    ), "Source latitude should remain lazy"


def test_aero_double_check_identity():
    """
    Aero Protocol Rule 4: The "Double-Check Test".
    Verify that regridding results are identical for NumPy and Dask backends.
    """
    try:
        import dask.array as da  # noqa: F401
    except ImportError:
        pytest.skip("Dask not installed")

    # 1. Setup grids
    ds_src = create_global_grid(20.0, 20.0)  # Low res for fast test
    ds_tgt = create_global_grid(10.0, 10.0)

    # 2. Setup data (Eager)
    data_eager = np.outer(
        np.cos(np.deg2rad(ds_src.lat.values)), np.sin(np.deg2rad(ds_src.lon.values))
    )
    # Only include dim coords to avoid validation error with bounds
    da_eager = xr.DataArray(
        data_eager,
        dims=["lat", "lon"],
        coords={"lat": ds_src.lat, "lon": ds_src.lon},
        name="test_data",
    )

    # 3. Setup data (Lazy)
    da_lazy = da_eager.chunk({"lat": 5, "lon": 10})

    # 4. Initialize Regridder
    regridder = Regridder(ds_src, ds_tgt, method="bilinear")

    # 5. Regrid both
    res_eager = regridder(da_eager)
    res_lazy = regridder(da_lazy)

    # 6. Assertions
    assert isinstance(res_eager.data, np.ndarray)
    assert hasattr(res_lazy.data, "dask")

    # Compare values
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values, rtol=1e-6)


def test_aero_diagnostics_crs_propagation():
    """
    Aero Protocol: Verify CRS propagation in diagnostics.
    """
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(5.0, 5.0)

    # Attach a mock CRS
    ds_tgt.attrs["crs"] = "EPSG:4326"

    regridder = Regridder(ds_src, ds_tgt, method="bilinear")
    ds_diag = regridder.diagnostics()

    assert "crs" in ds_diag.attrs
    # Check for EPSG 4326 in the WKT string
    assert "4326" in ds_diag.attrs["crs"]


if __name__ == "__main__":
    pytest.main([__file__])
