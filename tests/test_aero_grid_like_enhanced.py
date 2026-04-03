import numpy as np
import xarray as xr
from xregrid.utils import create_grid_like


def test_create_grid_like_enhanced_provenance():
    """
    Aero Protocol: Double-Check Test for create_grid_like.
    Verifies that values are identical between NumPy and Dask backends,
    and checks that history propagation is correctly handled.
    """
    # Create template with history
    template = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 20),
        },
        name="my_data",
    )
    template.attrs["history"] = "Original data history."

    res = 5

    # Eager (NumPy)
    ds_eager = create_grid_like(template, res=res, chunks=None)
    assert "Template history:\nOriginal data history." in ds_eager.attrs["history"]

    # Template has 10 pts from -90 to 90 -> res=20.
    # Extent becomes [-100, 100] for lat, and [-9.47, 369.47] approx for lon
    # Actually for 20 pts over 0-360, res is 360/19 approx 18.9.
    assert ds_eager.lat.size == 200 // res

    # Lazy (Dask)
    ds_lazy = create_grid_like(template, res=res, chunks={"lat": 10, "lon": 10})
    assert ds_lazy.chunks
    assert "Template history:\nOriginal data history." in ds_lazy.attrs["history"]

    # Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify internal backend (non-index coordinates like bounds should be dask-backed)
    assert hasattr(ds_lazy.lat_b.data, "dask")
    assert hasattr(ds_lazy.lon_b.data, "dask")


def test_create_grid_like_override_extent():
    """
    Test create_grid_like with explicit extent override to avoid computes.
    """
    template = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 20),
        },
    )

    # Override extent
    extent = (0, 180, -45, 45)  # min_lon, max_lon, min_lat, max_lat
    ds = create_grid_like(template, res=10, extent=extent)

    assert ds.lat.min() == -40  # -45 + 10/2
    assert ds.lon.min() == 5  # 0 + 10/2
    assert "(Override Extent)." in ds.attrs["history"]


def test_create_grid_like_size_1():
    """
    Test create_grid_like with size-1 dimension to verify IndexError fix.
    """
    # Eager case
    template = xr.DataArray(
        np.random.rand(1, 20),
        dims=["lat", "lon"],
        coords={
            "lat": [0.0],
            "lon": np.linspace(0, 360, 20),
        },
    )

    # This should not raise IndexError
    ds = create_grid_like(template, res=10)
    assert ds.lon.size > 0

    # Lazy case (triggers dask.compute logic)
    template_lazy = template.chunk({"lon": 10})
    ds_lazy = create_grid_like(template_lazy, res=10)
    assert ds_lazy.lon.size > 0
