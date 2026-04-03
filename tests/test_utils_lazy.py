import numpy as np
import xarray as xr
from xregrid.utils import create_grid_from_crs, create_mesh_from_coords


def test_create_grid_from_crs_lazy():
    """
    Double-Check Test for create_grid_from_crs: Eager (NumPy) vs Lazy (Dask).

    Ensures that the refactored lazy transformation produces identical results
    to the eager computation.
    """
    # Use a simple projected CRS (Web Mercator)
    crs = "EPSG:3857"
    extent = (0, 1000000, 0, 1000000)
    res = 100000  # 100km resolution for test

    # 1. Eager (NumPy) - Default
    ds_eager = create_grid_from_crs(crs, extent, res, add_bounds=True)

    # 2. Lazy (Dask) - By providing chunks
    ds_lazy = create_grid_from_crs(crs, extent, res, add_bounds=True, chunks=5)

    # Verify backend identity (Aero Protocol requirement)
    assert not hasattr(ds_eager.lat.data, "dask"), "Eager lat should be NumPy"
    assert hasattr(ds_lazy.lat.data, "dask"), "Lazy lat should be Dask-backed"
    assert hasattr(ds_lazy.lon.data, "dask"), "Lazy lon should be Dask-backed"
    assert hasattr(ds_lazy.lat_b.data, "dask"), "Lazy lat_b should be Dask-backed"
    assert hasattr(ds_lazy.lon_b.data, "dask"), "Lazy lon_b should be Dask-backed"

    # 3. Assert value identity
    # We use a small tolerance for floating point variations if any,
    # but since it's the same core function it should be exact.
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())


def test_create_mesh_from_coords_lazy():
    """
    Double-Check Test for create_mesh_from_coords: Eager (NumPy) vs Lazy (Dask).
    """
    crs = "EPSG:3857"
    x = np.linspace(0, 1000000, 10)
    y = np.linspace(0, 1000000, 10)

    # 1. Eager (NumPy)
    ds_eager = create_mesh_from_coords(x, y, crs)

    # 2. Lazy (Dask)
    ds_lazy = create_mesh_from_coords(x, y, crs, chunks=5)

    # Verify backend identity
    assert not hasattr(ds_eager.lat.data, "dask")
    assert hasattr(ds_lazy.lat.data, "dask")

    # 3. Assert value identity
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())
