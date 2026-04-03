import numpy as np
import xarray as xr
from xregrid import Regridder, create_global_grid


def test_aux_coord_regrid_optimization():
    """
    Verify that auxiliary spatial coordinates are correctly regridded
    and results are identical between Eager (NumPy) and Lazy (Dask) backends.
    """
    # 1. Setup grids
    src_grid = create_global_grid(10, 10)
    tgt_grid = create_global_grid(5, 5)

    # 2. Create source dataset with auxiliary spatial coordinates
    # An auxiliary spatial coordinate depends on the same dimensions as the data
    lat_2d, lon_2d = xr.broadcast(src_grid.lat, src_grid.lon)
    aux_coord = (lat_2d + lon_2d).rename("aux_spatial")

    ds_src = xr.Dataset(
        data_vars={
            "var1": (("lat", "lon"), np.random.rand(18, 36)),
            "var2": (("lat", "lon"), np.random.rand(18, 36)),
        },
        coords={"lat": src_grid.lat, "lon": src_grid.lon, "aux_spatial": aux_coord},
    )

    # 3. Eager Execution
    regridder_eager = Regridder(ds_src, tgt_grid, method="bilinear")
    ds_out_eager = regridder_eager(ds_src)

    # Verify auxiliary coordinate exists and is regridded
    assert "aux_spatial" in ds_out_eager.coords
    assert ds_out_eager.aux_spatial.shape == (36, 72)

    # 4. Lazy Execution (Dask)
    ds_src_lazy = ds_src.chunk({"lat": 9, "lon": 18})
    regridder_lazy = Regridder(ds_src_lazy, tgt_grid, method="bilinear")
    ds_out_lazy = regridder_lazy(ds_src_lazy)

    # Verify laziness
    assert hasattr(ds_out_lazy.var1.data, "dask")
    assert hasattr(ds_out_lazy.aux_spatial.data, "dask")

    # Compute and compare
    ds_out_lazy_computed = ds_out_lazy.compute()

    xr.testing.assert_allclose(ds_out_eager, ds_out_lazy_computed)

    # 5. Verify that the regridded auxiliary coordinate is the same for all variables
    # (Checking the internal optimization indirectly by ensuring consistency)
    xr.testing.assert_allclose(ds_out_eager.var1.aux_spatial, ds_out_eager.aux_spatial)
    xr.testing.assert_allclose(ds_out_eager.var2.aux_spatial, ds_out_eager.aux_spatial)


if __name__ == "__main__":
    test_aux_coord_regrid_optimization()
