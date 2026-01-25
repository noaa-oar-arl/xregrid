import xarray as xr
import numpy as np
import dask.array as da
from xregrid import Regridder


def test_cf_coords_detection():
    # Create dataset with non-standard coordinate names but with CF attributes
    def create_ds(lazy=False):
        np.random.seed(42)
        data = np.random.rand(10, 20)
        if lazy:
            data = da.from_array(data, chunks=(5, 10))

        ds = xr.Dataset(
            {"data": (("lat_dim", "lon_dim"), data)},
            coords={
                "latitude": (
                    ("lat_dim",),
                    np.linspace(-90, 90, 10),
                    {"units": "degrees_north", "standard_name": "latitude"},
                ),
                "longitude": (
                    ("lon_dim",),
                    np.linspace(-180, 180, 20),
                    {"units": "degrees_east", "standard_name": "longitude"},
                ),
            },
        )
        return ds

    ds_tgt = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(-90, 90, 15), {"units": "degrees_north"}),
            "lon": (("lon",), np.linspace(-180, 180, 25), {"units": "degrees_east"}),
        },
    )

    # Test Eager
    ds_src_eager = create_ds(lazy=False)
    regridder_eager = Regridder(ds_src_eager, ds_tgt)
    out_eager = regridder_eager(ds_src_eager["data"])
    assert out_eager.shape == (15, 25)
    assert not out_eager.chunks

    # Test Lazy
    ds_src_lazy = create_ds(lazy=True)
    regridder_lazy = Regridder(ds_src_lazy, ds_tgt)
    out_lazy = regridder_lazy(ds_src_lazy["data"])
    assert out_lazy.shape == (15, 25)
    assert out_lazy.chunks

    # Verify results are identical (within float precision)
    np.testing.assert_allclose(out_eager.values, out_lazy.compute().values)


def test_cf_bounds_detection():
    # Create dataset with non-standard bound names but with CF attributes
    ds_src = xr.Dataset(
        {"data": (("lat", "lon"), np.random.rand(10, 20))},
        coords={
            "lat": (
                ("lat",),
                np.linspace(-90, 90, 10),
                {"units": "degrees_north", "bounds": "lat_bounds"},
            ),
            "lon": (
                ("lon",),
                np.linspace(-180, 180, 20),
                {"units": "degrees_east", "bounds": "lon_bounds"},
            ),
            "lat_bounds": (("lat", "nv"), np.random.rand(10, 2)),  # Placeholder bounds
            "lon_bounds": (("lon", "nv"), np.random.rand(20, 2)),  # Placeholder bounds
        },
    )

    # We need to make the bounds contiguous for our converter to work correctly in this test
    lat_edges = np.linspace(-90, 90, 11)
    lat_bounds = np.stack([lat_edges[:-1], lat_edges[1:]], axis=1)
    lon_edges = np.linspace(-180, 180, 21)
    lon_bounds = np.stack([lon_edges[:-1], lon_edges[1:]], axis=1)

    ds_src.coords["lat_bounds"] = (("lat", "nv"), lat_bounds)
    ds_src.coords["lon_bounds"] = (("lon", "nv"), lon_bounds)

    ds_tgt = xr.Dataset(
        coords={
            "lat": (("lat",), np.linspace(-90, 90, 15), {"units": "degrees_north"}),
            "lon": (("lon",), np.linspace(-180, 180, 25), {"units": "degrees_east"}),
        },
    )

    regridder = Regridder(ds_src, ds_tgt, method="conservative")
    # If it reached here without error, it found the bounds and ESMPy initialized
    out = regridder(ds_src["data"])
    assert out.shape == (15, 25)
