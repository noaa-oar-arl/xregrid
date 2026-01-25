import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_grid_from_crs, create_global_grid


def create_sample_dataset(
    nlat=45,
    nlon=90,
    lat_range=(-90, 90),
    lon_range=(0, 360),
    dask=False,
    chunk_core=True,
):
    """Create a sample dataset with synthetic data."""
    lat = np.linspace(lat_range[0], lat_range[1], nlat)
    lon = np.linspace(lon_range[0], lon_range[1], nlon)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    data = np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid * 2))

    if dask:
        if chunk_core:
            data = da.from_array(data, chunks=(nlat // 2, nlon // 2))
        else:
            # Add a non-core dimension to chunk along
            data = data[np.newaxis, :, :]  # (time, lat, lon)
            data = da.from_array(data, chunks=(1, -1, -1))
            ds = xr.Dataset(
                {
                    "temperature": (["time", "lat", "lon"], data),
                },
                coords={
                    "time": [0],
                    "lat": (["lat"], lat),
                    "lon": (["lon"], lon),
                },
            )
            return ds

    ds = xr.Dataset(
        {
            "temperature": (["lat", "lon"], data),
        },
        coords={
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
        },
    )
    return ds


def test_rectilinear_regrid_numpy():
    source_ds = create_sample_dataset(nlat=45, nlon=90)
    target_ds = create_sample_dataset(nlat=60, nlon=120)

    regridder = Regridder(source_ds, target_ds, method="bilinear")
    regridded = regridder(source_ds["temperature"])

    assert regridded.shape == (60, 120)
    assert not np.isnan(regridded.values).any()
    assert regridded.min() >= source_ds.temperature.min() - 0.1
    assert regridded.max() <= source_ds.temperature.max() + 0.1


def test_rectilinear_regrid_dask_non_core_chunked():
    source_ds_np_full = create_sample_dataset(nlat=45, nlon=90, dask=False)
    source_ds_da = create_sample_dataset(nlat=45, nlon=90, dask=True, chunk_core=False)
    target_ds = create_sample_dataset(nlat=60, nlon=120)

    regridder = Regridder(source_ds_np_full, target_ds, method="bilinear")

    regridded_da = regridder(source_ds_da["temperature"])

    assert isinstance(regridded_da.data, da.Array)
    assert regridded_da.shape == (1, 60, 120)

    result = regridded_da.compute()
    assert not np.isnan(result.values).any()


def test_rectilinear_regrid_dask_core_chunked():
    source_ds_da = create_sample_dataset(nlat=45, nlon=90, dask=True, chunk_core=True)
    target_ds = create_sample_dataset(nlat=60, nlon=120)
    regridder = Regridder(source_ds_da, target_ds, method="bilinear")
    regridded_da = regridder(source_ds_da["temperature"])

    assert isinstance(regridded_da.data, da.Array)
    result = regridded_da.compute()
    assert result.shape == (60, 120)
    assert not np.isnan(result.values).any()


def test_regrid_timing(benchmark):
    source_ds = create_sample_dataset(nlat=180, nlon=360)
    target_ds = create_sample_dataset(nlat=360, nlon=720)

    regridder = Regridder(source_ds, target_ds, method="bilinear")

    def do_regrid():
        return regridder(source_ds["temperature"]).values

    benchmark(do_regrid)


def test_provenance():
    source_ds = create_sample_dataset(nlat=10, nlon=20)
    target_ds = create_sample_dataset(nlat=15, nlon=25)
    regridder = Regridder(source_ds, target_ds)
    regridded = regridder(source_ds["temperature"])

    assert "history" in regridded.attrs
    assert "Regridder" in regridded.attrs["history"]
    assert "bilinear" in regridded.attrs["history"]


def test_type_hints():
    # Basic check that the class has expected annotations
    # With from __future__ import annotations, they might be strings
    ann = Regridder.__init__.__annotations__["method"]
    assert ann == "str" or ann is str


def test_viz_static_call():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
    from xregrid import plot_static

    source_ds = create_sample_dataset(nlat=10, nlon=20)
    # This should work without error even if it just calls da.plot
    plot_static(source_ds["temperature"])
    plt.close("all")


def test_dask_numpy_identity():
    source_ds = create_sample_dataset(nlat=10, nlon=20)
    target_ds = create_sample_dataset(nlat=15, nlon=25)
    regridder = Regridder(source_ds, target_ds)

    # Eager
    da_eager = source_ds["temperature"]
    res_eager = regridder(da_eager)

    # Lazy
    da_lazy = da_eager.chunk({"lat": 5, "lon": 10})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


def test_attribute_preservation():
    source_ds = create_sample_dataset()
    source_ds.temperature.attrs["units"] = "K"
    target_ds = create_sample_dataset(nlat=10, nlon=20)
    regridder = Regridder(source_ds, target_ds)
    out = regridder(source_ds.temperature)
    assert out.attrs["units"] == "K"


def test_plot_static_custom_ax():
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        pytest.skip("matplotlib or cartopy not installed")
    from xregrid import plot_static

    da = create_sample_dataset()["temperature"]
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    plot_static(da, ax=ax)
    plt.close(fig)


def test_regridder_repr():
    source_ds = create_sample_dataset(nlat=10, nlon=20)
    target_ds = create_sample_dataset(nlat=15, nlon=25)
    regridder = Regridder(source_ds, target_ds, method="bilinear", periodic=False)
    rep = repr(regridder)
    assert "Regridder" in rep
    assert "method=bilinear" in rep
    assert "periodic=False" in rep
    assert "(10, 20)" in rep
    assert "(15, 25)" in rep


def test_weights_format():
    from scipy.sparse import csr_matrix

    source_ds = create_sample_dataset(nlat=10, nlon=20)
    target_ds = create_sample_dataset(nlat=15, nlon=25)
    regridder = Regridder(source_ds, target_ds)
    assert isinstance(regridder._weights_matrix, csr_matrix)


def test_regrid_with_crs_grid():
    # Source grid: global 10 degree
    src_ds = create_global_grid(res_lat=10, res_lon=10)
    src_ds["data"] = (("lat", "lon"), np.ones((src_ds.lat.size, src_ds.lon.size)))

    # Target grid: UTM zone 33N
    extent = (400000, 500000, 5000000, 5100000)
    res = 10000
    tgt_ds = create_grid_from_crs("EPSG:32633", extent, res)

    regridder = Regridder(src_ds, tgt_ds, method="bilinear")
    out = regridder(src_ds["data"])

    assert out.shape == (tgt_ds.y.size, tgt_ds.x.size)
    assert "x" in out.coords
    assert "y" in out.coords


def test_dataset_regrid_identity():
    """Double-Check Test: Verify Dataset regridding matches DataArray regridding for both Eager and Lazy."""
    nlat_in, nlon_in = 10, 20
    nlat_out, nlon_out = 15, 25

    source_ds = create_sample_dataset(nlat=nlat_in, nlon=nlon_in)
    # Add another variable
    source_ds["humidity"] = source_ds["temperature"] * 0.8
    # Add a non-spatial variable
    source_ds["scalar"] = xr.DataArray(42.0)

    target_grid = create_sample_dataset(nlat=nlat_out, nlon=nlon_out)
    regridder = Regridder(source_ds, target_grid)

    # 1. Eager test
    res_ds_eager = regridder(source_ds)
    assert isinstance(res_ds_eager, xr.Dataset)
    assert "temperature" in res_ds_eager
    assert "humidity" in res_ds_eager
    assert "scalar" in res_ds_eager
    assert res_ds_eager["temperature"].shape == (nlat_out, nlon_out)
    assert res_ds_eager["humidity"].shape == (nlat_out, nlon_out)

    # Compare with individual DataArray regridding
    res_da_temp = regridder(source_ds["temperature"])
    # Need to remove history for comparison as they differ
    res_ds_temp = res_ds_eager["temperature"].copy()
    res_ds_temp.attrs.pop("history", None)
    res_da_temp_no_hist = res_da_temp.copy()
    res_da_temp_no_hist.attrs.pop("history", None)
    xr.testing.assert_allclose(res_ds_temp, res_da_temp_no_hist)

    # 2. Lazy test
    source_ds_lazy = source_ds.chunk({"lat": 5, "lon": 10})
    res_ds_lazy = regridder(source_ds_lazy).compute()

    # Compare eager and lazy results (ignoring history)
    res_ds_eager_no_hist = res_ds_eager.copy()
    res_ds_lazy_no_hist = res_ds_lazy.copy()
    res_ds_eager_no_hist.attrs.pop("history", None)
    res_ds_lazy_no_hist.attrs.pop("history", None)
    for v in res_ds_eager_no_hist.data_vars:
        res_ds_eager_no_hist[v].attrs.pop("history", None)
        res_ds_lazy_no_hist[v].attrs.pop("history", None)

    xr.testing.assert_allclose(res_ds_eager_no_hist, res_ds_lazy_no_hist)
