import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xregrid import ESMPyRegridder


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

    regridder = ESMPyRegridder(source_ds, target_ds, method="bilinear")
    regridded = regridder(source_ds["temperature"])

    assert regridded.shape == (60, 120)
    assert not np.isnan(regridded.values).any()
    assert regridded.min() >= source_ds.temperature.min() - 0.1
    assert regridded.max() <= source_ds.temperature.max() + 0.1


def test_rectilinear_regrid_dask_non_core_chunked():
    source_ds_np_full = create_sample_dataset(nlat=45, nlon=90, dask=False)
    source_ds_da = create_sample_dataset(nlat=45, nlon=90, dask=True, chunk_core=False)
    target_ds = create_sample_dataset(nlat=60, nlon=120)

    regridder = ESMPyRegridder(source_ds_np_full, target_ds, method="bilinear")

    regridded_da = regridder(source_ds_da["temperature"])

    assert isinstance(regridded_da.data, da.Array)
    assert regridded_da.shape == (1, 60, 120)

    result = regridded_da.compute()
    assert not np.isnan(result.values).any()


def test_rectilinear_regrid_dask_core_chunked():
    source_ds_da = create_sample_dataset(nlat=45, nlon=90, dask=True, chunk_core=True)
    target_ds = create_sample_dataset(nlat=60, nlon=120)
    regridder = ESMPyRegridder(source_ds_da, target_ds, method="bilinear")
    regridded_da = regridder(source_ds_da["temperature"])

    assert isinstance(regridded_da.data, da.Array)
    result = regridded_da.compute()
    assert result.shape == (60, 120)
    assert not np.isnan(result.values).any()


def test_regrid_timing(benchmark):
    source_ds = create_sample_dataset(nlat=180, nlon=360)
    target_ds = create_sample_dataset(nlat=360, nlon=720)

    regridder = ESMPyRegridder(source_ds, target_ds, method="bilinear")

    def do_regrid():
        return regridder(source_ds["temperature"]).values

    benchmark(do_regrid)


def test_provenance():
    source_ds = create_sample_dataset(nlat=10, nlon=20)
    target_ds = create_sample_dataset(nlat=15, nlon=25)
    regridder = ESMPyRegridder(source_ds, target_ds)
    regridded = regridder(source_ds["temperature"])

    assert "history" in regridded.attrs
    assert "ESMPyRegridder" in regridded.attrs["history"]
    assert "bilinear" in regridded.attrs["history"]


def test_type_hints():
    # Basic check that the class has expected annotations
    # With from __future__ import annotations, they might be strings
    ann = ESMPyRegridder.__init__.__annotations__["method"]
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
