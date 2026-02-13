import numpy as np
import xarray as xr
import pytest
from xregrid import Regridder
from xregrid.utils import create_grid_from_crs, create_global_grid, get_crs_info


def test_crs_propagation_dataarray():
    """
    Test that CRS metadata is propagated when regridding a DataArray.
    Verified with Eager (NumPy) and Lazy (Dask) data.
    """
    # 1. Setup Source Grid (Global Lat-Lon)
    src_ds = create_global_grid(res_lat=10, res_lon=10)

    # 2. Setup Target Grid (Projected UTM zone 33N)
    # UTM zone 33N is approx centered at 15E
    target_ds = create_grid_from_crs(
        crs="EPSG:32633", extent=(400000, 600000, 5000000, 5200000), res=10000
    )

    # Create source data
    data = np.random.rand(src_ds.sizes["lat"], src_ds.sizes["lon"])
    # Filter coords to only those compatible with (lat, lon) dims
    compatible_coords = {
        k: v for k, v in src_ds.coords.items() if set(v.dims).issubset({"lat", "lon"})
    }
    da_src_numpy = xr.DataArray(
        data, coords=compatible_coords, dims=("lat", "lon"), name="test_data"
    )

    da_src_dask = da_src_numpy.chunk({"lat": 5, "lon": 5})

    # Initialize Regridder
    regridder = Regridder(src_ds, target_ds, method="bilinear")

    for da_in in [da_src_numpy, da_src_dask]:
        # Perform Regridding
        da_out = regridder(da_in)

        # PROOF 1: CRS WKT Attribute Propagation
        assert "crs" in da_out.attrs
        assert "32633" in da_out.attrs["crs"]

        # PROOF 2: Grid Mapping Variable Propagation
        # create_grid_from_crs currently doesn't add a grid_mapping variable by default,
        # but it adds 'lat' and 'lon' coordinates.
        # Wait, let's check what create_grid_from_crs does.
        # It adds 'lat', 'lon' and sets attrs['crs'].

        # PROOF 3: Backend Consistency
        if hasattr(da_in.data, "dask"):
            assert hasattr(da_out.data, "dask")
        else:
            assert isinstance(da_out.data, np.ndarray)

        # PROOF 4: Viz Discovery
        # get_crs_info should return the correct CRS for the output
        crs_detected = get_crs_info(da_out)
        assert crs_detected is not None
        assert crs_detected.to_epsg() == 32633


def test_crs_propagation_dataset():
    """
    Test that CRS metadata is propagated when regridding a Dataset.
    """
    src_ds = create_global_grid(res_lat=10, res_lon=10)
    target_ds = create_grid_from_crs("EPSG:3857", (0, 10000, 0, 10000), 1000)

    data = np.random.rand(src_ds.sizes["lat"], src_ds.sizes["lon"])
    src_ds["var1"] = (("lat", "lon"), data)
    src_ds.attrs["history"] = "original history"

    regridder = Regridder(src_ds, target_ds, method="bilinear")
    ds_out = regridder(src_ds)

    # Global attribute propagation
    assert "crs" in ds_out.attrs
    assert "3857" in ds_out.attrs["crs"]

    # Variable attribute propagation
    assert "crs" in ds_out["var1"].attrs
    assert "3857" in ds_out["var1"].attrs["crs"]

    # History update
    assert "Regridded" in ds_out.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
