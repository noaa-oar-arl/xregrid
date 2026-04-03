"""
Example: Fixing Dateline Artifacts in Unstructured Regridding
===========================================================

This example demonstrates how xregrid automatically handles dateline
(180/-180 meridian) crossing for unstructured grids, preventing
the "streaking" artifacts often seen in Euclidean interpolations.
"""

import numpy as np
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid


def run_example():
    # 1. Create a synthetic unstructured swath crossing the dateline
    # We'll create points from 170E to 190E (which is 170W)
    n_pts = 1000
    lon_raw = np.linspace(170, 190, n_pts)
    # Normalize to [-180, 180]
    lon = (lon_raw + 180) % 360 - 180
    lat = np.linspace(20, 30, n_pts)

    # Create the source Dataset
    ds_src = xr.Dataset(
        coords={
            "lon": (
                "n_pts",
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
            "lat": (
                "n_pts",
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
        }
    )
    # Synthetic data: a gradient along the swath
    ds_src["swath_data"] = (["n_pts"], np.linspace(0, 1, n_pts))

    # 2. Create a global 1-degree target grid
    ds_tgt = create_global_grid(1.0, 1.0)

    # 3. Regrid
    # Aero Note: The Regridder now automatically detects geographic coordinates
    # and uses SPH_DEG (Spherical Degrees) instead of CART (Cartesian).
    # This ensures that the distance between 179E and -179W is ~2 degrees,
    # not ~358 degrees!
    regridder = Regridder(ds_src, ds_tgt, method="nearest_s2d")

    # Apply regridding
    ds_regrid = regridder(ds_src)

    print("Regridding complete.")
    print(f"Output Variables: {list(ds_regrid.data_vars)}")
    print(
        f"Source Longitude Range: {ds_src.lon.min().values:.1f} to {ds_src.lon.max().values:.1f}"
    )
    print("Coordinates crossing the dateline are handled correctly by using SPH_DEG.")

    # In a real environment with matplotlib:
    # regridder.plot_comparison(ds_src.swath_data, ds_regrid.swath_data)


if __name__ == "__main__":
    run_example()
