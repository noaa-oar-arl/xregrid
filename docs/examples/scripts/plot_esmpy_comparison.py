"""
ESMPy vs. XRegrid Comparison
============================

This example demonstrates the significant reduction in code complexity
when using XRegrid compared to the raw ESMPy library.

While ESMPy provides the powerful low-level engine for regridding,
it is a low-level library that requires significant boilerplate code
to use with xarray. XRegrid bridges this gap by providing a high-level
API while delivering better performance than other wrappers.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from xregrid import Regridder

# --- Part 1: Load Sample Data ---
# We use the standard air temperature tutorial dataset
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

# Define a target grid (e.g., 1.0° resolution)
target_lat = np.arange(15, 76, 1.0)
target_lon = np.arange(200, 331, 1.0)
target_grid = xr.Dataset(
    {
        "lat": (["lat"], target_lat, {"units": "degrees_north"}),
        "lon": (["lon"], target_lon, {"units": "degrees_east"}),
    }
)

# --- Part 2: Regridding with XRegrid ---
# In XRegrid, the entire process of creating grids, fields, and
# applying the regrid is abstracted into a two-step process.
regridder = Regridder(ds, target_grid, method="bilinear")
air_regridded = regridder(ds.air)

print("XRegrid regridding complete.")
print(f"Original shape: {ds.air.shape}")
print(f"Regridded shape: {air_regridded.shape}")


# --- Part 3: Comparison with raw ESMPy ---
# Below is a representative snippet of what would be required to
# perform the same operation using raw ESMPy.

"""
# The ESMPy equivalent would look something like this:

import esmpy

# 1. Create Source Grid (requires manual meshgrid and transposition)
src_grid = esmpy.Grid(
    np.array([ds.lon.size, ds.lat.size]),
    staggerloc=[esmpy.StaggerLoc.CENTER],
    coord_sys=esmpy.CoordSys.SPH_DEG
)
src_lon_ptr = src_grid.get_coords(0)
src_lat_ptr = src_grid.get_coords(1)
lon_mesh, lat_mesh = np.meshgrid(ds.lon.values, ds.lat.values)
src_lon_ptr[...] = lon_mesh.T  # ESMF uses (lon, lat) / Fortran order
src_lat_ptr[...] = lat_mesh.T

# 2. Create Target Grid
dst_grid = esmpy.Grid(
    np.array([len(target_lon), len(target_lat)]),
    staggerloc=[esmpy.StaggerLoc.CENTER],
    coord_sys=esmpy.CoordSys.SPH_DEG
)
dst_lon_ptr = dst_grid.get_coords(0)
dst_lat_ptr = dst_grid.get_coords(1)
lon_mesh_dst, lat_mesh_dst = np.meshgrid(target_lon, target_lat)
dst_lon_ptr[...] = lon_mesh_dst.T
dst_lat_ptr[...] = lat_mesh_dst.T

# 3. Create Fields
src_field = esmpy.Field(src_grid, name="air")
dst_field = esmpy.Field(dst_grid, name="air_regridded")

# 4. Initialize Regrid object
regrid = esmpy.Regrid(src_field, dst_field, regrid_method=esmpy.RegridMethod.BILINEAR)

# 5. Apply Regrid (requires manual data copy and transposition)
src_field.data[...] = ds.air.values.T
regrid(src_field, dst_field)

# 6. Extract result back to xarray
result = xr.DataArray(
    dst_field.data.T,
    coords={"lat": target_lat, "lon": target_lon},
    dims=("lat", "lon")
)
"""

# --- Part 4: Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ds.air.plot(ax=ax1, cmap="magma")
ax1.set_title(f"Original Data\n{ds.air.shape}")

air_regridded.plot(ax=ax2, cmap="magma")
ax2.set_title(f"XRegrid Result\n{air_regridded.shape}")

plt.tight_layout()
plt.show()
