import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from xregrid import Regridder, create_global_grid
from xregrid.viz import plot_static

# 1. Create toy data
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)
target_grid = create_global_grid(1.0, 1.0)

# 2. Regrid
regridder = Regridder(ds, target_grid, method="bilinear", periodic=True)
da_regridded = regridder(ds.air)

# 3. Visualization - Track A: Static (Publication)
print("Generating static plot...")
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
plot_static(da_regridded, ax=ax, cmap="RdBu_r")
ax.coastlines()
plt.savefig("regrid_static.png")
print("Static plot saved to regrid_static.png")

# 4. Visualization - Track B: Interactive (Exploration)
# (In a notebook environment)
# plot_interactive(da_regridded, rasterize=True)
