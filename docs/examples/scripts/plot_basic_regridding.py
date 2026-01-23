"""
Basic Rectilinear Grid Regridding
=================================

This example demonstrates the most common use case: regridding between
rectilinear latitude-longitude grids, such as those used in atmospheric
climate models.

We'll regrid synthetic temperature data from a coarse 1° grid to a finer
0.5° grid using bilinear interpolation.

Key concepts demonstrated:
- Creating rectilinear grids
- Using the ESMPyRegridder with bilinear method
- Handling global periodicity
- Visualizing regridding results
"""

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for gallery
import matplotlib.pyplot as plt
from xregrid import ESMPyRegridder

# Create source grid (1° resolution)
source_lats = np.linspace(-90, 90, 180)
source_lons = np.linspace(0, 359, 360)

source_grid = xr.Dataset({"lat": (["lat"], source_lats), "lon": (["lon"], source_lons)})

# Create target grid (0.5° resolution)
target_lats = np.linspace(-90, 90, 360)
target_lons = np.linspace(0, 359.5, 720)

target_grid = xr.Dataset({"lat": (["lat"], target_lats), "lon": (["lon"], target_lons)})

print(f"Source grid shape: {len(source_lats)} x {len(source_lons)}")
print(f"Target grid shape: {len(target_lats)} x {len(target_lons)}")

# Create synthetic temperature data with realistic spatial patterns
lons_2d, lats_2d = np.meshgrid(source_lons, source_lats)

# Create a temperature field with:
# - Latitudinal gradient (warmer at equator)
# - Seasonal cycle
# - Some realistic spatial patterns
temperature_pattern = (
    20 * np.cos(np.radians(lats_2d))  # Latitudinal gradient
    + 5
    * np.sin(2 * np.radians(lons_2d))
    * np.cos(np.radians(lats_2d))  # Longitudinal variation
    + np.random.normal(0, 2, lats_2d.shape)  # Random noise
)

# Add time dimension (12 months)
times = np.arange(12)
temperature_data = np.zeros((12, len(source_lats), len(source_lons)))

for i, month in enumerate(times):
    # Add seasonal cycle (stronger in NH)
    seasonal = 10 * np.cos(2 * np.pi * (month - 6) / 12) * np.maximum(0, lats_2d / 90)
    temperature_data[i] = temperature_pattern + seasonal

# Create xarray DataArray
temperature = xr.DataArray(
    temperature_data,
    dims=["time", "lat", "lon"],
    coords={"time": times, "lat": source_lats, "lon": source_lons},
    attrs={
        "units": "degrees_C",
        "long_name": "Surface Temperature",
        "standard_name": "air_temperature",
    },
)

print(f"\nCreated temperature data with shape: {temperature.shape}")
print(
    f"Temperature range: {temperature.min().values:.1f} to {temperature.max().values:.1f} °C"
)

# Create the regridder
print("\nCreating regridder...")
regridder = ESMPyRegridder(
    source_grid,
    target_grid,
    method="bilinear",
    periodic=True,  # Important for global grids!
)

# Apply regridding
print("Regridding temperature data...")
temp_regridded = regridder(temperature)

print(f"\nRegridded data shape: {temp_regridded.shape}")
print(
    f"Regridded temperature range: {temp_regridded.min().values:.1f} to {temp_regridded.max().values:.1f} °C"
)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Rectilinear Grid Regridding: 1° → 0.5°", fontsize=16)

# Original data (January)
im1 = axes[0, 0].pcolormesh(
    source_lons, source_lats, temperature.isel(time=0), shading="auto", cmap="RdYlBu_r"
)
axes[0, 0].set_title("Original 1° Grid (January)")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[0, 0], label="Temperature (°C)")

# Regridded data (January)
im2 = axes[0, 1].pcolormesh(
    target_lons,
    target_lats,
    temp_regridded.isel(time=0),
    shading="auto",
    cmap="RdYlBu_r",
)
axes[0, 1].set_title("Regridded 0.5° Grid (January)")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")
plt.colorbar(im2, ax=axes[0, 1], label="Temperature (°C)")

# Difference (July)
original_july = temperature.isel(time=6)
regridded_july = temp_regridded.isel(time=6)

# Interpolate original to target grid for comparison
original_interp = original_july.interp(lat=target_lats, lon=target_lons)
difference = regridded_july - original_interp

im3 = axes[1, 0].pcolormesh(
    target_lons,
    target_lats,
    difference,
    shading="auto",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5,
)
axes[1, 0].set_title("Difference vs Linear Interpolation (July)")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
plt.colorbar(im3, ax=axes[1, 0], label="Temperature Difference (°C)")

# Time series comparison at a point
point_lat, point_lon = 45.0, 180.0  # 45°N, 180°E

# Find nearest grid points
src_lat_idx = np.argmin(np.abs(source_lats - point_lat))
src_lon_idx = np.argmin(np.abs(source_lons - point_lon))
tgt_lat_idx = np.argmin(np.abs(target_lats - point_lat))
tgt_lon_idx = np.argmin(np.abs(target_lons - point_lon))

original_ts = temperature[:, src_lat_idx, src_lon_idx]
regridded_ts = temp_regridded[:, tgt_lat_idx, tgt_lon_idx]

axes[1, 1].plot(times, original_ts, "o-", label="Original 1°", linewidth=2)
axes[1, 1].plot(times, regridded_ts, "s-", label="Regridded 0.5°", linewidth=2)
axes[1, 1].set_title(f"Time Series at {point_lat}°N, {point_lon}°E")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Temperature (°C)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("basic_regridding.png", dpi=150, bbox_inches="tight")
print("\nSaved plot as 'basic_regridding.png'")

# Performance demonstration
import time  # noqa: E402

print("\n" + "=" * 50)
print("PERFORMANCE DEMONSTRATION")
print("=" * 50)

# Time the regridding operation
start_time = time.time()
for i in range(10):  # Multiple iterations for better timing
    _ = regridder(temperature)
end_time = time.time()

average_time = (end_time - start_time) / 10
print(f"Average regridding time: {average_time:.4f} seconds")
print(f"Points processed per second: {temperature.size / average_time:,.0f}")

# Memory usage estimate
memory_mb = temperature.nbytes / 1024 / 1024
print(f"Input data size: {memory_mb:.1f} MB")

output_memory_mb = temp_regridded.nbytes / 1024 / 1024
print(f"Output data size: {output_memory_mb:.1f} MB")

print("\nRegridding summary:")
print("- Method: Bilinear interpolation")
print(f"- Source resolution: 1.0° ({len(source_lats)}x{len(source_lons)})")
print(f"- Target resolution: 0.5° ({len(target_lats)}x{len(target_lons)})")
print(
    f"- Grid ratio: {len(target_lats)*len(target_lons) / (len(source_lats)*len(source_lons)):.1f}x more points"
)
print("- Global periodicity: Enabled")
print(f"- Processing time: {average_time*1000:.1f} ms per regridding")

# Show plot if running interactively
try:
    if __name__ == "__main__":
        plt.show()
except Exception:
    pass
