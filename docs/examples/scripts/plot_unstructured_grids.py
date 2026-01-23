"""
Unstructured Grid Regridding (MPAS/ICON)
========================================

This example demonstrates regridding from unstructured grids, which are used
by next-generation climate models like MPAS (Model for Prediction Across Scales)
and ICON (ICOsahedral Nonhydrostatic).

Unstructured grids have advantages:
- Variable resolution (refined in regions of interest)
- No polar singularities
- Efficient parallel computation
- Quasi-uniform grid spacing

Key concepts demonstrated:
- Creating synthetic unstructured grid data
- Regridding from unstructured to structured grids
- Handling 1D spatial dimensions
- Visualizing unstructured data
"""

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for gallery
import matplotlib.pyplot as plt
from xregrid import ESMPyRegridder

# Create a synthetic unstructured grid (simplified icosahedral-like)
np.random.seed(42)  # For reproducible results


# Generate points on a sphere using a spiral pattern (Fibonacci sphere)
def fibonacci_sphere(n_points):
    """Generate approximately uniform points on a sphere using Fibonacci spiral"""
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    # Generate points
    i = np.arange(0, n_points, dtype=float) + 0.5

    # Latitude: from -90 to 90
    lat = np.arcsin(1 - 2 * i / n_points)

    # Longitude: spiral pattern
    lon = (i * golden_angle) % (2 * np.pi)

    return np.degrees(lon), np.degrees(lat)


# Create unstructured grid with ~10,000 points (similar to coarse MPAS mesh)
n_cells = 10000
unstructured_lons, unstructured_lats = fibonacci_sphere(n_cells)

# Ensure longitude is in [0, 360) range
unstructured_lons = unstructured_lons % 360

# Create unstructured grid dataset
unstructured_grid = xr.Dataset(
    {"lat": (["nCells"], unstructured_lats), "lon": (["nCells"], unstructured_lons)}
)

print(f"Created unstructured grid with {n_cells} cells")
print(
    f"Latitude range: {unstructured_lats.min():.1f} to {unstructured_lats.max():.1f}°"
)
print(
    f"Longitude range: {unstructured_lons.min():.1f} to {unstructured_lons.max():.1f}°"
)

# Create target structured grid (1° resolution)
target_lats = np.linspace(-90, 90, 181)
target_lons = np.linspace(0, 359, 360)

structured_grid = xr.Dataset(
    {"lat": (["lat"], target_lats), "lon": (["lon"], target_lons)}
)

# Create realistic atmospheric data on the unstructured grid
# Simulate surface temperature with realistic patterns


def create_atmospheric_field(lats, lons, field_type="temperature"):
    """Create realistic atmospheric field on given coordinates"""

    if field_type == "temperature":
        # Temperature: strong latitudinal gradient + land-ocean contrasts
        base_temp = 15 + 20 * np.cos(np.radians(lats))  # Latitudinal gradient

        # Add zonal variations (simplified land-ocean pattern)
        zonal_var = 10 * np.sin(2 * np.radians(lons)) * np.cos(np.radians(lats))

        # Add topographic effects (simplified mountains)
        topo_effect = -5 * np.exp(
            -((lons - 90) ** 2 + (lats - 30) ** 2) / 500
        )  # Himalayas
        topo_effect += -3 * np.exp(
            -((lons - 280) ** 2 + (lats - 45) ** 2) / 300
        )  # Rockies
        topo_effect += -4 * np.exp(
            -((lons - 350) ** 2 + (lats - 65) ** 2) / 200
        )  # Greenland

        # Random variability
        noise = np.random.normal(0, 2, len(lats))

        return base_temp + zonal_var + topo_effect + noise

    elif field_type == "wind_speed":
        # Wind speed: jet streams and trade winds
        # Subtropical jets
        jet_nh = (
            15
            * np.exp(-(((lats - 35) / 10) ** 2))
            * (1 + 0.3 * np.sin(3 * np.radians(lons)))
        )
        jet_sh = (
            15
            * np.exp(-(((lats + 35) / 10) ** 2))
            * (1 + 0.3 * np.sin(3 * np.radians(lons)))
        )

        # Polar jets
        polar_nh = 25 * np.exp(-(((lats - 60) / 8) ** 2))
        polar_sh = 25 * np.exp(-(((lats + 60) / 8) ** 2))

        # Trade winds
        trades = 8 * np.exp(-(((np.abs(lats) - 15) / 8) ** 2))

        # Combine
        wind = jet_nh + jet_sh + polar_nh + polar_sh + trades

        # Add noise
        wind += np.random.gamma(1, 1, len(lats))  # Non-negative noise

        return wind


# Create temperature and wind data on unstructured grid
temperature_data = create_atmospheric_field(
    unstructured_lats, unstructured_lons, "temperature"
)
wind_data = create_atmospheric_field(unstructured_lats, unstructured_lons, "wind_speed")

# Create xarray DataArrays
temperature = xr.DataArray(
    temperature_data,
    dims=["nCells"],
    coords={"nCells": np.arange(n_cells)},
    attrs={
        "units": "degrees_C",
        "long_name": "Surface Air Temperature",
        "standard_name": "air_temperature",
    },
)

wind_speed = xr.DataArray(
    wind_data,
    dims=["nCells"],
    coords={"nCells": np.arange(n_cells)},
    attrs={
        "units": "m/s",
        "long_name": "10m Wind Speed",
        "standard_name": "wind_speed",
    },
)

print("\nCreated atmospheric data:")
print(
    f"Temperature range: {temperature.min().values:.1f} to {temperature.max().values:.1f} °C"
)
print(
    f"Wind speed range: {wind_speed.min().values:.1f} to {wind_speed.max().values:.1f} m/s"
)

# Create regridders
print("\nCreating regridders...")
temp_regridder = ESMPyRegridder(
    unstructured_grid,
    structured_grid,
    method="bilinear",  # Good for intensive quantities like temperature
)

wind_regridder = ESMPyRegridder(
    unstructured_grid,
    structured_grid,
    method="bilinear",  # Also good for wind speed
)

# Apply regridding
print("Regridding temperature...")
temp_regridded = temp_regridder(temperature)

print("Regridding wind speed...")
wind_regridded = wind_regridder(wind_speed)

print("\nRegridded data shapes:")
print(f"Temperature: {temp_regridded.shape}")
print(f"Wind speed: {wind_regridded.shape}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Unstructured → Structured Grid Regridding", fontsize=16)

# Temperature plots
# Original unstructured data (scatter plot)
sc1 = axes[0, 0].scatter(
    unstructured_lons,
    unstructured_lats,
    c=temperature_data,
    s=1,
    cmap="RdYlBu_r",
    alpha=0.7,
)
axes[0, 0].set_title("Temperature - Unstructured Grid")
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
axes[0, 0].set_xlim(0, 360)
axes[0, 0].set_ylim(-90, 90)
plt.colorbar(sc1, ax=axes[0, 0], label="Temperature (°C)")

# Regridded structured data
im1 = axes[0, 1].pcolormesh(
    target_lons, target_lats, temp_regridded, shading="auto", cmap="RdYlBu_r"
)
axes[0, 1].set_title("Temperature - Regridded to 1° Grid")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[0, 1], label="Temperature (°C)")

# Data density plot (number of unstructured points per structured grid cell)
# Create 2D histogram to show sampling density
density, _, _ = np.histogram2d(
    unstructured_lons, unstructured_lats, bins=[target_lons, target_lats]
)

im2 = axes[0, 2].pcolormesh(
    target_lons, target_lats, density.T, shading="auto", cmap="viridis"
)
axes[0, 2].set_title("Unstructured Point Density")
axes[0, 2].set_xlabel("Longitude")
axes[0, 2].set_ylabel("Latitude")
plt.colorbar(im2, ax=axes[0, 2], label="Points per grid cell")

# Wind speed plots
sc2 = axes[1, 0].scatter(
    unstructured_lons, unstructured_lats, c=wind_data, s=1, cmap="plasma", alpha=0.7
)
axes[1, 0].set_title("Wind Speed - Unstructured Grid")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
axes[1, 0].set_xlim(0, 360)
axes[1, 0].set_ylim(-90, 90)
plt.colorbar(sc2, ax=axes[1, 0], label="Wind Speed (m/s)")

im3 = axes[1, 1].pcolormesh(
    target_lons, target_lats, wind_regridded, shading="auto", cmap="plasma"
)
axes[1, 1].set_title("Wind Speed - Regridded to 1° Grid")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")
plt.colorbar(im3, ax=axes[1, 1], label="Wind Speed (m/s)")

# Statistics comparison
stats_data = {
    "Variable": ["Temperature", "Temperature", "Wind Speed", "Wind Speed"],
    "Grid": ["Unstructured", "Regridded", "Unstructured", "Regridded"],
    "Mean": [
        temperature.mean().values,
        temp_regridded.mean().values,
        wind_speed.mean().values,
        wind_regridded.mean().values,
    ],
    "Std": [
        temperature.std().values,
        temp_regridded.std().values,
        wind_speed.std().values,
        wind_regridded.std().values,
    ],
    "Min": [
        temperature.min().values,
        temp_regridded.min().values,
        wind_speed.min().values,
        wind_regridded.min().values,
    ],
    "Max": [
        temperature.max().values,
        temp_regridded.max().values,
        wind_speed.max().values,
        wind_regridded.max().values,
    ],
}

# Create table
axes[1, 2].axis("tight")
axes[1, 2].axis("off")
table = axes[1, 2].table(
    cellText=[
        [
            f"{stats_data['Variable'][i]}",
            f"{stats_data['Grid'][i]}",
            f"{stats_data['Mean'][i]:.1f}",
            f"{stats_data['Std'][i]:.1f}",
            f"{stats_data['Min'][i]:.1f}",
            f"{stats_data['Max'][i]:.1f}",
        ]
        for i in range(len(stats_data["Variable"]))
    ],
    colLabels=["Variable", "Grid Type", "Mean", "Std", "Min", "Max"],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
axes[1, 2].set_title("Statistics Comparison")

plt.tight_layout()
plt.savefig("unstructured_regridding.png", dpi=150, bbox_inches="tight")
print("\nSaved plot as 'unstructured_regridding.png'")

# Performance analysis
import time  # noqa: E402

print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

# Time the regridding
start_time = time.time()
for _ in range(10):
    _ = temp_regridder(temperature)
regrid_time = (time.time() - start_time) / 10

print("Unstructured → Structured regridding:")
print(f"- Source points: {n_cells:,}")
print(f"- Target points: {len(target_lats) * len(target_lons):,}")
print(f"- Regridding time: {regrid_time:.4f} seconds")
print(f"- Points processed per second: {n_cells / regrid_time:,.0f}")

# Memory usage
memory_mb = temperature.nbytes / 1024 / 1024
output_memory_mb = temp_regridded.nbytes / 1024 / 1024
print("\nMemory usage:")
print(f"- Input data: {memory_mb:.2f} MB")
print(f"- Output data: {output_memory_mb:.1f} MB")
print(f"- Memory expansion: {output_memory_mb / memory_mb:.1f}x")

# Grid statistics
print("\nGrid characteristics:")
print(
    f"- Unstructured grid density: ~{n_cells / (4 * np.pi * (180/np.pi)**2):.1f} points per degree²"
)
print(
    f"- Structured grid density: {len(target_lats) * len(target_lons) / (360 * 180):.1f} points per degree²"
)
print(
    f"- Effective resolution ratio: {(len(target_lats) * len(target_lons)) / n_cells:.1f}"
)

# Advantages of unstructured grids
print("\nAdvantages of unstructured grids:")
print("- Variable resolution capability")
print("- No polar singularities")
print("- Efficient for regional refinement")
print("- Better parallel scalability")
print("")
print("XRegrid advantages for unstructured data:")
print("- Automatic detection of grid type")
print("- Efficient ESMF LocStream handling")
print("- Same API for structured/unstructured grids")
print("- Optimized sparse matrix operations")

# Show plot if running interactively
try:
    if __name__ == "__main__":
        plt.show()
except Exception:
    pass
