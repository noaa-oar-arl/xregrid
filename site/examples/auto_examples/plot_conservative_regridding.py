"""
Conservative Regridding for Flux Data
====================================

This example demonstrates conservative regridding, which is essential for
flux quantities like precipitation, radiation, or any extensive variable
where mass/energy conservation is important.

Conservative regridding ensures that the total integral of the field is
preserved during interpolation, making it ideal for:
- Precipitation rates
- Radiation fluxes
- Heat fluxes
- Any extensive quantity

Key concepts demonstrated:
- Conservative vs bilinear regridding comparison
- Flux conservation verification
- Handling precipitation-like data
- Area-weighted integration
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for gallery
import matplotlib.pyplot as plt
from xregrid import ESMPyRegridder

# Create source and target grids with different resolutions
# Source: 2° resolution (coarse precipitation model)
source_lats = np.linspace(-90, 90, 91)  # 2° spacing
source_lons = np.linspace(0, 358, 180)  # 2° spacing

source_grid = xr.Dataset({
    'lat': (['lat'], source_lats),
    'lon': (['lon'], source_lons)
})

# Target: 1° resolution (analysis grid)
target_lats = np.linspace(-90, 90, 181)  # 1° spacing
target_lons = np.linspace(0, 359, 360)   # 1° spacing

target_grid = xr.Dataset({
    'lat': (['lat'], target_lats),
    'lon': (['lon'], target_lons)
})

print(f"Source grid: {len(source_lats)}x{len(source_lons)} (2° resolution)")
print(f"Target grid: {len(target_lats)}x{len(target_lons)} (1° resolution)")

# Create realistic precipitation patterns
lons_2d, lats_2d = np.meshgrid(source_lons, source_lats)

# Simulate precipitation with:
# - ITCZ (Intertropical Convergence Zone)
# - Storm tracks in mid-latitudes
# - Dry regions in subtropics
# - Monsoon patterns

# Base precipitation pattern
precip_base = np.zeros_like(lats_2d)

# ITCZ - high precipitation near equator
itcz = 8 * np.exp(-((lats_2d - 5) / 8)**2)  # Shifted slightly north

# Storm tracks - mid-latitude precipitation
storm_tracks_nh = 4 * np.exp(-((lats_2d - 50) / 12)**2)  # Northern hemisphere
storm_tracks_sh = 4 * np.exp(-((lats_2d + 50) / 12)**2)  # Southern hemisphere

# Monsoon patterns - enhanced precipitation in certain longitude bands
monsoon_asia = 6 * np.exp(-((lons_2d - 120) / 30)**2) * np.exp(-((lats_2d - 25) / 15)**2)
monsoon_africa = 4 * np.exp(-((lons_2d - 15) / 25)**2) * np.exp(-((lats_2d - 10) / 12)**2)

# Combine patterns
precip_base = itcz + storm_tracks_nh + storm_tracks_sh + monsoon_asia + monsoon_africa

# Add some randomness for realistic variability
np.random.seed(42)  # For reproducible results
precip_noise = np.random.gamma(2, 0.5, lats_2d.shape)  # Gamma distribution for precipitation

# Create seasonal cycle
times = np.arange(12)
precip_data = np.zeros((12, len(source_lats), len(source_lons)))

for i, month in enumerate(times):
    # Seasonal migration of ITCZ and monsoons
    seasonal_shift = 10 * np.sin(2 * np.pi * (month - 3) / 12)  # Peak in June (month 5)
    shifted_lat = lats_2d + seasonal_shift

    # Recalculate patterns with seasonal shift
    seasonal_itcz = 8 * np.exp(-((shifted_lat - 5) / 8)**2)
    seasonal_monsoon = 6 * np.exp(-((lons_2d - 120) / 30)**2) * np.exp(-((shifted_lat - 25) / 15)**2)

    precip_data[i] = np.maximum(0, precip_base + seasonal_itcz + seasonal_monsoon + precip_noise)

# Create xarray DataArray
precipitation = xr.DataArray(
    precip_data,
    dims=['time', 'lat', 'lon'],
    coords={
        'time': times,
        'lat': source_lats,
        'lon': source_lons
    },
    attrs={
        'units': 'mm/day',
        'long_name': 'Precipitation Rate',
        'standard_name': 'precipitation_flux'
    }
)

print(f"\nCreated precipitation data with shape: {precipitation.shape}")
print(f"Precipitation range: {precipitation.min().values:.1f} to {precipitation.max().values:.1f} mm/day")

# Create both conservative and bilinear regridders for comparison
print("\nCreating regridders...")
regridder_conservative = ESMPyRegridder(
    source_grid, target_grid,
    method='conservative',
    periodic=True
)

regridder_bilinear = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear',
    periodic=True
)

# Apply both regridding methods
print("Applying conservative regridding...")
precip_conservative = regridder_conservative(precipitation)

print("Applying bilinear regridding...")
precip_bilinear = regridder_bilinear(precipitation)

# Calculate grid cell areas for integration
def calculate_grid_areas(lats, lons):
    """Calculate grid cell areas in km²"""
    R_earth = 6371.0  # Earth radius in km

    # Convert to radians
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    # Calculate lat/lon differences
    dlat = np.diff(lats_rad)
    dlon = np.diff(lons_rad)

    # Handle periodic boundary for longitude
    if len(dlon) == len(lons) - 1:
        dlon = np.append(dlon, dlon[0])  # Assume periodic

    # Create 2D arrays
    dlat_2d = np.repeat(dlat[:, np.newaxis], len(lons), axis=1)
    dlon_2d = np.repeat(dlon[np.newaxis, :], len(lats), axis=0)
    lats_2d = np.repeat(lats_rad[:, np.newaxis], len(lons), axis=1)

    # Calculate areas: dA = R² * cos(lat) * dlat * dlon
    areas = R_earth**2 * np.cos(lats_2d) * dlat_2d * dlon_2d

    return areas

# Calculate areas for both grids
source_areas = calculate_grid_areas(source_lats, source_lons)
target_areas = calculate_grid_areas(target_lats, target_lons)

# Calculate total precipitation (area-weighted) for conservation check
def calculate_total_precip(precip_data, areas):
    """Calculate total precipitation (volume per time)"""
    # Convert mm/day to km³/day: mm/day * km² * (1 km / 1e6 mm) = km³/day * 1e-6
    return np.sum(precip_data * areas) * 1e-6

# Check conservation for July (month 6)
july_idx = 6
original_total = calculate_total_precip(
    precipitation.isel(time=july_idx).values, source_areas
)
conservative_total = calculate_total_precip(
    precip_conservative.isel(time=july_idx).values, target_areas
)
bilinear_total = calculate_total_precip(
    precip_bilinear.isel(time=july_idx).values, target_areas
)

print(f"\n" + "="*60)
print("CONSERVATION CHECK (July)")
print("="*60)
print(f"Original total precipitation:   {original_total:.3f} km³/day")
print(f"Conservative regridding:        {conservative_total:.3f} km³/day")
print(f"Bilinear regridding:           {bilinear_total:.3f} km³/day")
print(f"")
print(f"Conservative error: {abs(conservative_total - original_total)/original_total*100:.4f}%")
print(f"Bilinear error:     {abs(bilinear_total - original_total)/original_total*100:.2f}%")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Conservative vs Bilinear Regridding for Precipitation', fontsize=16)

# Select July for visualization
july_data = {
    'original': precipitation.isel(time=july_idx),
    'conservative': precip_conservative.isel(time=july_idx),
    'bilinear': precip_bilinear.isel(time=july_idx)
}

grids = {
    'original': (source_lons, source_lats),
    'conservative': (target_lons, target_lats),
    'bilinear': (target_lons, target_lats)
}

# Plot precipitation fields
vmax = max([data.max().values for data in july_data.values()])

for i, (method, data) in enumerate(july_data.items()):
    lons, lats = grids[method]
    im = axes[0, i].pcolormesh(
        lons, lats, data,
        shading='auto', cmap='Blues', vmin=0, vmax=vmax
    )
    axes[0, i].set_title(f'{method.title()} (July)')
    axes[0, i].set_xlabel('Longitude')
    axes[0, i].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0, i], label='Precipitation (mm/day)')

# Plot differences
diff_conservative = precip_conservative.isel(time=july_idx) - precip_bilinear.isel(time=july_idx)

im_diff = axes[1, 0].pcolormesh(
    target_lons, target_lats, diff_conservative,
    shading='auto', cmap='RdBu_r',
    vmin=-2, vmax=2
)
axes[1, 0].set_title('Conservative - Bilinear')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Latitude')
plt.colorbar(im_diff, ax=axes[1, 0], label='Difference (mm/day)')

# Conservation time series
original_totals = []
conservative_totals = []
bilinear_totals = []

for month in range(12):
    orig = calculate_total_precip(precipitation.isel(time=month).values, source_areas)
    cons = calculate_total_precip(precip_conservative.isel(time=month).values, target_areas)
    bili = calculate_total_precip(precip_bilinear.isel(time=month).values, target_areas)

    original_totals.append(orig)
    conservative_totals.append(cons)
    bilinear_totals.append(bili)

axes[1, 1].plot(times, original_totals, 'k-', label='Original', linewidth=3)
axes[1, 1].plot(times, conservative_totals, 'b--', label='Conservative', linewidth=2)
axes[1, 1].plot(times, bilinear_totals, 'r:', label='Bilinear', linewidth=2)
axes[1, 1].set_title('Total Precipitation Conservation')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Total Precipitation (km³/day)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Relative error time series
cons_error = [(c - o)/o * 100 for o, c in zip(original_totals, conservative_totals)]
bili_error = [(b - o)/o * 100 for o, b in zip(original_totals, bilinear_totals)]

axes[1, 2].plot(times, cons_error, 'b-', label='Conservative', linewidth=2)
axes[1, 2].plot(times, bili_error, 'r-', label='Bilinear', linewidth=2)
axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1, 2].set_title('Conservation Error')
axes[1, 2].set_xlabel('Month')
axes[1, 2].set_ylabel('Relative Error (%)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('conservative_regridding.png', dpi=150, bbox_inches='tight')
print("\nSaved plot as 'conservative_regridding.png'")

# Performance comparison
import time

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

# Time conservative regridding
start_time = time.time()
for _ in range(5):
    _ = regridder_conservative(precipitation.isel(time=0))
cons_time = (time.time() - start_time) / 5

# Time bilinear regridding
start_time = time.time()
for _ in range(5):
    _ = regridder_bilinear(precipitation.isel(time=0))
bili_time = (time.time() - start_time) / 5

print(f"Conservative regridding: {cons_time:.4f} seconds")
print(f"Bilinear regridding:     {bili_time:.4f} seconds")
print(f"Conservative is {cons_time/bili_time:.1f}x slower than bilinear")

print(f"\nWhen to use each method:")
print(f"- Conservative: Flux quantities (precipitation, radiation, heat flux)")
print(f"- Bilinear:     Intensive quantities (temperature, pressure, humidity)")
print(f"")
print(f"Key advantage of conservative regridding:")
print(f"- Preserves total integral (essential for flux conservation)")
print(f"- Error: {abs(np.mean(cons_error)):.4f}% vs {abs(np.mean(bili_error)):.2f}%")

# Show plot if running interactively
try:
    if __name__ == "__main__":
        plt.show()
except:
    pass
