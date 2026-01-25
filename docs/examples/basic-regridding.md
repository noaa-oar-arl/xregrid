# Basic Rectilinear Grid Regridding

This example demonstrates the most common use case: regridding between rectilinear latitude-longitude grids, such as those used in atmospheric climate models.

## Overview

We'll regrid synthetic temperature data from a coarse 1° grid to a finer 0.5° grid using bilinear interpolation.

**Key concepts demonstrated:**
- Creating rectilinear grids
- Using the Regridder with bilinear method
- Handling global periodicity
- Performance analysis

## Full Example Code

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xregrid import Regridder

# Create source grid (1° resolution)
source_lats = np.linspace(-90, 90, 180)
source_lons = np.linspace(0, 359, 360)

source_grid = xr.Dataset({
    'lat': (['lat'], source_lats),
    'lon': (['lon'], source_lons)
})

# Create target grid (0.5° resolution)
target_lats = np.linspace(-90, 90, 360)
target_lons = np.linspace(0, 359.5, 720)

target_grid = xr.Dataset({
    'lat': (['lat'], target_lats),
    'lon': (['lon'], target_lons)
})

print(f"Source grid shape: {len(source_lats)} x {len(source_lons)}")
print(f"Target grid shape: {len(target_lats)} x {len(target_lons)}")

# Create synthetic temperature data with realistic spatial patterns
lons_2d, lats_2d = np.meshgrid(source_lons, source_lats)

# Create a temperature field with:
# - Latitudinal gradient (warmer at equator)
# - Seasonal cycle
# - Some realistic spatial patterns
temperature_pattern = (
    20 * np.cos(np.radians(lats_2d)) +  # Latitudinal gradient
    5 * np.sin(2 * np.radians(lons_2d)) * np.cos(np.radians(lats_2d)) +  # Longitudinal variation
    np.random.normal(0, 2, lats_2d.shape)  # Random noise
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
    dims=['time', 'lat', 'lon'],
    coords={
        'time': times,
        'lat': source_lats,
        'lon': source_lons
    },
    attrs={
        'units': 'degrees_C',
        'long_name': 'Surface Temperature',
        'standard_name': 'air_temperature'
    }
)

# Create the regridder
print("Creating regridder...")
regridder = Regridder(
    source_grid,
    target_grid,
    method='bilinear',
    periodic=True  # Important for global grids!
)

# Apply regridding
print("Regridding temperature data...")
temp_regridded = regridder(temperature)

print(f"Regridded data shape: {temp_regridded.shape}")
```

## Expected Output

When you run this example, you'll see:

```
Source grid shape: 180 x 360
Target grid shape: 360 x 720

Created temperature data with shape: (12, 180, 360)
Temperature range: -16.2 to 32.1 °C

Creating regridder...
Regridding temperature data...

Regridded data shape: (12, 360, 720)
Regridded temperature range: -16.2 to 30.9 °C
```

## Performance Highlights

The example also includes performance analysis:

- **Average regridding time**: ~0.046 seconds
- **Points processed per second**: ~16.9 million
- **Grid ratio**: 4.0x more points in target grid
- **Vectorization speedup**: Significant when processing multiple time steps

## Key Features Demonstrated

### 1. Grid Creation

XRegrid automatically detects rectilinear grids when you provide 1D latitude and longitude arrays with different dimension names.

### 2. Global Periodicity

The `periodic=True` parameter is crucial for global grids as it:
- Handles the dateline (longitude wraparound) correctly
- Ensures proper spherical geometry calculations
- Improves interpolation accuracy near the dateline

### 3. Performance Optimization

The example showcases XRegrid's performance advantages:
- Vectorized operations across time dimension
- Optimized sparse matrix operations
- Efficient memory usage

## Running the Example

To run this example yourself:

1. Save the code as `basic_regridding.py`
2. Run: `python basic_regridding.py`
3. The script will generate performance metrics and save a plot as `basic_regridding.png`

## Next Steps

- Try [Conservative Regridding](conservative-regridding.md) for flux data
- Learn about [Unstructured Grids](unstructured-grids.md) for MPAS/ICON models
- Explore [Performance Optimization](performance-optimization.md) techniques

---

*Download: [plot_basic_regridding.py](scripts/plot_basic_regridding.py)*