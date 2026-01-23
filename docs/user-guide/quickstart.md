# Quick Start Guide

This guide will get you up and running with XRegrid in just a few minutes.

## Basic Regridding

### 1. Import Libraries

```python
import xarray as xr
import numpy as np
from xregrid import ESMPyRegridder
```

### 2. Create or Load Grids

XRegrid works with xarray datasets that contain latitude and longitude coordinates.

```python
# Create a coarse source grid (1° resolution)
source_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 180)),
    'lon': (['lon'], np.linspace(0, 359, 360))
})

# Create a finer target grid (0.5° resolution)
target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 360)),
    'lon': (['lon'], np.linspace(0, 359.5, 720))
})
```

### 3. Initialize the Regridder

```python
# Create regridder with bilinear interpolation
regridder = ESMPyRegridder(
    source_grid,
    target_grid,
    method='bilinear',
    periodic=True  # Important for global grids!
)
```

### 4. Apply to Your Data

```python
# Create some example data
data = xr.DataArray(
    np.random.rand(12, 180, 360),  # time, lat, lon
    dims=['time', 'lat', 'lon'],
    coords={
        'time': pd.date_range('2020-01-01', periods=12, freq='M'),
        'lat': source_grid.lat,
        'lon': source_grid.lon
    },
    name='temperature'
)

# Regrid the data
data_regridded = regridder(data)
print(data_regridded.shape)  # (12, 360, 720)
```

## Grid Types

### Rectilinear Grids

Most common for atmospheric models (CMIP6, ERA5, etc.):

```python
# 1D latitude and longitude arrays
lats = np.linspace(-90, 90, 180)
lons = np.linspace(0, 359, 360)

grid = xr.Dataset({
    'lat': (['lat'], lats),
    'lon': (['lon'], lons)
})
```

### Curvilinear Grids

Common for ocean models (ORCA family):

```python
# 2D coordinate arrays
grid = xr.Dataset({
    'lat': (['y', 'x'], lat_2d_array),
    'lon': (['y', 'x'], lon_2d_array)
})
```

### Unstructured Grids

For models like MPAS or ICON:

```python
# 1D arrays with same dimension name
grid = xr.Dataset({
    'lat': (['nCells'], cell_latitudes),
    'lon': (['nCells'], cell_longitudes)
})
```

## Regridding Methods

### Bilinear (Default)

Best for continuous fields like temperature:

```python
regridder = ESMPyRegridder(source, target, method='bilinear')
```

### Conservative

Best for extensive quantities like precipitation:

```python
regridder = ESMPyRegridder(source, target, method='conservative')
```

### Nearest Neighbor

Best for categorical data:

```python
# Source to destination
regridder = ESMPyRegridder(source, target, method='nearest_s2d')

# Destination to source
regridder = ESMPyRegridder(source, target, method='nearest_d2s')
```

## Performance Tips

### Use Weight Reuse

For repeated regridding with the same grids:

```python
# Save weights on first use
regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)

# Subsequent uses will load existing weights
regridder2 = ESMPyRegridder(
    source, target,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)
```

### Dask Integration

For large datasets, use Dask chunking:

```python
# Load data with chunks
data = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Regridding preserves chunks automatically
data_regridded = regridder(data.temperature)
```

### Global Grids

Always use `periodic=True` for global grids:

```python
regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    periodic=True  # Handles dateline correctly
)
```

## Handling Missing Data

### Skip NaN Values

```python
regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    skipna=True,
    na_thres=0.5  # Require at least 50% valid source points
)
```

### Using Masks

```python
# Add mask to source grid (1=valid, 0=masked)
source_grid['mask'] = (['lat', 'lon'], land_sea_mask)

regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    mask_var='mask'
)
```

## Common Patterns

### Processing Multiple Variables

```python
# Create regridder once
regridder = ESMPyRegridder(source, target, method='bilinear')

# Apply to multiple variables
variables = ['temperature', 'humidity', 'pressure']
results = {}

for var in variables:
    results[var] = regridder(dataset[var])

# Combine into new dataset
regridded_ds = xr.Dataset(results)
```

### Batch Processing

```python
# Process multiple files
files = ['file1.nc', 'file2.nc', 'file3.nc']
regridder = ESMPyRegridder(source, target, method='bilinear')

for file in files:
    ds = xr.open_dataset(file)
    ds_regridded = regridder(ds.temperature)

    output_file = file.replace('.nc', '_regridded.nc')
    ds_regridded.to_netcdf(output_file)
```

## Next Steps

- Explore the [Examples Gallery](../examples/index.md) for more complex use cases
- Learn about [Performance Optimization](performance.md) for large datasets
- Check the [API Reference](../api/regridder.md) for all available options

## Troubleshooting

### Common Errors

**"Grid coordinates not found"**
- Ensure your dataset has 'lat' and 'lon' variables
- Check coordinate names and dimensions

**"Regridding failed"**
- Verify coordinate ranges (latitude: -90 to 90, longitude: 0 to 360)
- Check for invalid coordinates (NaN, inf)
- Try a different regridding method

**Slow performance**
- Use `periodic=True` for global grids
- Enable weight reuse for repeated operations
- Consider chunking large datasets with Dask
