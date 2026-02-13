# Quick Start Guide

This guide will get you up and running with XRegrid in just a few minutes.

## Basic Regridding

### 1. Import Libraries

```python
import xarray as xr
import numpy as np
from xregrid import Regridder
```

### 2. Create or Load Grids

XRegrid works with xarray datasets that contain latitude and longitude coordinates. You can also use the built-in utilities to quickly create standard grids.

```python
from xregrid import create_global_grid, create_regional_grid

# Create a 1° global source grid
source_grid = create_global_grid(res_lat=1.0, res_lon=1.0)

# Create a 0.5° global target grid
target_grid = create_global_grid(res_lat=0.5, res_lon=0.5)

# Or create a regional grid
regional_grid = create_regional_grid(
    lat_range=(35, 70),
    lon_range=(-10, 40),
    res_lat=0.25,
    res_lon=0.25
)
```

### 3. Initialize the Regridder

```python
# Create regridder with bilinear interpolation
regridder = Regridder(
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
regridder = Regridder(source, target, method='bilinear')
```

### Conservative

Best for extensive quantities like precipitation:

```python
regridder = Regridder(source, target, method='conservative')
```

### Nearest Neighbor

Best for categorical data:

```python
# Source to destination
regridder = Regridder(source, target, method='nearest_s2d')

# Destination to source
regridder = Regridder(source, target, method='nearest_d2s')
```

## Performance Tips

### Use Weight Reuse

For repeated regridding with the same grids:

```python
# Save weights on first use
regridder = Regridder(
    source, target,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)

# Subsequent uses will load existing weights
regridder2 = Regridder(
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
regridder = Regridder(
    source, target,
    method='bilinear',
    periodic=True  # Handles dateline correctly
)
```

## CRS and Metadata

### Automatic Propagation

XRegrid automatically propagates Coordinate Reference System (CRS) information from the target grid to the regridded output. This ensures that your data remains "geospatially aware" and ready for plotting.

- **`crs` attribute**: If the target grid has a `crs` attribute (e.g., in WKT format), it is copied to the output.
- **`grid_mapping`**: If the target grid uses a CF-compliant grid mapping variable, the output will correctly point to it.

### Projection Discovery

The `get_crs_info` utility can be used to robustly identify the CRS of any XRegrid-regridded object:

```python
from xregrid.utils import get_crs_info

crs = get_crs_info(data_regridded)
if crs:
    print(f"Detected CRS: {crs.to_epsg()}")
```

## Handling Missing Data

### Skip NaN Values

```python
regridder = Regridder(
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

regridder = Regridder(
    source, target,
    method='bilinear',
    mask_var='mask'
)
```

## Common Patterns

### Processing Multiple Variables

```python
# Create regridder once
regridder = Regridder(source, target, method='bilinear')

# Apply directly to the whole dataset
# This will regrid all data variables containing 'lat' and 'lon'
regridded_ds = regridder(dataset)
```

### Batch Processing

```python
# Process multiple files
files = ['file1.nc', 'file2.nc', 'file3.nc']
regridder = Regridder(source, target, method='bilinear')

for file in files:
    ds = xr.open_dataset(file)
    ds_regridded = regridder(ds.temperature)

    output_file = file.replace('.nc', '_regridded.nc')
    ds_regridded.to_netcdf(output_file)
```

## Next Steps

- Explore the [Examples Gallery](../examples/generated/index.md) for more complex use cases
- Learn about [Performance Optimization](performance.md) for large datasets
- Check the [API Reference](../api/regridder.md) for all available options

## Troubleshooting

### Common Errors

**"Grid coordinates not found"**
- Ensure your dataset has 'lat' and 'lon' variables, or follow CF-conventions (XRegrid will automatically detect them via `cf-xarray`)
- Check coordinate names and dimensions

**"Regridding failed"**
- Verify coordinate ranges (latitude: -90 to 90, longitude: 0 to 360)
- Check for invalid coordinates (NaN, inf)
- Try a different regridding method

**Slow performance or Memory errors**
- Use `periodic=True` for global grids
- Enable weight reuse for repeated operations (`reuse_weights=True`)
- Consider chunking large datasets with Dask to manage memory
- Process time series together for better vectorization
