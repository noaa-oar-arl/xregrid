# API Reference

## ESMPyRegridder

::: xregrid.ESMPyRegridder

The `ESMPyRegridder` is the main class for performing regridding operations in XRegrid. It provides an optimized interface to ESMF (Earth System Modeling Framework) for interpolating data between different grids.

### Key Features

- **Multiple grid types**: Supports rectilinear, curvilinear, and unstructured grids
- **Various interpolation methods**: Bilinear, conservative, nearest neighbor, and patch recovery
- **Weight reuse**: Save and load regridding weights for repeated operations
- **Dask integration**: Seamless support for chunked arrays and parallel processing
- **Global grid support**: Proper handling of periodic boundary conditions

### Usage Examples

#### Basic Regridding

```python
import xarray as xr
import numpy as np
from xregrid import ESMPyRegridder

# Define source and target grids
source_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 180)),
    'lon': (['lon'], np.linspace(0, 359, 360))
})

target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 360)),
    'lon': (['lon'], np.linspace(0, 359.5, 720))
})

# Create regridder
regridder = ESMPyRegridder(
    source_grid, target_grid, 
    method='bilinear',
    periodic=True
)

# Apply to data
data_regridded = regridder(data)
```

#### Weight Reuse

```python
# Save weights on first use
regridder = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)

# Subsequent uses load existing weights
regridder2 = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)
```

#### Conservative Regridding

```python
# For flux quantities (precipitation, radiation, etc.)
regridder = ESMPyRegridder(
    source_grid, target_grid,
    method='conservative',
    periodic=True
)

precipitation_regridded = regridder(precipitation)
```

### Methods

| Method | Use Case | Accuracy | Speed |
|--------|----------|----------|-------|
| `bilinear` | Continuous fields (temperature, pressure) | High | Fast |
| `conservative` | Flux quantities (precipitation, radiation) | Highest for fluxes | Medium |
| `nearest_s2d` | Categorical data, sparse grids | Variable | Fastest |
| `nearest_d2s` | Categorical data, dense grids | Variable | Fastest |
| `patch` | High-order interpolation | Highest | Slowest |

### Grid Type Support

#### Rectilinear Grids
```python
# 1D lat/lon arrays
grid = xr.Dataset({
    'lat': (['lat'], latitude_1d),
    'lon': (['lon'], longitude_1d)
})
```

#### Curvilinear Grids
```python
# 2D coordinate arrays
grid = xr.Dataset({
    'lat': (['y', 'x'], latitude_2d),
    'lon': (['y', 'x'], longitude_2d)
})
```

#### Unstructured Grids
```python
# 1D arrays with same dimension
grid = xr.Dataset({
    'lat': (['nCells'], cell_latitudes),
    'lon': (['nCells'], cell_longitudes)
})
```

### Performance Considerations

- **Use `periodic=True`** for global grids to ensure proper spherical geometry
- **Enable weight reuse** for repeated regridding operations (10-100x speedup)
- **Process time series together** rather than timestep-by-timestep for vectorization benefits
- **Use conservative method sparingly** - only when flux conservation is critical
- **Chunk large datasets** appropriately for memory management

### Error Handling

The regridder performs several validation checks:

- Coordinate validity (lat: [-90, 90], lon: [0, 360])
- Grid compatibility
- Method availability for grid types
- Memory and dimension constraints

Common errors and solutions:

- **"Invalid coordinates"**: Check coordinate ranges and NaN values
- **"Regridding failed"**: Try different method or check grid validity
- **Memory errors**: Use smaller chunks or enable weight reuse