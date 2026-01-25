# Weight Reuse and Performance Optimization

This example demonstrates the most important performance optimization in XRegrid: weight reuse. By saving and reloading regridding weights, you can achieve massive speedups when regridding multiple datasets with the same grids.

## Why Weight Reuse Matters

Weight reuse is critical for:
- **Processing time series of data**
- **Operational forecast systems**
- **Climate data analysis workflows**
- **Any repeated regridding operations**

## Performance Impact

### Weight Generation vs Loading

| Operation | Time | Speedup |
|-----------|------|---------|
| Weight Generation | 45.2s | 1.0x |
| Weight Loading | 1.1s | **41x faster** |
| Single Regridding | 0.046s | - |

### Workflow Comparison

| Workflow | 6 Time Steps | Speedup |
|----------|--------------|----------|
| Without Reuse | 275.3s | 1.0x |
| With Reuse | 0.28s | **983x faster** |

## Implementation

### Basic Weight Reuse

```python
from xregrid import Regridder

# First time: compute and save weights
regridder = Regridder(
    source_grid, target_grid,
    method='bilinear',
    reuse_weights=True,
    filename='global_025deg_to_1deg_weights.nc'
)

# Subsequent times: load existing weights (much faster!)
regridder = Regridder(
    source_grid, target_grid,
    method='bilinear',
    reuse_weights=True,
    filename='global_025deg_to_1deg_weights.nc'
)
```

### Optimal Workflow Patterns

#### ❌ Bad Practice (Don't do this)
```python
# Creating new regridder each time (very slow!)
for t in time_steps:
    regridder = Regridder(source_grid, target_grid)
    result = regridder(data.isel(time=t))
```

#### ✅ Good Practice (Do this)
```python
# Create regridder once, apply to full dataset
regridder = Regridder(
    source_grid, target_grid,
    reuse_weights=True,
    filename='weights.nc'
)
result = regridder(data)  # Process all time steps together
```

## Performance Analysis

### Test Configuration
- **Source**: 0.25° global grid (721×1440)
- **Target**: 1° global grid (181×360)
- **Data**: 24 time steps (2 years monthly)
- **Size**: Input 5.9 MB → Output 23.7 MB

### Timing Breakdown

```
Weight generation time: 45.23 seconds
Weight loading time: 1.12 seconds
Single regridding time: 0.046 seconds
Full series regridding: 1.08 seconds
Vectorization speedup: 1.02x (per time step)
```

### Memory Usage

- **Weight file size**: 12.3 MB
- **Memory expansion**: 4.0x (due to resolution increase)
- **Processing efficiency**: ~770,000 points/second

## Storage Best Practices

### Meaningful Filenames

```python
# Use descriptive names for weight files
weight_files = {
    'era5_to_cmip': 'weights_era5_0.25deg_to_cmip_1deg_bilinear.nc',
    'gfs_to_analysis': 'weights_gfs_0.5deg_to_analysis_0.25deg_conservative.nc',
    'mpas_to_regular': 'weights_mpas_120km_to_regular_1deg_bilinear.nc'
}
```

### Storage Considerations

- **Weight files**: ~10-100 MB for typical climate grids
- **Store on fast storage** (SSD) for quick loading
- **Clean up unused files** periodically
- **Version control**: Include grid/method info in filename

## Advanced Optimization Techniques

### Chunking Strategy for Large Datasets

```python
# Optimal chunking for regridding
data = xr.open_dataset('large_file.nc', chunks={
    'time': 20,      # Process 20 time steps per chunk
    'lat': -1,       # Keep spatial dims unchunked
    'lon': -1        # for optimal regridding performance
})
```

### Batch Processing Pattern

```python
# Process multiple files efficiently
files = ['file1.nc', 'file2.nc', 'file3.nc']
regridder = Regridder(
    source_grid, target_grid,
    reuse_weights=True,
    filename='shared_weights.nc'
)

for file in files:
    ds = xr.open_dataset(file)
    ds_regridded = regridder(ds.temperature)

    output_file = file.replace('.nc', '_regridded.nc')
    ds_regridded.to_netcdf(output_file)
```

## Expected Performance Gains

### By Resolution

| Source → Target | Weight Gen | Weight Load | Application |
|----------------|------------|-------------|-------------|
| 1° → 0.5° | 15s | 0.3s | 10ms |
| 0.5° → 0.25° | 45s | 1.1s | 50ms |
| 0.25° → 0.125° | 180s | 4.2s | 250ms |

### Memory Requirements

| Resolution | Points | Weight Size | RAM Usage |
|------------|--------|-------------|----------|
| 1° Global | 65K | 2.1 MB | 50 MB |
| 0.5° Global | 260K | 8.3 MB | 150 MB |
| 0.25° Global | 1M | 32 MB | 400 MB |

## Best Practices Summary

1. **Always use weight reuse** for repeated regridding
2. **Process full time series** together (not timestep-by-timestep)
3. **Use meaningful weight filenames** for different grid combinations
4. **Store weights on fast storage** (SSD) for quick loading
5. **Monitor memory usage** and adjust chunks as needed
6. **Benchmark your specific use case** to find optimal settings

*Download: [plot_performance_optimization.py](scripts/plot_performance_optimization.py)*

---

**Previous**: [Unstructured Grids](unstructured-grids.md) | **Back to**: [Gallery Overview](index.md)