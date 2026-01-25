# Unstructured Grid Regridding (MPAS/ICON)

This example demonstrates regridding from unstructured grids, which are used by next-generation climate models like MPAS (Model for Prediction Across Scales) and ICON (ICOsahedral Nonhydrostatic).

## Why Unstructured Grids?

Unstructured grids offer several advantages:
- **Variable resolution**: Refined in regions of interest
- **No polar singularities**: Avoids numerical issues at poles
- **Efficient parallel computation**: Better load balancing
- **Quasi-uniform grid spacing**: More even resolution distribution

## Grid Types Supported

### Creating Unstructured Grid Data

```python
# Unstructured grids have 1D spatial dimensions
# with the same dimension name for lat/lon
unstructured_grid = xr.Dataset({
    'lat': (['nCells'], cell_latitudes),
    'lon': (['nCells'], cell_longitudes)
})

# XRegrid automatically detects this as unstructured
regridder = Regridder(
    unstructured_grid, structured_grid,
    method='bilinear'
)
```

### Synthetic Icosahedral Grid

The example creates a synthetic unstructured grid using a Fibonacci sphere pattern:

```python
def fibonacci_sphere(n_points):
    """Generate approximately uniform points on a sphere"""
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle

    i = np.arange(0, n_points, dtype=float) + 0.5
    lat = np.arcsin(1 - 2 * i / n_points)
    lon = (i * golden_angle) % (2 * np.pi)

    return np.degrees(lon), np.degrees(lat)

# Create ~10,000 point unstructured grid
lons, lats = fibonacci_sphere(10000)
```

## Atmospheric Data Generation

The example creates realistic atmospheric fields:

### Temperature Field
```python
# Temperature with latitudinal gradient + land effects
base_temp = 15 + 20 * np.cos(np.radians(lats))
topo_effect = -5 * np.exp(-((lons - 90)**2 + (lats - 30)**2) / 500)  # Himalayas
temperature = base_temp + topo_effect + noise
```

### Wind Speed Field
```python
# Wind with jet streams and trade winds
jet_nh = 15 * np.exp(-((lats - 35) / 10)**2)
trades = 8 * np.exp(-((np.abs(lats) - 15) / 8)**2)
wind_speed = jet_nh + trades + polar_jets
```

## Performance Analysis

### Regridding Performance
- **Source points**: 10,000 unstructured cells
- **Target points**: 64,800 structured points (1° grid)
- **Regridding time**: ~0.05 seconds
- **Memory expansion**: ~6.4x (unstructured → structured)

### Grid Characteristics
- **Unstructured density**: ~2.8 points per degree²
- **Structured density**: 1.0 points per degree²
- **Effective resolution ratio**: 6.5x more target points

## Visualization Techniques

The example demonstrates how to visualize unstructured data:

### Scatter Plots for Unstructured Data
```python
# Original unstructured data
plt.scatter(unstructured_lons, unstructured_lats,
           c=temperature_data, s=1, cmap='RdYlBu_r')
```

### Regular Grids for Structured Data
```python
# Regridded structured data
plt.pcolormesh(target_lons, target_lats,
              temp_regridded, shading='auto')
```

## Real-World Applications

### MPAS Integration
For actual MPAS data:
```python
# Load MPAS file
ds = xr.open_dataset('mpas_output.nc')

# MPAS uses 'nCells' dimension
mpas_grid = xr.Dataset({
    'lat': ds.latCell,  # Already in degrees
    'lon': ds.lonCell   # Convert from radians if needed
})

regridder = Regridder(mpas_grid, target_grid)
```

### ICON Integration
```python
# ICON typically uses different variable names
icon_grid = xr.Dataset({
    'lat': ds.clat * 180/np.pi,  # Convert radians to degrees
    'lon': ds.clon * 180/np.pi   # Convert radians to degrees
})
```

## Advantages of XRegrid for Unstructured Data

1. **Automatic Detection**: No need to specify grid type
2. **Efficient ESMF LocStream**: Optimized for 1D spatial data
3. **Same API**: Identical interface for all grid types
4. **Optimized Performance**: Sparse matrix operations

*Download: [plot_unstructured_grids.py](scripts/plot_unstructured_grids.py)*

---

**Next**: [Performance Optimization](performance-optimization.md) | **Previous**: [Conservative Regridding](conservative-regridding.md)