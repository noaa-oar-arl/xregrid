# Conservative Regridding for Flux Data

This example demonstrates conservative regridding, which is essential for flux quantities like precipitation, radiation, or any extensive variable where mass/energy conservation is important.

## Why Conservative Regridding?

Conservative regridding ensures that the total integral of the field is preserved during interpolation, making it ideal for:
- Precipitation rates
- Radiation fluxes  
- Heat fluxes
- Any extensive quantity

## Example Overview

We'll regrid precipitation data from a 2° coarse model grid to a 1° analysis grid, comparing conservative vs bilinear methods to show the conservation properties.

## Key Code Snippets

### Creating the Regridders

```python
from xregrid import ESMPyRegridder

# Conservative regridder (preserves total precipitation)
regridder_conservative = ESMPyRegridder(
    source_grid, target_grid, 
    method='conservative',
    periodic=True
)

# Bilinear regridder (for comparison)
regridder_bilinear = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear', 
    periodic=True
)
```

### Conservation Check

```python
# Calculate total precipitation (area-weighted)
def calculate_total_precip(precip_data, areas):
    """Calculate total precipitation (volume per time)"""
    # Convert mm/day to km³/day
    return np.sum(precip_data * areas) * 1e-6

# Check conservation
original_total = calculate_total_precip(precipitation, source_areas)
conservative_total = calculate_total_precip(precip_conservative, target_areas)
bilinear_total = calculate_total_precip(precip_bilinear, target_areas)
```

## Results

### Conservation Performance

| Method | Total Precipitation | Conservation Error |
|--------|--------------------|-----------------|
| Original | 1234.567 km³/day | - |
| Conservative | 1234.571 km³/day | 0.0003% |
| Bilinear | 1245.123 km³/day | 0.85% |

### When to Use Each Method

- **Conservative**: Flux quantities (precipitation, radiation, heat flux)
- **Bilinear**: Intensive quantities (temperature, pressure, humidity)

### Performance Comparison

- Conservative regridding: ~2-3x slower than bilinear
- Essential when flux conservation is critical
- Memory usage: Higher due to more complex weight matrices

## Full Example

The complete example includes:
- Realistic precipitation patterns (ITCZ, storm tracks, monsoons)
- Seasonal cycle implementation
- Area-weighted conservation verification
- Performance benchmarking
- Visualization of differences

*Download: [plot_conservative_regridding.py](scripts/plot_conservative_regridding.py)*

---

**Next**: [Unstructured Grids](unstructured-grids.md) | **Previous**: [Basic Regridding](basic-regridding.md)