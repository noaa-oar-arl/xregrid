# XRegrid

An optimized ESMF-based regridder for xarray that provides significant performance improvements over xESMF.

## Overview

XRegrid is a high-performance regridding library that builds on top of ESMF (Earth System Modeling Framework) to provide fast and accurate interpolation between different grids. It offers substantial performance improvements over existing solutions while maintaining full compatibility with xarray data structures.

## Key Features

- **High Performance**: Up to 30x faster than xESMF for single time-step regridding
- **Correct ESMF Integration**: Native support for rectilinear and curvilinear grids
- **Dask Integration**: Seamless parallel processing with Dask arrays
- **Memory Efficient**: Optimized sparse matrix operations using scipy
- **xarray Compatible**: Native support for xarray datasets and data arrays
- **Automatic coordinate detection**: Support for `cf-xarray` for easy coordinate and boundary identification
- **Weight Reuse**: Save and load regridding weights to/from NetCDF files
- **Grid Utilities**: Built-in functions for quick global and regional grid generation

## Quick Example

```python
import xarray as xr
import numpy as np
from xregrid import Regridder, create_global_grid

# Create source and target grids
source_grid = create_global_grid(res_lat=1.0, res_lon=1.0)
target_grid = create_global_grid(res_lat=0.5, res_lon=0.5)

# Create regridder
regridder = Regridder(
    source_grid, target_grid,
    method='bilinear',
    periodic=True
)

# Apply to your data
temperature = xr.DataArray(
    np.random.rand(10, 180, 360),
    dims=['time', 'lat', 'lon'],
    coords={'lat': source_grid.lat, 'lon': source_grid.lon}
)

temperature_regridded = regridder(temperature)
```

## Installation

Install via mamba (recommended):

```bash
mamba env create -f environment.yml
mamba activate xregrid
```

Or install from source:

```bash
pip install .
```

## Documentation

Full documentation is available at [https://xregrid.readthedocs.io](https://xregrid.readthedocs.io)

- [Quick Start Guide](docs/user-guide/quickstart.md)
- [API Reference](docs/api/regridder.md)

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

XRegrid is released under the MIT License.
