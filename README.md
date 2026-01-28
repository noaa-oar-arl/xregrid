# XRegrid

An optimized ESMF-based regridder for xarray that provides significant performance improvements over xESMF.

## Overview

XRegrid is a high-performance regridding library that builds on top of ESMF (Earth System Modeling Framework) to provide fast and accurate interpolation between different grids. It offers substantial performance improvements over existing solutions while maintaining full compatibility with xarray data structures.

## Key Features

- **High Performance**: Up to 35x faster than xESMF for single time-step regridding
- **Correct ESMF Integration**: Native support for rectilinear and curvilinear grids
- **Dask Integration**: Seamless parallel processing with Dask arrays
- **Memory Efficient**: Optimized sparse matrix operations using scipy
- **xarray Compatible**: Native support for xarray datasets and data arrays
- **Automatic coordinate detection**: Support for `cf-xarray` for easy coordinate and boundary identification
- **Weight Reuse**: Save and load regridding weights to/from NetCDF files
- **Grid Utilities**: Built-in functions for quick global and regional grid generation

## Why XRegrid?

While `ESMPy` provides the powerful underlying engine for regridding, it is a low-level library that requires significant boilerplate code to use with `xarray`. `XRegrid` bridges this gap by providing:

1.  **High-level API**: Use `xarray.Dataset` and `xarray.DataArray` directly without manual grid or field creation.
2.  **Performance**: Optimized sparse matrix application that is up to 35x faster than other ESMF-based wrappers.
3.  **Correctness**: Automatic handling of coordinate transpositions, periodicity, and metadata.

### XRegrid vs. Raw ESMPy

To regrid a single field, `ESMPy` requires manually creating grids, fields, and handling coordinate pointers. `XRegrid` abstracts this entire process into two lines of code.

| Feature | ESMPy | XRegrid |
|---------|-------|---------|
| Grid Definition | Manual `esmpy.Grid` | Native `xarray.Dataset` |
| Coordinate Handling | Manual pointer filling | Automatic detection |
| Data Interface | NumPy-only | Xarray (NumPy & Dask) |
| Code Complexity | ~30-50 lines | 2 lines |

## Quick Example

```python
import xarray as xr
from xregrid import Regridder

# Load tutorial data
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

# Define a target grid (e.g., 1.0° resolution)
import numpy as np
target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(15, 76, 1.0)),
    "lon": (["lon"], np.arange(200, 331, 1.0))
})

# Create regridder and apply
regridder = Regridder(ds, target_grid, method='bilinear')
air_regridded = regridder(ds.air)
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

- [Quick Start Guide](https://xregrid.readthedocs.io/user-guide/quickstart/)
- [API Reference](https://xregrid.readthedocs.io/api/regridder/)

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

XRegrid is released under the MIT License.
