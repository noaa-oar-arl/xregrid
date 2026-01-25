# Installation Guide

There are several ways to install XRegrid and its dependencies. We recommend using `mamba` or `micromamba` for the fastest and easiest setup.

## Quick Install (Recommended)

The easiest way to install all dependencies, including `esmpy` and `xesmf`, is using `mamba` with the `conda-forge` channel.

### 1. Using the Environment File

```bash
# Clone the repository
git clone https://github.com/xregrid/xregrid.git
cd xregrid

# Create the environment from the provided yaml file
mamba env create -f environment.yml

# Activate the environment
mamba activate xregrid
```

### 2. Manual Mamba Installation

If you prefer to set up the environment manually:

```bash
# Create a new environment
mamba create -n xregrid python>=3.9

# Activate the environment
mamba activate xregrid

# Install dependencies
mamba install -c conda-forge esmpy cf-xarray xesmf xarray numpy scipy dask netcdf4 pandas pytest

# Install xregrid from source
pip install -e .
```

## Development Installation

For development, install in editable mode with additional tools:

```bash
# Install in development mode
pip install -e ".[test]"

# Install documentation tools (optional)
pip install mkdocs mkdocs-material mkdocs-gallery
```

## Alternative: Installing ESMPy from Source

If you need to build `esmpy` from source (e.g., for a specific ESMF version or custom build), follow these steps:

### Prerequisites

1. **ESMF Library**: You must have the ESMF C++ library built and installed on your system.
2. **Environment Variables**: Set `ESMF_DIR` to the path where ESMF is installed.

```bash
export ESMF_DIR=/path/to/esmf
```

Depending on your build, you might also need:

```bash
export ESMF_COMPILER=gfortran
export ESMF_COMM=openmpi
```

### Build and Install ESMPy

`esmpy` is located in the ESMF source tree under `src/addon/esmpy`.

```bash
# Navigate to the esmpy directory in the ESMF source
cd $ESMF_DIR/src/addon/esmpy

# Install from source
pip install .
```

### Install XRegrid

Once `esmpy` is installed, you can install this package:

```bash
# From the root of this repository
pip install .
```

## Verification

To verify your installation, run the following test:

```python
import xarray as xr
import numpy as np
from xregrid import Regridder

# Create simple test grids
source = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 10)),
    'lon': (['lon'], np.linspace(0, 350, 10))
})

target = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 20)),
    'lon': (['lon'], np.linspace(0, 355, 20))
})

# Test regridder creation
regridder = Regridder(source, target, method='bilinear')
print("✓ XRegrid installation successful!")
```

## Dependencies

XRegrid requires the following packages:

### Core Dependencies

- **Python** ≥ 3.9
- **xarray** - For data structure support
- **cf-xarray** - Automatic CF-compliant coordinate detection
- **numpy** - Numerical computations
- **scipy** - Sparse matrix operations
- **esmpy** - Earth System Modeling Framework Python interface
- **dask** - Parallel computing
- **netCDF4** - File I/O

### Optional Dependencies

- **xesmf** - For comparison and testing
- **pytest** - For running tests
- **matplotlib** - For plotting examples

## Troubleshooting

### Common Issues

**Import Error: No module named 'esmpy'**

This usually means ESMF/ESMPy is not properly installed. Try:

1. Installing via mamba: `mamba install -c conda-forge esmpy`
2. Checking that `ESMF_DIR` is set correctly if building from source
3. Verifying that the ESMF library is compatible with your system

**Performance Issues**

If you experience slow performance:

1. Ensure you're using the latest scipy version
2. Check that your grid dimensions are properly ordered
3. Consider using `periodic=True` for global grids
4. Use weight reuse for repeated regridding operations

**Memory Issues**

For large grids:

1. Use Dask arrays for chunked processing
2. Enable weight reuse to avoid recomputation
3. Consider using conservative regridding for memory efficiency

### Getting Help

- Check the [Examples Gallery](examples/index.md) for common use cases
- Review the [API documentation](api/regridder.md) for detailed parameter descriptions
- Submit issues on GitHub for bugs or feature requests

## System Requirements

### Minimum Requirements

- 4 GB RAM
- Python 3.9+
- 64-bit operating system

### Recommended

- 16 GB+ RAM for high-resolution grids
- SSD storage for better I/O performance
- Multiple CPU cores for Dask parallelization

### Supported Platforms

- Linux (tested on Ubuntu, CentOS, RHEL)
- macOS (Intel and Apple Silicon)
- Windows (via WSL or native with conda)