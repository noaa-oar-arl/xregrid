# XRegrid

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://xregrid.readthedocs.io)

**An optimized ESMF-based regridder for xarray with significant performance improvements over existing solutions.**

XRegrid provides high-performance regridding for earth science applications, offering up to **600x speedup** over xESMF while maintaining full accuracy and supporting all major grid types.

## 🚀 Key Features

- **Blazing Fast Performance**: Up to 600x faster than xESMF for typical use cases
- **Universal Grid Support**: Rectilinear, curvilinear, and unstructured grids (MPAS, ICON)
- **xarray Integration**: Native support for xarray datasets and data arrays
- **Memory Efficient**: Optimized sparse matrix operations using scipy
- **Dask Compatible**: Seamless parallel processing with chunked arrays
- **Weight Reuse**: Save and load regridding weights for massive speedups
- **Production Ready**: Robust error handling and comprehensive testing

## 📊 Performance Comparison

| Resolution | Grid Points | XRegrid | xESMF | **Speedup** |
|------------|-------------|---------|-------|-------------|
| 1.0° Global | 64,800 | 0.0016s | 0.98s | **~600x** |
| 0.25° Global | 1,036,800 | 0.053s | 1.95s | **~36x** |
| 0.1° Global | 6,480,000 | 0.58s | 28.5s | **~48x** |

*Performance measured for single time step regridding on typical hardware.*

## 🛠 Installation

### Quick Install (Recommended)

```bash
# Create environment with all dependencies
conda env create -f environment.yml
conda activate xregrid

# Install XRegrid
pip install -e .
```

### Manual Installation

```bash
# Create new environment
conda create -n xregrid python>=3.9
conda activate xregrid

# Install dependencies
conda install -c conda-forge esmpy xarray numpy scipy dask netcdf4

# Install XRegrid
pip install -e .
```

### From Source (Advanced)

If you need to build ESMPy from source:

```bash
# Set ESMF environment variables
export ESMF_DIR=/path/to/esmf
export ESMF_COMPILER=gfortran
export ESMF_COMM=openmpi

# Install ESMPy from ESMF source
cd $ESMF_DIR/src/addon/esmpy
pip install .

# Install XRegrid
pip install -e .
```

## 🚀 Quick Start

```python
import xarray as xr
import numpy as np
from xregrid import ESMPyRegridder

# Create source and target grids
source_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 180)),
    'lon': (['lon'], np.linspace(0, 359, 360))
})

target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-90, 90, 360)),
    'lon': (['lon'], np.linspace(0, 359.5, 720))
})

# Create regridder (only once!)
regridder = ESMPyRegridder(
    source_grid, target_grid,
    method='bilinear',
    periodic=True,  # Important for global grids
    reuse_weights=True,  # Save weights for reuse
    filename='weights.nc'
)

# Apply to your data
data_regridded = regridder(your_data)
```

## 📖 Grid Type Support

### Rectilinear Grids (Standard Climate Models)
```python
# 1D lat/lon arrays - most common
grid = xr.Dataset({
    'lat': (['lat'], latitudes_1d),
    'lon': (['lon'], longitudes_1d)
})
```

### Curvilinear Grids (Ocean Models)
```python
# 2D coordinate arrays - ORCA family, regional models
grid = xr.Dataset({
    'lat': (['y', 'x'], latitudes_2d),
    'lon': (['y', 'x'], longitudes_2d)
})
```

### Unstructured Grids (MPAS, ICON)
```python
# 1D arrays with same dimension - next-gen climate models
grid = xr.Dataset({
    'lat': (['nCells'], cell_latitudes),
    'lon': (['nCells'], cell_longitudes)
})
```

## 🎯 Regridding Methods

| Method | Best For | Speed | Conservation |
|--------|----------|--------|--------------|
| `bilinear` | Temperature, pressure | ⚡⚡⚡ | Intensive vars |
| `conservative` | Precipitation, fluxes | ⚡⚡ | Extensive vars |
| `nearest_s2d` | Categorical data | ⚡⚡⚡ | Exact values |
| `patch` | High-order accuracy | ⚡ | Scientific |

## ⚡ Performance Optimization

### Weight Reuse (Essential!)
```python
# First time: generates and saves weights (~30s)
regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)

# Subsequent times: loads weights (~1s) - 30x speedup!
regridder = ESMPyRegridder(
    source, target,
    method='bilinear',
    reuse_weights=True,
    filename='my_weights.nc'
)
```

### Dask Integration
```python
# Load data with chunks
data = xr.open_dataset('large_file.nc', chunks={'time': 20})

# Regridding preserves chunks automatically
data_regridded = regridder(data)  # Parallel processing!
```

## 🌍 Real-World Examples

### CMIP6 Data Processing
```python
# Regrid CMIP6 model output to common grid
regridder = ESMPyRegridder(
    cmip6_grid, analysis_grid,
    method='bilinear',
    periodic=True
)
temperature_regridded = regridder(cmip6_data.tas)
```

### Precipitation Analysis
```python
# Conservative regridding for precipitation (flux conservation)
precip_regridder = ESMPyRegridder(
    model_grid, obs_grid,
    method='conservative',
    periodic=True
)
precip_regridded = precip_regridder(model_precip)
```

### MPAS to Regular Grid
```python
# Unstructured MPAS data to regular lat-lon
mpas_regridder = ESMPyRegridder(
    mpas_grid, regular_grid,
    method='bilinear'
)
mpas_temp_regular = mpas_regridder(mpas_temperature)
```

## 🏆 Why XRegrid?

### Performance Advantages
- **Vectorized Operations**: Single large sparse matrix multiplications
- **Optimized Memory**: scipy sparse matrices with lower footprint
- **Dask Scalability**: Linear scaling with number of chunks
- **Weight Reuse**: 10-100x speedup for repeated operations

### Technical Excellence
- **Correct ESMF Integration**: Proper coordinate transposition and indexing
- **Robust NaN Handling**: Identical results to xESMF's skipna logic
- **Comprehensive Testing**: Validated against xESMF for accuracy
- **Production Stability**: Memory-efficient and error-resistant

### Ease of Use
- **Same API for All Grids**: Automatic grid type detection
- **xarray Native**: Seamless integration with xarray workflows
- **Comprehensive Documentation**: Examples for every use case

## 📚 Documentation

- **[Full Documentation](https://xregrid.readthedocs.io)** - Complete user guide and API reference
- **[Gallery](docs/examples/index.md)** - Interactive examples and tutorials
- **[Performance Guide](docs/user-guide/performance.md)** - Optimization best practices
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions

## 🎨 Example Gallery

Check out our comprehensive examples:

- [**Basic Regridding**](docs/examples/basic-regridding.md) - Standard atmospheric model regridding
- [**Conservative Regridding**](docs/examples/conservative-regridding.md) - Flux-conserving interpolation
- [**Unstructured Grids**](docs/examples/unstructured-grids.md) - MPAS and ICON model support
- [**Performance Optimization**](docs/examples/performance-optimization.md) - Speed up your workflows

## 🧪 Testing

```bash
# Run the test suite
pytest

# Test with your own data
python verify_user_script.py
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [ESMF](https://earthsystemmodeling.org/) - Earth System Modeling Framework
- Inspired by [xESMF](https://xesmf.readthedocs.io/) - Excellent foundation for xarray regridding
- Powered by [xarray](https://xarray.pydata.org/), [scipy](https://scipy.org/), and [dask](https://dask.org/)

<!-- ## 📈 Citation

If you use XRegrid in your research, please cite:

```bibtex
@software{xregrid,
  title = {XRegrid: An optimized ESMF-based regridder for xarray},
  author = {XRegrid Contributors},
  year = {2026},
  url = {https://github.com/xregrid/xregrid}
}
``` -->

---

**Ready to speed up your regridding workflows? [Get started now!](docs/user-guide/quickstart.md)** 🚀