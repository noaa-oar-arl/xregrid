# XRegrid Examples Gallery

This gallery demonstrates the capabilities of XRegrid through practical, runnable examples leveraging standard xarray tutorial datasets.

## Examples Overview

### [Basic Regridding](plot_basic_regridding.py)
Standard rectilinear grid regridding using ERA-Interim data.

### [Conservative Regridding](plot_conservative_regridding.py)
Flux-conserving interpolation essential for preserving area-weighted integrals.

### [Curvilinear Grids](plot_curvilinear_grids.py)
Regridding from curvilinear Arctic grids (RASM) to standard rectilinear grids.

### [Weather Data: Station to Grid](plot_weather_data.py)
Regridding station-like point data to a regular 2D grid using nearest-neighbor methods.

### [Performance Optimization](plot_performance_optimization.py)
Efficient workflows using weight reuse to speed up repeated regridding operations.

### [ESMPy vs. XRegrid](plot_esmpy_comparison.py)
A comparison of code complexity between raw ESMPy and the XRegrid API.
