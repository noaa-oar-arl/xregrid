# XRegrid Examples

XRegrid provides a comprehensive gallery of examples demonstrating various earth science regridding use cases.

## View the Examples Gallery

The most up-to-date and visual examples are available in our [Online Documentation Gallery](https://xregrid.readthedocs.io/examples/generated/).

## Local Examples

You can also find the source code for all examples in the [docs/examples/scripts/](docs/examples/scripts/) directory of this repository.

Each script is self-contained and demonstrates a specific feature:

- `plot_basic_regridding.py`: Standard rectilinear grid regridding.
- `plot_conservative_regridding.py`: Flux-conserving interpolation for precipitation.
- `plot_curvilinear_grids.py`: Handling curvilinear Arctic grids (RASM).
- `plot_unstructured_grids.py`: Regridding MPAS/ICON style unstructured grids.
- `plot_weather_data.py`: Regridding station-like point data.
- `plot_performance_optimization.py`: Efficient workflows using weight reuse.
- `plot_esmpy_comparison.py`: Comparing code complexity: ESMPy vs. XRegrid.

## Running Examples

To run an example locally:

```bash
# Ensure you have the dependencies installed
pip install xarray numpy matplotlib esmpy

# Run a specific example
python docs/examples/scripts/plot_basic_regridding.py
```
