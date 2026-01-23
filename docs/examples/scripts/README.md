# XRegrid Examples Gallery

This gallery demonstrates the capabilities of XRegrid through practical, runnable examples.

## Overview

XRegrid provides high-performance regridding for earth science applications. These examples showcase:

- **Basic Operations**: Getting started with common regridding tasks
- **Advanced Features**: Handling different grid types and optimization techniques  
- **Real-world Applications**: Practical examples for climate and atmospheric science

## Examples Description

### Basic Regridding
Demonstrates standard rectilinear grid regridding from 1° to 0.5° resolution using bilinear interpolation.

### Conservative Regridding  
Shows flux-conserving interpolation essential for precipitation and radiation data.

### Unstructured Grids
Illustrates regridding from unstructured grids (MPAS/ICON style) to structured grids.

### Performance Optimization
Focuses on weight reuse and other optimization techniques for production workflows.

## Running the Examples

Each example is self-contained and includes synthetic data generation. To run:

```bash
python plot_basic_regridding.py
```

The examples generate both plots and performance metrics to help you understand XRegrid's capabilities.