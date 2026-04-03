# Installation Guide

XRegrid recommends installation via `mamba` or `micromamba` for the most reliable dependency management, especially for the ESMPy requirement.

## Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/bbakernoaa/xregrid.git
cd xregrid

# Create the environment from the provided yaml file
mamba env create -f environment.yml

# Activate the environment
mamba activate xregrid
```

## Important: ESMPy Dependency

XRegrid depends on `esmpy`, which is the Python interface to the Earth System Modeling Framework (ESMF). **ESMPy is not available on PyPI**.

If you are not using `mamba`/`conda`, you must ensure `esmpy` is installed manually before or after installing `xregrid`. XRegrid's installation will proceed without it, but it will not function until `esmpy` is present.

## Detailed Instructions

For more detailed installation options, including building from source or handling specific platform requirements, please see our [Online Installation Guide](https://bbakernoaa.github.io/xregrid/installation).

## Documentation

Full documentation and examples are available at [https://bbakernoaa.github.io/xregrid/](https://bbakernoaa.github.io/xregrid/).
