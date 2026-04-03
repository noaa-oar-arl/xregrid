import os
import importlib.util
from setuptools import setup

# Use tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        # Fallback for basic installations during bootstrap if tomli is not yet available
        tomllib = None


def get_install_requires():
    """Read dependencies from pyproject.toml and adjust for ESMF environment."""
    # Static fallback list in case pyproject.toml cannot be parsed
    # Must be kept in sync with the primary list in pyproject.toml
    default_deps = [
        "xarray",
        "numpy",
        "scipy",
        "dask",
        "netCDF4",
        "esmpy",
        "cf-xarray",
        "pyproj",
    ]

    if not os.path.exists("pyproject.toml") or tomllib is None:
        deps = default_deps
    else:
        try:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            # Read from the custom [tool.xregrid] section
            deps = (
                data.get("tool", {})
                .get("xregrid", {})
                .get("dependencies", default_deps)
            )
        except Exception:
            deps = default_deps

    # ESMPy Handling:
    # ESMPy is not available on PyPI for many platforms. To ensure xregrid
    # can be installed against an existing ESMPy installation (e.g., from conda
    # or built from source), we check for its presence here.
    # If it's already installed, or if we want to allow installation to proceed
    # so the user can install it manually later, we remove it from install_requires.
    esmpy_installed = False
    try:
        esmpy_installed = importlib.util.find_spec("esmpy") is not None
    except Exception:
        pass

    if "esmpy" in deps:
        if esmpy_installed:
            # Already installed, remove from requires to avoid pip trying to fetch from PyPI
            deps = [d for d in deps if d != "esmpy"]
        elif os.environ.get("ESMFMKFILE"):
            # User has ESMF but maybe not esmpy yet; they likely want to build it.
            print("\n" + "=" * 80)
            print("NOTICE: ESMFMKFILE detected but esmpy is not installed.")
            print(
                "We are omitting the 'esmpy' requirement to allow manual installation."
            )
            print("Please install esmpy from the ESMF source tree:")
            print("  cd $ESMF_DIR/src/addon/esmpy && python setup.py install")
            print("=" * 80 + "\n")
            deps = [d for d in deps if d != "esmpy"]
        else:
            # Not installed and no ESMFMKFILE. Still remove it because it's not on PyPI.
            # We'll warn the user that they need it.
            print("\n" + "=" * 80)
            print("WARNING: 'esmpy' is not installed and is not available on PyPI.")
            print("xregrid requires esmpy to function correctly.")
            print("Please install it via conda-forge:")
            print("  conda install -c conda-forge esmpy")
            print("Or build it from source if you have an existing ESMF installation.")
            print("=" * 80 + "\n")
            deps = [d for d in deps if d != "esmpy"]

    return deps


if __name__ == "__main__":
    setup(
        install_requires=get_install_requires(),
    )
