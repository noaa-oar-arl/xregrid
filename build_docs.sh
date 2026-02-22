#!/bin/bash
set -e

echo "XRegrid Documentation Builder (Zensical Edition)"
echo "==============================================="

# Check if we are in a conda environment, if not try to activate
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Activating conda environment..."
    # Attempt to activate, but don't fail if conda is not installed
    command -v conda >/dev/null 2>&1 && conda activate xregrid || echo "Warning: Could not activate conda environment. Continuing with current environment."
fi

# Install documentation dependencies if needed
echo "Checking documentation dependencies..."
pip install zensical mkdocstrings mkdocstrings-python mkdocs-gallery matplotlib cartopy pooch --quiet

# Install package in development mode
echo "Installing xregrid package..."
pip install -e . --quiet --no-deps || echo "Warning: Could not install in editable mode."

# Clean previous build
echo "Cleaning previous build..."
rm -rf site/

# Build documentation
echo "Building documentation with Zensical..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Monkeypatch for mkdocs-gallery compatibility if needed (applied to current python process)
# and then call zensical CLI
python -c "import ast; \
ast.Str = getattr(ast, 'Str', ast.Constant); \
ast.Num = getattr(ast, 'Num', ast.Constant); \
ast.Bytes = getattr(ast, 'Bytes', ast.Constant); \
ast.NameConstant = getattr(ast, 'NameConstant', ast.Constant); \
import subprocess; subprocess.run(['zensical', 'build', '--clean'], check=True)"

echo ""
echo "Documentation built successfully!"
echo "To serve locally: zensical serve"
echo "To deploy: zensical build && # follow zensical deployment guide"
echo ""
echo "Built site is in: ./site/"
