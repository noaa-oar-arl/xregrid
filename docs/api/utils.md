# Utilities

XRegrid provides several utility functions for creating standard grids, loading ESMF-formatted files, and performing common spatial operations.

## Grid Generation

### create_global_grid

::: xregrid.create_global_grid

Create a global rectilinear grid dataset with a specified resolution.

```python
from xregrid import create_global_grid

# Create a 1x1 degree global grid with bounds
ds = create_global_grid(res_lat=1.0, res_lon=1.0)
```

### create_regional_grid

::: xregrid.create_regional_grid

Create a regional rectilinear grid dataset for a specific geographic bounding box.

```python
from xregrid import create_regional_grid

# Create a regional grid over Europe
ds = create_regional_grid(
    lat_range=(35, 70),
    lon_range=(-10, 40),
    res_lat=0.25,
    res_lon=0.25
)
```

### create_grid_like

::: xregrid.create_grid_like

Create a new grid dataset with the same extent and CRS as an existing object.

```python
from xregrid.utils import create_grid_like

# Create a 0.5 degree grid matching the extent of an existing dataset
new_grid = create_grid_like(ds, res=0.5)
```

### create_grid_from_crs

::: xregrid.create_grid_from_crs

Create a structured grid dataset from a Coordinate Reference System (CRS) and extent.

```python
from xregrid import create_grid_from_crs

# Create a Lambert Conformal Conic grid over North America
extent = (-2500000, 2500000, -2000000, 2000000)
res = (12000, 12000) # 12km
crs = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

ds = create_grid_from_crs(crs, extent, res)
```

### create_grid_from_ioapi

::: xregrid.create_grid_from_ioapi

Create a structured grid dataset from IOAPI-compliant metadata.

```python
from xregrid.utils import create_grid_from_ioapi

metadata = {
    "GDTYP": 2,
    "P_ALP": 30.0,
    "P_BET": 60.0,
    "XCENT": -97.0,
    "YCENT": 40.0,
    "XORIG": -1000.0,
    "YORIG": -1000.0,
    "XCELL": 500.0,
    "YCELL": 500.0,
    "NCOLS": 100,
    "NROWS": 100,
}

ds = create_grid_from_ioapi(metadata)
```

### create_mesh_from_coords

::: xregrid.utils.create_mesh_from_coords

Create an unstructured mesh dataset from 1D coordinates and a CRS.

```python
from xregrid.utils import create_mesh_from_coords
import numpy as np

lons = np.random.uniform(0, 360, 1000)
lats = np.random.uniform(-90, 90, 1000)
ds_mesh = create_mesh_from_coords(lons, lats, crs="EPSG:4326")
```

## Spatial Operations

### spatial_slice

::: xregrid.utils.spatial_slice

Slice an xarray object to a spatial extent, robustly handling longitude wrapping.

```python
from xregrid.utils import spatial_slice

# Slice a 0-360 grid to a region crossing the dateline (-20 to 20 lon)
subset = spatial_slice(ds, extent=(-20, 20, 30, 50))
```

### unstructured_to_scrip

::: xregrid.utils.unstructured_to_scrip

Canonicalize an unstructured dataset (UGRID or MPAS) to SCRIP format.

```python
from xregrid.utils import unstructured_to_scrip

scrip_ds = unstructured_to_scrip(ds_ugrid)
```

### mpas_to_scrip

::: xregrid.utils.mpas_to_scrip

Convert an MPAS-native dataset to a CF-compliant SCRIP-style format.

```python
from xregrid.utils import mpas_to_scrip

scrip_ds = mpas_to_scrip(ds_mpas)
```

## ESMF File Support

### load_esmf_file

::: xregrid.load_esmf_file

Load an ESMF mesh, mosaic, or grid file into an xarray Dataset.

```python
from xregrid import load_esmf_file

# Load an ESMF mesh file
ds = load_esmf_file("path/to/mesh.nc")
```

## High-Performance Computing

### get_rdhpcs_cluster

::: xregrid.utils.get_rdhpcs_cluster

Create a dask-jobqueue SLURMCluster for NOAA RDHPCS systems (Hera, Jet, Gaea, Ursa).

```python
from xregrid.utils import get_rdhpcs_cluster
from distributed import Client

# Automatically detect machine and setup cluster
cluster = get_rdhpcs_cluster(account="your_account")
cluster.scale(jobs=4)
client = Client(cluster)
```
