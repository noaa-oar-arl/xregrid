# High Performance Computing (HPC) and Dask

XRegrid is designed for high-performance regridding on both local workstations and large-scale HPC systems. This guide shows you how to leverage Dask to parallelize your regridding tasks.

## Parallelization Strategies

XRegrid uses Dask to parallelize two distinct phases:

1.  **Parallel Weight Generation**: Large grids (e.g., 3km global) can take minutes to generate weights on a single core. XRegrid can distribute this process across a Dask cluster.
2.  **Parallel Weight Application**: For large datasets, applying the regridding weights can be parallelized using Dask chunks.

---

## 1. Local Dask Cluster

On a local machine with multiple cores, you can use a `LocalCluster` to parallelize your work.

```python
import dask.distributed
from xregrid import Regridder

# Start a local cluster
client = dask.distributed.Client()

# Create regridder with parallel weight generation
regridder = Regridder(
    ds_src, ds_tgt,
    method='bilinear',
    parallel=True # Distribute weight generation
)

# Apply to chunked data
ds_chunked = ds_src.chunk({'time': 10})
regridded = regridder(ds_chunked)
```

---

## 2. NOAA RDHPCS Systems (Hera, Jet, Gaea, Ursa)

XRegrid provides a dedicated helper for setting up SLURM clusters on NOAA's RDHPCS systems. It automatically detects your machine and provides reasonable defaults for each architecture.

```python
from xregrid.utils import get_rdhpcs_cluster
from distributed import Client

# Automatically detect machine (Hera, Jet, Gaea, or Ursa)
# and set up a SLURM cluster
cluster = get_rdhpcs_cluster(account="your_slurm_account")

# Scale the cluster to use 4 SLURM jobs
cluster.scale(jobs=4)
client = Client(cluster)

# All regridding will now be distributed across the cluster
regridder = Regridder(ds_src, ds_tgt, parallel=True)
regridded = regridder(ds_src)
```

### Supported Machines

| Machine | Name | Default Queue | Default Memory |
|---------|------|---------------|----------------|
| **Hera**| `hera` | `hera` | 160 GB / node |
| **Jet** | `jet` | `batch` | 120 GB / node |
| **Gaea**| `gaea-c5`, `gaea-c6` | `batch` | 256/384 GB / node |
| **Ursa**| `ursa` | `u1-compute` | 384 GB / node |

---

## 3. Parallel Weight Generation

When `parallel=True` is passed to the `Regridder` constructor, XRegrid:
- Splits the **target grid** into multiple chunks.
- Submits weight generation tasks to different Dask workers.
- **Assembles weights on the cluster**: To protect the driver from Out-Of-Memory (OOM) crashes, the weights are concatenated on a worker and stored as a Dask Future.

```python
regridder = Regridder(ds_src, ds_tgt, parallel=True, compute=True)
```

- `compute=True` (default): Triggers calculation and assembly immediately.
- `compute=False`: Submits tasks but defers assembly until you call `regridder.compute()`.

---

## 4. Large-Scale Memory Management

When working with massive grids and datasets, follow these best practices:

### Use Weight Reuse
Always save your weights for large grids to avoid recomputing them.

```python
regridder = Regridder(
    ds_src, ds_tgt,
    parallel=True,
    reuse_weights=True,
    filename='3km_weights.nc'
)
```

### Chunk in Time, Not Space
For optimal performance, keep spatial dimensions unchunked and chunk along the time or level dimension.

- **Good**: `{'time': 20, 'lat': 720, 'lon': 1440}`
- **Bad**: `{'time': 20, 'lat': 100, 'lon': 100}`

### Worker-Local Caching
XRegrid automatically caches weights on Dask workers. When regridding multiple time steps, the weight matrix is only sent to each worker once, significantly reducing network overhead.

---

## Troubleshooting Dask

- **"Worker Memory Exceeded"**: Try reducing your chunk size in the time dimension.
- **"Communication Timeout"**: For very large grids, increase the Dask communication timeout: `dask.config.set({"distributed.comm.timeouts.connect": "60s"})`.
- **"Killed Worker"**: Ensure you are not chunking your spatial dimensions. ESMF regridding requires the full spatial domain to be available to each worker for a given time chunk.
