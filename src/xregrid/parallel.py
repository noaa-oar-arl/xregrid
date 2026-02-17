from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import xarray as xr

from xregrid.core import _WORKER_CACHE, _setup_worker_cache
from xregrid.grid import _create_esmf_grid


def _assemble_weights_task(
    results: list[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]],
    n_src: int,
    n_dst: int,
) -> Any:
    """
    Internal worker task to assemble weights from multiple chunks.

    Parameters
    ----------
    results : list of tuples
        List of (rows, cols, data, error) from worker tasks.
    n_src : int
        Total number of source points.
    n_dst : int
        Total number of destination points.

    Returns
    -------
    scipy.sparse.csr_matrix
        The concatenated sparse weight matrix.
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    all_rows, all_cols, all_data = [], [], []
    for r, c, d, err in results:
        if err:
            raise RuntimeError(f"Weight generation error: {err}")
        if len(r) > 0:
            all_rows.append(r)
            all_cols.append(c)
            all_data.append(d)

    if not all_rows:
        return coo_matrix((n_dst, n_src)).tocsr()

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)
    return coo_matrix((data, (rows, cols)), shape=(n_dst, n_src)).tocsr()


def _get_weights_sum_task(matrix: Any) -> np.ndarray:
    """
    Internal worker task to compute the sum of weights for normalization.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The sparse weight matrix.

    Returns
    -------
    np.ndarray
        The sum of weights per destination point.
    """
    import numpy as np

    # Return flattened 1D array (n_dst,) for easier reshaping and broadcasting
    return np.array(matrix.sum(axis=1)).flatten()


def _get_nnz_task(matrix: Any) -> int:
    """
    Internal worker task to compute the number of non-zero elements.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The sparse weight matrix.

    Returns
    -------
    int
        The number of non-zero elements.
    """
    return int(matrix.nnz)


def _populate_cache_task(value: Any, key: str) -> None:
    """
    Internal worker task to populate the worker-local cache with a value.

    Parameters
    ----------
    value : Any
        The value to cache (e.g., weight matrix).
    key : str
        The cache key.
    """
    _setup_worker_cache(key, value)


def _sync_cache_from_worker_data(
    future_key: str, cache_key: str, dask_worker: Any = None
) -> None:
    """
    Internal worker task to sync worker-local cache from Dask worker data.

    Used to ensure all workers have a replicated Future in their local cache.
    Accepts dask_worker argument provided by client.run().

    Parameters
    ----------
    future_key : str
        The Dask key of the replicated Future.
    cache_key : str
        The key to use in _WORKER_CACHE.
    dask_worker : Any, optional
        The Dask worker instance, automatically passed by client.run().
    """
    if dask_worker is not None and future_key in dask_worker.data:
        _setup_worker_cache(cache_key, dask_worker.data[future_key])
    else:
        # Fallback to get_worker() if run manually
        try:
            from dask.distributed import get_worker

            worker = get_worker()
            if future_key in worker.data:
                _setup_worker_cache(cache_key, worker.data[future_key])
        except (ImportError, ValueError):
            pass


def _compute_chunk_weights(
    source_ds: xr.Dataset,
    chunk_ds: xr.Dataset,
    method: str,
    dest_slice_info: Union[np.ndarray, Tuple[int, int, int, int, int]],
    extrap_method: Optional[str] = None,
    extrap_dist_exponent: float = 2.0,
    mask_var: Optional[str] = None,
    periodic: bool = False,
    coord_sys: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """
    Worker function to compute weights for a specific chunk of the target grid.

    Uses worker-local caching for source ESMF objects to avoid redundant initialization.

    Parameters
    ----------
    source_ds : xr.Dataset
        The source grid dataset.
    chunk_ds : xr.Dataset
        A chunk of the target grid dataset.
    method : str
        The regridding method.
    dest_slice_info : np.ndarray or tuple
        If tuple, contains (i0_start, i0_end, i1_start, i1_end, total_size1) to
        reconstruct global indices locally. If ndarray, used directly.
    extrap_method : str, optional
        The extrapolation method.
    extrap_dist_exponent : float, default 2.0
        The IDW extrapolation exponent.
    mask_var : str, optional
        Variable name for the mask.
    periodic : bool, default False
        Whether the grid is periodic in longitude.
    coord_sys : Any, optional
        The coordinate system (esmpy.CoordSys).

    Returns
    -------
    rows : np.ndarray
        Destination indices (0-based, global).
    cols : np.ndarray
        Source indices (0-based).
    data : np.ndarray
        Regridding weights.
    error : str or None
        Error message if weight generation failed.
    """
    try:
        import esmpy

        # Initialize Manager if not already done in this process
        esmpy.Manager(debug=False)

        # 1. Get or create source ESMF Field
        # We use id(source_ds) as a key. Dask ensures the same object is reused on a worker.
        src_cache_key = (id(source_ds), method, periodic, mask_var, coord_sys)
        if src_cache_key in _WORKER_CACHE:
            src_field, src_orig_idx = _WORKER_CACHE[src_cache_key]
        else:
            src_obj, _, src_orig_idx = _create_esmf_grid(
                source_ds, method, periodic, mask_var, coord_sys=coord_sys
            )
            if isinstance(src_obj, esmpy.Mesh):
                meshloc = (
                    esmpy.MeshLoc.ELEMENT
                    if method == "conservative"
                    else esmpy.MeshLoc.NODE
                )
                src_field = esmpy.Field(src_obj, name="src", meshloc=meshloc)
            else:
                src_field = esmpy.Field(src_obj, name="src")
            _WORKER_CACHE[src_cache_key] = (src_field, src_orig_idx)

        # 2. Create target ESMF object (chunk is small, no need to cache)
        dst_obj, _, dst_orig_idx = _create_esmf_grid(
            chunk_ds, method, periodic=False, mask_var=None, coord_sys=coord_sys
        )
        if isinstance(dst_obj, esmpy.Mesh):
            meshloc = (
                esmpy.MeshLoc.ELEMENT
                if method == "conservative"
                else esmpy.MeshLoc.NODE
            )
            dst_field = esmpy.Field(dst_obj, name="dst", meshloc=meshloc)
        else:
            dst_field = esmpy.Field(dst_obj, name="dst")

        # 3. Setup regridding parameters
        method_map = {
            "bilinear": esmpy.RegridMethod.BILINEAR,
            "conservative": esmpy.RegridMethod.CONSERVE,
            "nearest_s2d": esmpy.RegridMethod.NEAREST_STOD,
            "nearest_d2s": esmpy.RegridMethod.NEAREST_DTOS,
            "patch": esmpy.RegridMethod.PATCH,
        }
        extrap_method_map = {
            "nearest_s2d": esmpy.ExtrapMethod.NEAREST_STOD,
            "nearest_idw": esmpy.ExtrapMethod.NEAREST_IDAVG,
            "creep_fill": esmpy.ExtrapMethod.CREEP_FILL,
        }

        regrid_kwargs = {
            "regrid_method": method_map[method],
            "unmapped_action": esmpy.UnmappedAction.IGNORE,
            "factors": True,
        }
        if extrap_method:
            regrid_kwargs["extrap_method"] = extrap_method_map[extrap_method]
            regrid_kwargs["extrap_dist_exponent"] = extrap_dist_exponent

        if mask_var:
            regrid_kwargs["src_mask_values"] = np.array([0], dtype=np.int32)

        if method == "conservative":
            regrid_kwargs["norm_type"] = esmpy.NormType.FRACAREA

        # 4. Generate weights
        regrid = esmpy.Regrid(src_field, dst_field, **regrid_kwargs)
        weights = regrid.get_weights_dict(deep_copy=True)

        # 5. Dask Resource Hygiene: Destroy temporary ESMF objects (Aero Protocol)
        # We don't destroy src_field because it's cached.
        regrid.destroy()
        dst_field.destroy()
        if hasattr(dst_obj, "destroy"):
            dst_obj.destroy()

        # 6. Map local destination indices to global grid indices
        if isinstance(dest_slice_info, np.ndarray):
            # Backward compatibility or direct index passing
            global_indices = dest_slice_info
        else:
            # Reconstruct global indices locally to save driver memory (Aero Protocol)
            i0_start, i0_end, i1_start, i1_end, total_size1 = dest_slice_info
            if total_size1 == 0:
                # Unstructured target (1D)
                global_indices = np.arange(i0_start, i0_end)
            else:
                # Structured target (2D)
                n0, n1 = i0_end - i0_start, i1_end - i1_start
                global_indices = (
                    (np.arange(n0)[:, None] + i0_start) * total_size1
                    + (np.arange(n1) + i1_start)
                ).flatten()

        row_dst = weights["row_dst"] - 1
        col_src = weights["col_src"] - 1

        if dst_orig_idx is not None and method == "conservative":
            row_dst = dst_orig_idx[row_dst]

        if src_orig_idx is not None and method == "conservative":
            col_src = src_orig_idx[col_src]

        rows = global_indices[row_dst]
        cols = col_src
        data = weights["weights"]

        return rows, cols, data, None

    except Exception as e:
        import traceback

        return (
            np.array([]),
            np.array([]),
            np.array([]),
            f"{str(e)}\n{traceback.format_exc()}",
        )
