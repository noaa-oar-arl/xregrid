from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


# Global cache for workers to reuse ESMF source objects and weight matrices
# We use builtins to ensure the cache survives module re-imports in Dask workers.
import builtins

if not hasattr(builtins, "_XREGRID_WORKER_CACHE"):
    builtins._XREGRID_WORKER_CACHE = {}  # type: ignore
_WORKER_CACHE = builtins._XREGRID_WORKER_CACHE  # type: ignore


def _setup_worker_cache(key: str, value: Any) -> None:
    """
    Setup a value in the worker-local cache.

    Parameters
    ----------
    key : str
        The cache key.
    value : Any
        The value to store.
    """
    _WORKER_CACHE[key] = value


def _matmul(matrix: Any, data: np.ndarray) -> np.ndarray:
    """
    Backend-agnostic matrix multiplication (matrix @ data.T).T.

    Handles both NumPy and CuPy backends and ensures a NumPy array is returned.

    Parameters
    ----------
    matrix : Any
        The sparse weight matrix.
    data : np.ndarray
        The dense data array (2D: other x spatial).

    Returns
    -------
    np.ndarray
        The result of (matrix @ data.T).T as a NumPy array.
    """
    res = (matrix @ data.T).T
    if hasattr(res, "get"):
        return res.get()
    return res


def _apply_weights_core(
    data_block: np.ndarray,
    weights_matrix: Any,
    dims_source: Tuple[str, ...],
    shape_target: Tuple[int, ...],
    skipna: bool = False,
    total_weights: Optional[np.ndarray] = None,
    na_thres: float = 1.0,
    weights_key: Optional[str] = None,
) -> np.ndarray:
    """
    Apply regridding weights to a data block (NumPy array).

    Parameters
    ----------
    data_block : np.ndarray
        The input data block. Core dimensions must be at the end.
    weights_matrix : scipy.sparse.csr_matrix or str
        The sparse weight matrix or a string key for worker-local cache.
    dims_source : tuple of str
        The names of the source spatial dimensions.
    shape_target : tuple of int
        The shape of the target spatial grid.
    skipna : bool, default False
        Whether to handle NaNs by re-normalizing weights.
    total_weights : np.ndarray, optional
        Pre-computed sum of weights for each destination cell.
    na_thres : float, default 1.0
        Threshold for NaN handling.
    weights_key : str, optional
        Explicit key for the weights in the worker cache.

    Returns
    -------
    np.ndarray
        The regridded data block.
    """
    # Worker-local cache retrieval (Aero Protocol: Dask Efficiency)
    weights_matrix_key = weights_key
    if isinstance(weights_matrix, str):
        weights_matrix_key = weights_matrix
        weights_matrix = _WORKER_CACHE.get(weights_matrix_key)

    if weights_matrix is None:
        raise RuntimeError(
            f"Weights key '{weights_matrix_key}' not found in worker cache."
        )

    if isinstance(total_weights, str):
        total_weights_key = total_weights
        total_weights = _WORKER_CACHE.get(total_weights_key)
        if total_weights is None:
            raise RuntimeError(
                f"Total weights key '{total_weights_key}' not found in worker cache."
            )

    original_shape = data_block.shape
    # Core dimensions are at the end
    n_source_dims = len(dims_source)
    spatial_shape = original_shape[len(original_shape) - n_source_dims :]
    other_dims_shape = original_shape[: len(original_shape) - n_source_dims]
    n_spatial = int(np.prod(spatial_shape))
    n_other = int(np.prod(other_dims_shape))

    # Optimization: avoid reshape if already 2D and spatial is flat
    if len(original_shape) == 2 and n_other == original_shape[0]:
        flat_data = data_block
    else:
        flat_data = data_block.reshape(n_other, n_spatial)

    if skipna:
        # Use a more memory-efficient NaN detection (Aero Protocol: Performance)
        mask = np.isnan(flat_data)
        has_nans = np.any(mask)

        if not has_nans:
            # Fast path: No NaNs in this data block
            result = _matmul(weights_matrix, flat_data)
            if total_weights is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result /= total_weights
        else:
            # Slow path: Handle NaNs by re-normalizing weights
            is_mask_stationary = True
            if n_other > 1:
                # Optimized stationary mask detection using heuristic early exit
                # (Aero Protocol: Speedup for large grids)
                mask0 = mask[0]
                sample_size = min(1000, n_spatial)
                # Check first sample points across all time steps first
                if not np.all(mask[:, :sample_size] == mask0[:sample_size]):
                    is_mask_stationary = False
                else:
                    for i in range(1, n_other):
                        if not np.array_equal(mask[i], mask0):
                            is_mask_stationary = False
                            break

            zero = flat_data.dtype.type(0)
            if is_mask_stationary:
                # Memory win: Use broadcasting for the stationary mask
                mask = mask[0:1]
                safe_data = np.where(mask, zero, flat_data)
            else:
                safe_data = np.where(mask, zero, flat_data)

            result = _matmul(weights_matrix, safe_data)

            if is_mask_stationary:
                # Optimization: Cache the stationary weights_sum to avoid redundant sparse matmuls
                # across multiple Dask chunks (e.g. different time segments).
                weights_sum = None
                if weights_matrix_key:
                    ws_cache_key = f"ws_{weights_matrix_key}"
                    mask_cache_key = f"mask_{weights_matrix_key}"

                    if ws_cache_key in _WORKER_CACHE:
                        # Validate that the mask is identical to the cached one
                        cached_mask = _WORKER_CACHE.get(mask_cache_key)
                        if np.array_equal(mask[0:1], cached_mask):
                            weights_sum = _WORKER_CACHE[ws_cache_key]

                if weights_sum is None:
                    # Compute normalization only for the first (representative) mask
                    # Use float32 for normalization weights to save memory on large grids
                    valid_mask_single = np.logical_not(mask[0:1]).astype(np.float32)
                    weights_sum = _matmul(weights_matrix, valid_mask_single)

                    if weights_matrix_key:
                        _WORKER_CACHE[ws_cache_key] = weights_sum
                        _WORKER_CACHE[mask_cache_key] = mask[0:1].copy()
            else:
                # Sum weights of valid (non-NaN) points for each slice
                # We use float32 to keep peak memory down for ~1km grids
                valid_mask = np.logical_not(mask).astype(np.float32)
                weights_sum = _matmul(weights_matrix, valid_mask)

            with np.errstate(divide="ignore", invalid="ignore"):
                result /= weights_sum
                if total_weights is not None:
                    fraction_valid = weights_sum / total_weights
                    # Masking of low-confidence points
                    # Ensure NaN value doesn't force promotion to float64
                    nan_val = result.dtype.type(np.nan)
                    result = np.where(
                        fraction_valid < (1.0 - na_thres - 1e-6), nan_val, result
                    )
    else:
        # Standard path (skipna=False): Just apply weights
        result = _matmul(weights_matrix, flat_data)

    new_shape = other_dims_shape + shape_target
    return result.reshape(new_shape).astype(data_block.dtype, copy=False)
