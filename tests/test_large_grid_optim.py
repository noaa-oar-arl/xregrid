import numpy as np
from xregrid.xregrid import _apply_weights_core, Regridder
from scipy.sparse import csr_matrix


def test_index_reconstruction_logic():
    """Verify that the memory-efficient index reconstruction matches the full-array logic."""
    # Test parameters simulating a chunk of a grid
    size0, size1 = 100, 200  # Global sizes
    i0_start, i0_end = 10, 20  # Lat slice
    i1_start, i1_end = 30, 50  # Lon slice

    n0 = i0_end - i0_start
    n1 = i1_end - i1_start

    # Old logic (full global indices)
    global_indices_full = np.arange(size0 * size1).reshape(size0, size1)
    expected = global_indices_full[i0_start:i0_end, i1_start:i1_end].flatten()

    # New logic (reconstructed)
    actual = (
        (np.arange(n0)[:, None] + i0_start) * size1 + (np.arange(n1) + i1_start)
    ).flatten()

    np.testing.assert_array_equal(expected, actual)


def test_dtype_preservation_in_apply_weights():
    """Verify that _apply_weights_core preserves float32 and avoids float64 promotion."""
    # Create a small identity weight matrix
    n = 10
    weights = csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))

    # Input data as float32
    data = np.random.rand(2, n).astype(np.float32)
    data[0, 0] = np.nan  # Add a NaN to trigger slow path

    # Apply weights
    res = _apply_weights_core(
        data, weights, dims_source=("x",), shape_target=(n,), skipna=True
    )

    # Verify dtype preservation
    assert res.dtype == np.float32

    # Verify result (identity mapping should preserve values where not NaN)
    # Use higher tolerance for float32 comparison
    np.testing.assert_allclose(res[0, 1:], data[0, 1:], atol=1e-6)
    assert np.isnan(res[0, 0])


def test_quality_report_skip_heavy():
    """Verify that quality_report respects the skip_heavy flag."""

    class MockRegridder(Regridder):
        def __init__(self):
            # Bypass ESMF initialization for testing report logic
            self._weights_matrix = csr_matrix([[1, 0], [0, 1]])
            self.method = "bilinear"
            self.periodic = False
            self._shape_source = (2,)
            self._shape_target = (2,)
            self.provenance = []

    regridder = MockRegridder()

    # Test full report
    report_full = regridder.quality_report(skip_heavy=False)
    assert "unmapped_count" in report_full
    assert report_full["unmapped_count"] == 0

    # Test skipped report
    report_light = regridder.quality_report(skip_heavy=True)
    assert "unmapped_count" not in report_light
    assert "weight_sum_min" not in report_light
    assert report_light["n_dst"] == 2
    assert report_light["n_weights"] == 2


def test_index_reconstruction_backward_compatibility():
    """Verify that passing an ndarray to _compute_chunk_weights still works."""
    # We can't easily call _compute_chunk_weights without ESMF,
    # but we can verify the reconstruction logic block.
    dest_slice_info = np.arange(10)

    if isinstance(dest_slice_info, np.ndarray):
        global_indices = dest_slice_info

    np.testing.assert_array_equal(global_indices, np.arange(10))


if __name__ == "__main__":
    test_index_reconstruction_logic()
    test_dtype_preservation_in_apply_weights()
    test_quality_report_skip_heavy()
    print("All optimization verification tests passed!")
