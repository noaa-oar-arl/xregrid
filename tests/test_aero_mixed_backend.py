import numpy as np
import xarray as xr
import dask.array as da
from distributed import Client, LocalCluster
from conftest import setup_esmpy_mock
from xregrid import Regridder, create_global_grid
from xregrid.utils import create_regional_grid


def test_mixed_backend_dataset_regrid():
    """
    Double-Check Test: Verify that a Dataset with mixed NumPy and Dask variables
    regrids correctly when weights are remote (Dask Futures).
    """
    # Start a local cluster
    with LocalCluster(n_workers=2, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            # Setup mocks on workers
            client.run(setup_esmpy_mock)

            # 1. Create grids
            ds_src = create_global_grid(10.0, 10.0)
            ds_tgt = create_regional_grid((20, 40), (40, 60), 10.0, 10.0)

            # 2. Create mixed data
            shape_src = (ds_src.sizes["lat"], ds_src.sizes["lon"])

            # Eager variable (NumPy)
            data_eager = np.random.rand(*shape_src)
            # Lazy variable (Dask)
            data_lazy = da.from_array(np.random.rand(*shape_src), chunks=(9, 18))

            ds_input = xr.Dataset(
                {
                    "var_eager": (["lat", "lon"], data_eager),
                    "var_lazy": (["lat", "lon"], data_lazy),
                },
                coords=ds_src.coords,
            )

            # 3. Initialize Regridder with parallel=True to create remote weights
            regridder = Regridder(ds_src, ds_tgt, method="bilinear", parallel=True)

            # Ensure weights are indeed remote
            assert hasattr(
                regridder._weights_matrix, "key"
            ), "Weights should be Dask Futures"

            # 4. Regrid!
            # Before the fix, this would crash when processing "var_eager"
            ds_out = regridder(ds_input)

            # 5. Verify Results
            assert "var_eager" in ds_out
            assert "var_lazy" in ds_out

            # Check backends are preserved
            assert not hasattr(
                ds_out.var_eager.data, "dask"
            ), "var_eager should remain NumPy-backed"
            assert hasattr(
                ds_out.var_lazy.data, "dask"
            ), "var_lazy should remain Dask-backed"

            # Check shape
            expected_shape_tgt = (ds_tgt.sizes["lat"], ds_tgt.sizes["lon"])
            assert ds_out.var_eager.shape == expected_shape_tgt
            assert ds_out.var_lazy.shape == expected_shape_tgt

            # Verify values
            # (Using .values on dask array triggers compute)
            assert not np.isnan(ds_out.var_eager.values).all()
            assert not np.isnan(ds_out.var_lazy.values).all()

            # Check provenance/history
            assert "Regridded" in ds_out.attrs["history"]
            assert "var_eager" in ds_out.data_vars
            assert "var_lazy" in ds_out.data_vars
