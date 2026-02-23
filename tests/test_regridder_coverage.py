import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from xregrid import Regridder, create_global_grid
from xregrid.core import _WORKER_CACHE
from xregrid.grid import (
    _create_esmf_grid,
    _get_mesh_info,
    _get_non_spatial_dims,
    _get_unstructured_mesh_info,
)
from xregrid.parallel import (
    _assemble_weights_task,
    _get_nnz_task,
    _sync_cache_from_worker_data,
)


def test_regridder_mpi_parallel_error():
    """Verify ValueError when both mpi and parallel are True."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    with pytest.raises(ValueError, match="Cannot use both MPI and Dask"):
        Regridder(src, tgt, mpi=True, parallel=True)


def test_regridder_missing_dask_error():
    """Verify ImportError when parallel=True but dask.distributed is missing."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="Dask distributed is required"):
            Regridder(src, tgt, parallel=True)


def test_regridder_save_load_weights(tmp_path):
    """Verify saving and loading weights."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    weight_file = str(tmp_path / "weights.nc")

    # Weights are saved automatically if reuse_weights=True and file doesn't exist
    Regridder(src, tgt, method="bilinear", filename=weight_file, reuse_weights=True)
    assert os.path.exists(weight_file)

    # Load weights
    regridder2 = Regridder.from_weights(weight_file, src, tgt)
    assert regridder2.method == "bilinear"

    # Verify validation fails with wrong parameters
    with pytest.raises(ValueError, match="does not match loaded weights method"):
        Regridder.from_weights(weight_file, src, tgt, method="conservative")


def test_regrid_dataset_coverage():
    """Verify _regrid_dataset with various variable types."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    ds = xr.Dataset(
        data_vars={
            "var1": (["lat", "lon"], np.random.rand(18, 36)),
            "var2": (["lat", "lon"], np.random.rand(18, 36)),
            "scalar": 42,
            "other": (["time"], [1, 2, 3]),
        },
        coords={"lat": src.lat, "lon": src.lon, "time": [0, 1, 2]},
    )

    regridder = Regridder(src, tgt)
    res = regridder(ds)

    assert "var1" in res.data_vars
    assert "var2" in res.data_vars
    assert "scalar" in res.data_vars
    assert "other" in res.data_vars
    # create_global_grid(5, 5) gives (36, 72)
    assert res.var1.shape == (36, 72)


def test_extrap_methods_coverage():
    """Verify different extrapolation methods."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    for method in ["nearest_s2d", "nearest_idw", "creep_fill"]:
        regridder = Regridder(src, tgt, extrap_method=method, extrap_dist_exponent=3.0)
        assert regridder.extrap_method == method


def test_regridder_repr_lazy():
    """Verify __repr__ with lazy weights."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)

    regridder = Regridder(src, tgt, parallel=True, compute=False)

    class MockFuture:
        def __init__(self):
            self.key = "some_key"

    regridder._weights_matrix = MockFuture()

    repr_str = repr(regridder)
    assert "quality=lazy" in repr_str


def test_regridder_quality_report_coverage():
    """Verify quality_report with different options."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)

    report = regridder.quality_report(format="dataset")
    assert isinstance(report, xr.Dataset)
    assert "unmapped_fraction" in report.data_vars


# --- Additional Coverage Tests ---


def test_get_non_spatial_dims_z_axis():
    """Verify Z axis detection in _get_non_spatial_dims."""
    ds = xr.Dataset(coords={"lev": [1, 2, 3]})
    ds.lev.attrs["axis"] = "Z"
    dims = _get_non_spatial_dims(ds)
    assert "lev" in dims


def test_get_mesh_info_uxarray_mock():
    """Verify uxarray object support in _get_mesh_info."""
    mock_uxgrid = MagicMock()
    mock_uxgrid.node_lat = xr.DataArray([0.0, 10.0], dims="n_node")
    mock_uxgrid.node_lon = xr.DataArray([0.0, 20.0], dims="n_node")

    class MockDS:
        def __init__(self):
            self.uxgrid = mock_uxgrid
            self.data_vars = {}
            self.dims = {}
            self.coords = {}

        def isel(self, *args, **kwargs):
            return self

    ds = MockDS()

    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    assert is_unstructured
    assert "n_node" in dims
    assert shape == (2,)


def test_get_mesh_info_lat_node_fallback():
    """Verify fallback to lat_node/lon_node in _get_mesh_info."""
    ds = xr.Dataset(
        coords={
            "lat_node": (["n_node"], [0.0, 10.0]),
            "lon_node": (["n_node"], [0.0, 20.0]),
        }
    )
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
    assert is_unstructured
    assert "n_node" in dims


def test_get_unstructured_mesh_info_mpas_no_nedges():
    """Verify MPAS support without nEdgesOnCell in _get_unstructured_mesh_info."""
    ds = xr.Dataset(
        data_vars={
            "verticesOnCell": (["nCells", "maxEdges"], [[1, 2, 3]]),
            "latVertex": (["nVertices"], [0.0, 1.0, 2.0]),
            "lonVertex": (["nVertices"], [0.0, 1.0, 2.0]),
        }
    )
    res = _get_unstructured_mesh_info(ds)
    assert res is not None
    assert len(res) == 6


def test_get_unstructured_mesh_info_ugrid_node_coords_attr():
    """Verify UGRID node_coordinates attribute support."""
    ds = xr.Dataset(
        data_vars={
            "mesh": (
                [],
                0,
                {
                    "cf_role": "mesh_topology",
                    "face_node_connectivity": "face_nodes",
                    "node_coordinates": "lon_u lat_u",
                },
            ),
            "face_nodes": (["n_face", "n_node_per_face"], [[0, 1, 2]]),
            "lon_u": (["n_node"], [0.0, 1.0, 2.0]),
            "lat_u": (["n_node"], [0.0, 1.0, 2.0]),
        }
    )
    ds.face_nodes.attrs["start_index"] = 0
    res = _get_unstructured_mesh_info(ds)
    assert res is not None
    assert np.allclose(res[0], [0.0, 1.0, 2.0])


def test_create_esmf_grid_periodic_bounds():
    """Verify periodic grid with bounds in _create_esmf_grid."""
    lon = np.linspace(0, 360, 36, endpoint=False)
    lat = np.linspace(-90, 90, 19)
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    ds.lat.attrs["standard_name"] = "latitude"
    ds.lon.attrs["standard_name"] = "longitude"
    ds.coords["lat_b"] = (["lat", "nv"], np.zeros((19, 2)))
    ds.coords["lon_b"] = (["lon", "nv"], np.zeros((36, 2)))
    ds.lat.attrs["bounds"] = "lat_b"
    ds.lon.attrs["bounds"] = "lon_b"

    with patch("esmpy.Grid") as mock_grid_cls:
        mock_grid = MagicMock()
        mock_grid.get_coords.side_effect = lambda dim, staggerloc: np.zeros(
            (36, 19) if staggerloc == 0 else (36, 20)
        )
        mock_grid_cls.return_value = mock_grid

        grid, prov, _ = _create_esmf_grid(ds, method="conservative", periodic=True)
        assert grid is not None


def test_create_esmf_grid_locstream_cart():
    """Verify LocStream creation with CART coordinate system."""
    ds = xr.Dataset(
        coords={
            "lat": (["n"], [0.0, 10.0]),
            "lon": (["n"], [0.0, 20.0]),
        }
    )
    import esmpy

    grid, prov, _ = _create_esmf_grid(
        ds, method="nearest_s2d", periodic=False, coord_sys=esmpy.CoordSys.CART
    )
    assert isinstance(grid, esmpy.LocStream)


def test_regridder_normalize_descending():
    """Verify sorting of descending coordinates in Regridder."""
    src = xr.Dataset(
        coords={
            "lat": (["lat"], [10.0, 0.0]),
            "lon": (["lon"], [0.0, 10.0]),
        }
    )
    src.lat.attrs["units"] = "degrees_north"
    src.lon.attrs["units"] = "degrees_east"
    tgt = create_global_grid(10, 10)

    regridder = Regridder(src, tgt)
    assert regridder._src_was_sorted
    assert regridder.source_grid_ds.lat.values[0] == 0.0


def test_regridder_mpi_non_root_rank():
    """Verify logic for non-root ranks in MPI mode."""

    mock_mpi = MagicMock()
    mock_comm = MagicMock()
    mock_mpi.COMM_WORLD = mock_comm
    mock_comm.gather.return_value = None

    with (
        patch("esmpy.pet_count", return_value=2),
        patch("esmpy.local_pet", return_value=1),
        patch.dict(sys.modules, {"mpi4py": mock_mpi}),
    ):
        src = create_global_grid(10, 10)
        tgt = create_global_grid(5, 5)
        regridder = Regridder(src, tgt, mpi=True)
        assert regridder.weights.nnz == 0


def test_assemble_weights_task_error():
    """Verify error handling in _assemble_weights_task."""
    results = [(None, None, None, "Mock Error")]
    with pytest.raises(RuntimeError, match="Weight generation error: Mock Error"):
        _assemble_weights_task(results, 10, 10)


def test_assemble_weights_task_empty():
    """Verify _assemble_weights_task with no results."""
    results = [(np.array([]), np.array([]), np.array([]), None)]
    res = _assemble_weights_task(results, 5, 5)
    assert res.nnz == 0
    assert res.shape == (5, 5)


def test_sync_cache_from_worker_data_fallback():
    """Verify fallback to get_worker in _sync_cache_from_worker_data."""
    with patch("dask.distributed.get_worker") as mock_get_worker:
        mock_worker = MagicMock()
        mock_worker.data = {"f_key": "val"}
        mock_get_worker.return_value = mock_worker

        _sync_cache_from_worker_data("f_key", "c_key")
        assert _WORKER_CACHE["c_key"] == "val"


def test_regrid_dataset_non_spatial_preservation():
    """Verify preservation of non-spatial variables in _regrid_dataset."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    ds = xr.Dataset(
        data_vars={
            "spatial": (["lat", "lon"], np.random.rand(18, 36)),
            "time_var": (["time"], [1, 2, 3]),
        },
        coords={"lat": src.lat, "lon": src.lon, "time": [0, 1, 2]},
    )
    regridder = Regridder(src, tgt)
    res = regridder(ds)
    assert "time_var" in res.data_vars
    assert res.time_var.dims == ("time",)


def test_regrid_dataarray_aux_coords():
    """Verify regridding of auxiliary spatial coordinates."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        coords={
            "lat": src.lat,
            "lon": src.lon,
            "aux": (["lat", "lon"], np.random.rand(18, 36)),
        },
        name="test",
    )
    regridder = Regridder(src, tgt)
    res = regridder(da)
    assert "aux" in res.coords
    assert res.aux.shape == (36, 72)


def test_regrid_dataarray_dim_renaming():
    """Verify dimension renaming logic in _regrid_dataarray."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("y", "x"),
        name="test",
    )
    da.coords["y"] = (["y"], np.linspace(-90, 90, 18))
    da.coords["x"] = (["x"], np.linspace(0, 360, 36))
    da.y.attrs["standard_name"] = "latitude"
    da.x.attrs["standard_name"] = "longitude"

    regridder = Regridder(src, tgt)
    res = regridder(da)
    assert res.shape == (36, 72)


def test_quality_report_skip_heavy_remote():
    """Verify skip_heavy=True logic for remote weights in quality_report."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt, parallel=True, compute=False)

    class MockFuture:
        def __init__(self):
            self.key = "future_key"

    regridder._weights_matrix = MockFuture()
    report = regridder.quality_report(skip_heavy=True)
    assert report["n_weights"] == -1


def test_quality_report_nnz_remote_exception():
    """Verify exception handling when computing nnz remotely."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt, parallel=True, compute=False)

    class MockFuture:
        def __init__(self):
            self.key = "future_key"

    regridder._weights_matrix = MockFuture()
    regridder._dask_client = MagicMock()
    regridder._dask_client.submit.side_effect = ValueError("Mock Error")
    report = regridder.quality_report(skip_heavy=True)
    assert report["n_weights"] == -1


def test_regrid_dataset_grid_mapping_removal():
    """Verify removal of stale grid_mapping in _regrid_dataset."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    ds = xr.Dataset(
        {"var": (["lat", "lon"], np.random.rand(18, 36))}, coords=src.coords
    )
    ds.attrs["grid_mapping"] = "crs"
    ds.coords["crs"] = ([], 0, {"grid_mapping_name": "latitude_longitude"})
    regridder = Regridder(src, tgt)
    res = regridder(ds)
    assert "grid_mapping" not in res.attrs


def test_regrid_dataset_ugrid_attr_removal():
    """Verify removal of UGRID attributes when target is not UGRID."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    # Filter coords to match dimensions
    da_coords = {
        k: v for k, v in src.coords.items() if set(v.dims).issubset({"lat", "lon"})
    }
    da = xr.DataArray(
        np.random.rand(18, 36), dims=("lat", "lon"), coords=da_coords, name="var"
    )
    da.attrs["mesh"] = "some_mesh"
    da.attrs["location"] = "face"
    regridder = Regridder(src, tgt)
    res = regridder(da)
    assert "mesh" not in res.attrs
    assert "location" not in res.attrs


def test_get_nnz_task():
    """Verify _get_nnz_task."""
    matrix = MagicMock()
    matrix.nnz = 42
    assert _get_nnz_task(matrix) == 42


def test_regridder_persist_non_parallel():
    """Verify persist() on non-parallel regridder."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt, parallel=False)
    res = regridder.persist()
    assert res is regridder


def test_regridder_validate_weights_errors(tmp_path):
    """Verify error paths in _validate_weights."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    weight_file = str(tmp_path / "weights_err.nc")

    Regridder(src, tgt, method="bilinear", filename=weight_file, reuse_weights=True)

    with pytest.raises(ValueError, match="does not match loaded weights periodic"):
        Regridder.from_weights(weight_file, src, tgt, periodic=True)

    with pytest.raises(ValueError, match="does not match loaded weights skipna"):
        Regridder.from_weights(weight_file, src, tgt, skipna=True)

    with pytest.raises(ValueError, match="does not match loaded weights na_thres"):
        Regridder.from_weights(weight_file, src, tgt, skipna=False, na_thres=0.5)


def test_plot_static_unstructured_fallback():
    """Verify fallback logic in plot_static for unstructured grids."""
    from xregrid.viz import plot_static

    da = xr.DataArray([1.0, 2.0], dims=("cell",), name="test")
    # No lat/lon coordinates or attributes
    with patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())):
        with patch("xregrid.utils.get_crs_info", return_value=None):
            # This should hit the unstructured fallback in viz.py
            ax = plot_static(da)
            assert ax is not None


def test_plot_diagnostics_invalid_mode():
    """Verify ValueError for invalid mode in plot_diagnostics."""
    src = create_global_grid(10, 10)
    tgt = create_global_grid(5, 5)
    regridder = Regridder(src, tgt)
    with pytest.raises(ValueError, match="Unknown plotting mode"):
        regridder.plot_diagnostics(mode="invalid")
