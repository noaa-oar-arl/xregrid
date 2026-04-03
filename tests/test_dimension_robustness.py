import numpy as np
import xarray as xr
from xregrid import Regridder


def test_regridder_time_dimension_detection():
    # Setup source and target grids with time
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat": (
                ["time", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    # This should now work without failing during weight generation
    regridder = Regridder(src_ds, tgt_ds, method="bilinear")

    # Create data with time and vertical dimensions
    levs = np.arange(5)
    data = np.random.rand(len(times), len(levs), len(lats), len(lons))
    da = xr.DataArray(
        data,
        coords={
            "time": (["time"], times),
            "lev": (["lev"], levs),
            "lat": (["time", "lat"], np.broadcast_to(lats, (1, 10))),
            "lon": (["lon"], lons),
        },
        dims=("time", "lev", "lat", "lon"),
        name="temp",
    )

    # Regrid DataArray
    res_da = regridder(da)

    # Check that time and lev are preserved
    assert "time" in res_da.dims
    assert "lev" in res_da.dims
    assert res_da.shape == (1, 5, 5, 10)

    # Regrid Dataset
    ds = xr.Dataset({"temp": da, "time_var": (["time"], times)})
    res_ds = regridder(ds)

    assert "time" in res_ds.dims
    assert "temp" in res_ds.data_vars
    assert "time_var" in res_ds.data_vars
    assert res_ds["temp"].shape == (1, 5, 5, 10)
    assert res_ds["time_var"].dims == ("time",)


def test_regridder_dtype_time_fallback():
    # Setup with time-like dtype but non-standard name
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    times = [np.datetime64("2020-01-01")]

    src_ds = xr.Dataset(
        coords={
            "mytime": (["mytime"], times),  # Non-standard name, no CF attributes
            "lat": (
                ["mytime", "lat"],
                np.broadcast_to(lats, (1, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # Verify mytime was detected as non-spatial
    assert "mytime" not in regridder._dims_source

    # Test DataArray regridding with this non-standard time dim
    da = xr.DataArray(
        np.random.rand(1, 10, 20), coords=src_ds.coords, dims=("mytime", "lat", "lon")
    )

    res = regridder(da)
    assert "mytime" in res.dims
    assert res.shape == (1, 5, 10)


def test_non_regriddable_object():
    # Test passing something that shouldn't be regridded
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)

    src_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                lats,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )
    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)

    # A DataArray that only has one dimension (time)
    time_da = xr.DataArray([1, 2, 3], dims="time", name="time_var")

    # Should return unchanged
    res = regridder(time_da)
    xr.testing.assert_identical(res, time_da)


def test_regridder_vertical_dimension_detection():
    # Setup source with vertical dimension in lats
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 360, 20)
    levs = np.arange(3)

    src_ds = xr.Dataset(
        coords={
            "lev": (["lev"], levs, {"standard_name": "altitude"}),
            "lat": (
                ["lev", "lat"],
                np.broadcast_to(lats, (3, 10)),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                lons,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    tgt_ds = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(src_ds, tgt_ds)
    assert "lev" not in regridder._dims_source

    da = xr.DataArray(
        np.random.rand(3, 10, 20), coords=src_ds.coords, dims=("lev", "lat", "lon")
    )

    res = regridder(da)
    assert "lev" in res.dims
    assert res.shape == (3, 5, 10)


def test_regridder_ugrid_with_time():
    # Setup mocked uxarray object with time dimension
    from unittest.mock import MagicMock

    class UxDatasetMock(xr.Dataset):
        def __init__(self, ds, uxgrid):
            super().__init__(ds.data_vars, coords=ds.coords, attrs=ds.attrs)
            self.uxgrid = uxgrid

    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    mock_uxgrid = MagicMock()
    mock_uxgrid.node_lat = xr.DataArray(np.linspace(-90, 90, n_node), dims=["n_node"])
    mock_uxgrid.node_lon = xr.DataArray(np.linspace(0, 360, n_node), dims=["n_node"])
    # face coords with time dimension (moving mesh case, though xregrid assumes static)
    mock_uxgrid.face_lat = xr.DataArray(
        np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
        dims=["time", "n_face"],
        coords={"time": times},
    )
    mock_uxgrid.face_lon = xr.DataArray(
        np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
        dims=["time", "n_face"],
        coords={"time": times},
    )

    # Create connectivity
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, i + 1, (i + 2) % n_node]

    mock_uxgrid.face_node_connectivity = xr.DataArray(
        conn, dims=["n_face", "n_max_face_nodes"]
    )
    mock_uxgrid.face_node_connectivity.attrs["start_index"] = 0
    mock_uxgrid.face_node_connectivity.attrs["_FillValue"] = -1

    # Mock UxDataset with time-varying variable
    ds_base = xr.Dataset(
        {"test_var": (["time", "n_face"], np.random.rand(1, n_face))},
        coords={"time": (["time"], times, {"standard_name": "time"})},
    )
    ds = UxDatasetMock(ds_base, mock_uxgrid)

    target_grid = xr.Dataset(
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 5),
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 10),
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        }
    )

    regridder = Regridder(ds, target_grid, method="nearest_s2d")

    # Verify time was detected as non-spatial
    assert "time" not in regridder._dims_source
    assert regridder._dims_source == ("n_face",)

    # Regrid DataArray
    res = regridder(ds["test_var"])

    assert "time" in res.dims
    assert res.shape == (1, 5, 10)


def test_regridder_raw_ugrid_with_time():
    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    # Create a raw dataset following UGRID convention
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, (i + 1) % n_node, (i + 2) % n_node]

    ds = xr.Dataset(
        data_vars={
            "temp": (["time", "n_face"], np.random.rand(1, n_face)),
            "face_node_connectivity": (["n_face", "n_max_face_nodes"], conn),
        },
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
                {"units": "degrees_north"},
            ),
            "lon_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
                {"units": "degrees_east"},
            ),
            "lat_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(-90, 90, n_node), (1, n_node)),
                {"units": "degrees_north"},
            ),
            "lon_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(0, 360, n_node), (1, n_node)),
                {"units": "degrees_east"},
            ),
        },
    )

    ds.face_node_connectivity.attrs["cf_role"] = "face_node_connectivity"
    ds.face_node_connectivity.attrs["start_index"] = 0

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    regridder = Regridder(ds, target_grid, method="nearest_s2d")

    assert "time" not in regridder._dims_source
    # Since it is UGRID, it should have detected n_face as the spatial dimension for variables
    assert "n_face" in regridder._dims_source

    res = regridder(ds["temp"])
    assert "time" in res.dims
    assert res.shape == (1, 18, 36)


def test_regridder_user_specific_structure():
    # Mimic user's dataset structure: (time, node)
    # node is string coordinate, lat/lon are (node)
    n_node = 10
    n_time = 5
    times = np.arange(n_time).astype("datetime64[D]")
    nodes = np.array([f"NODE_{i}" for i in range(n_node)], dtype="<U19")

    src_ds = xr.Dataset(
        data_vars={
            "day_of_year": (
                ["time", "node"],
                np.random.rand(n_time, n_node).astype("float32"),
            ),
            "aod_550nm": (
                ["time", "node"],
                np.random.rand(n_time, n_node).astype("float32"),
            ),
            "mesh": ([], np.int32(1)),
        },
        coords={
            "time": (["time"], times),
            "node": (["node"], nodes),
            "latitude": (["node"], np.linspace(-90, 90, n_node)),
            "longitude": (["node"], np.linspace(0, 360, n_node)),
        },
    )

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    # Use bilinear (nearest_s2d for mock compatibility)
    regridder = Regridder(src_ds, target_grid, method="nearest_s2d")

    assert "time" not in regridder._dims_source
    assert regridder._dims_source == ("node",)

    # Regrid a variable
    res = regridder(src_ds["aod_550nm"])

    assert "time" in res.dims
    assert res.shape == (n_time, 18, 36)
    assert res.dtype == "float32"

    # Regrid the whole dataset
    res_ds = regridder(src_ds)
    assert "time" in res_ds.dims
    assert "aod_550nm" in res_ds.data_vars
    assert res_ds["aod_550nm"].shape == (n_time, 18, 36)
    assert "node" not in res_ds["aod_550nm"].dims  # Space dimension should be replaced
    assert "mesh" in res_ds.data_vars  # Non-spatial data var should be preserved


def test_regridder_raw_ugrid_conservative_with_time():
    n_face = 10
    n_node = 12
    times = [np.datetime64("2020-01-01")]

    # Create a raw dataset following UGRID convention for conservative regridding
    # Conservative needs faces and nodes (for connectivity)
    conn = np.zeros((n_face, 3), dtype=int)
    for i in range(n_face):
        conn[i] = [i, (i + 1) % n_node, (i + 2) % n_node]

    ds = xr.Dataset(
        data_vars={
            "temp": (["time", "n_face"], np.random.rand(1, n_face)),
            "face_node_connectivity": (["n_face", "n_max_face_nodes"], conn),
        },
        coords={
            "time": (["time"], times, {"standard_name": "time"}),
            "lat_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(-90, 90, n_node), (1, n_node)),
                {"units": "degrees_north"},
            ),
            "lon_node": (
                ["time", "n_node"],
                np.broadcast_to(np.linspace(0, 360, n_node), (1, n_node)),
                {"units": "degrees_east"},
            ),
            "lat_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(-90, 90, n_face), (1, n_face)),
                {"units": "degrees_north"},
            ),
            "lon_face": (
                ["time", "n_face"],
                np.broadcast_to(np.linspace(0, 360, n_face), (1, n_face)),
                {"units": "degrees_east"},
            ),
        },
    )

    ds.face_node_connectivity.attrs["cf_role"] = "face_node_connectivity"
    ds.face_node_connectivity.attrs["start_index"] = 0

    from xregrid import create_global_grid

    target_grid = create_global_grid(10, 10)

    # This should trigger _get_unstructured_mesh_info
    regridder = Regridder(ds, target_grid, method="conservative")

    assert "time" not in regridder._dims_source
    assert "n_face" in regridder._dims_source

    res = regridder(ds["temp"])
    assert "time" in res.dims
    assert res.shape == (1, 18, 36)
