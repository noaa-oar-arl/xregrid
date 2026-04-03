import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder


def test_unstructured_bilinear_mesh_enhanced():
    """
    Aero Protocol: Verify unstructured bilinear regridding using Mesh (NODE-based).
    Double-Check Test: NumPy vs Dask backends.
    """
    # Create a simple triangle mesh (4 nodes, 2 triangles)
    # 3 -- 2
    # |  / |
    # 0 -- 1
    lon = xr.DataArray(
        [0.0, 1.0, 1.0, 0.0], dims="n_pts", name="lon", attrs={"units": "degrees_east"}
    )
    lat = xr.DataArray(
        [0.0, 0.0, 1.0, 1.0], dims="n_pts", name="lat", attrs={"units": "degrees_north"}
    )

    # Face-node connectivity (0-based for UGRID)
    # Tri 1: 0, 1, 2
    # Tri 2: 0, 2, 3
    conn = xr.DataArray(
        [[0, 1, 2], [0, 2, 3]],
        dims=("n_face", "n_vertex"),
        name="face_node_connectivity",
        attrs={"cf_role": "face_node_connectivity", "start_index": 0},
    )

    src_grid = xr.Dataset(
        coords={"lon_node": lon, "lat_node": lat},
        data_vars={"face_node_connectivity": conn},
    )

    # Target grid: a single point in the middle
    # We use a 1D target to keep it simple
    tgt_lon = xr.DataArray([0.5], dims="n_dst", name="lon")
    tgt_lat = xr.DataArray([0.5], dims="n_dst", name="lat")
    tgt_grid = xr.Dataset(coords={"lon": tgt_lon, "lat": tgt_lat})

    # Data on nodes
    data_val = np.array([10.0, 20.0, 30.0, 40.0])
    da_eager = xr.DataArray(
        data_val,
        dims="n_pts",
        coords={"lon_node": lon, "lat_node": lat},
        name="test_data",
    )

    # Initialize Regridder with bilinear
    # Ensure mesh variable is identified correctly
    src_grid.face_node_connectivity.attrs["mesh"] = "mesh"
    src_grid["mesh"] = (
        [],
        0,
        {
            "cf_role": "mesh_topology",
            "face_node_connectivity": "face_node_connectivity",
            "node_coordinates": "lon_node lat_node",
        },
    )

    regridder = Regridder(src_grid, tgt_grid, method="bilinear")

    # 1. Test Eager (NumPy)
    res_eager = regridder(da_eager)

    # 2. Test Lazy (Dask)
    da_lazy = da_eager.chunk({"n_pts": 2})
    res_lazy = regridder(da_lazy)

    # Assertions
    assert isinstance(res_eager.data, np.ndarray)
    assert hasattr(res_lazy.data, "dask")

    # Results should be identical
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verification of shape
    assert res_eager.shape == (1,)

    # Verify provenance
    assert "method=bilinear" in res_eager.attrs["history"]
    assert "ESMF" in res_eager.attrs["history"]


def test_unstructured_patch_mesh_enhanced():
    """Verify that 'patch' method also works on unstructured meshes."""
    lon = xr.DataArray([0.0, 1.0, 1.0, 0.0], dims="n_pts", name="lon")
    lat = xr.DataArray([0.0, 0.0, 1.0, 1.0], dims="n_pts", name="lat")
    conn = xr.DataArray(
        [[0, 1, 2], [0, 2, 3]],
        dims=("n_face", "n_vertex"),
        name="face_node_connectivity",
        attrs={"cf_role": "face_node_connectivity", "start_index": 0},
    )
    src_grid = xr.Dataset(
        coords={"lon_node": lon, "lat_node": lat},
        data_vars={"face_node_connectivity": conn},
    )

    tgt_lon = xr.DataArray([0.5], dims="n_dst", name="lon")
    tgt_lat = xr.DataArray([0.5], dims="n_dst", name="lat")
    tgt_grid = xr.Dataset(coords={"lon": tgt_lon, "lat": tgt_lat})

    # Ensure mesh variable is identified correctly
    src_grid.face_node_connectivity.attrs["mesh"] = "mesh"
    src_grid["mesh"] = (
        [],
        0,
        {
            "cf_role": "mesh_topology",
            "face_node_connectivity": "face_node_connectivity",
            "node_coordinates": "lon_node lat_node",
        },
    )
    regridder = Regridder(src_grid, tgt_grid, method="patch")
    da = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0], dims="n_pts", coords={"lon_node": lon, "lat_node": lat}
    )

    res = regridder(da)
    assert res.shape == (1,)
    assert "method=patch" in res.attrs["history"]


if __name__ == "__main__":
    pytest.main([__file__])
