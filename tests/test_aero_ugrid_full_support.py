import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder, create_global_grid


def create_mock_ugrid(n_nodes=16, n_faces=9):
    """Create a mock UGRID dataset using a regular quad mesh."""
    # Ensure n_nodes is a square for simplicity in this mock
    n_side = int(np.sqrt(n_nodes))
    if n_side**2 != n_nodes:
        # Fallback to next square
        n_side = int(np.ceil(np.sqrt(n_nodes)))
        n_nodes = n_side**2

    x = np.linspace(0, 350, n_side)
    y = np.linspace(-80, 80, n_side)
    node_x_2d, node_y_2d = np.meshgrid(x, y)
    node_x = node_x_2d.flatten()
    node_y = node_y_2d.flatten()

    # Create quads
    face_nodes = []
    face_x = []
    face_y = []
    for j in range(n_side - 1):
        for i in range(n_side - 1):
            n0 = j * n_side + i
            n1 = j * n_side + i + 1
            n2 = (j + 1) * n_side + i + 1
            n3 = (j + 1) * n_side + i
            face_nodes.append([n0, n1, n2, n3])
            face_x.append((node_x[n0] + node_x[n2]) / 2)
            face_y.append((node_y[n0] + node_y[n2]) / 2)

    face_nodes = np.array(face_nodes)
    n_faces = len(face_nodes)
    face_x = np.array(face_x)
    face_y = np.array(face_y)

    ds = xr.Dataset(
        data_vars={
            "mesh_topology": (
                [],
                0,
                {
                    "cf_role": "mesh_topology",
                    "topology_dimension": 2,
                    "node_coordinates": "node_lon node_lat",
                    "face_node_connectivity": "face_nodes",
                    "face_coordinates": "face_lon face_lat",
                },
            ),
            "face_nodes": (
                ["n_face", "n_node_per_face"],
                face_nodes,
                {"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "temp": (
                ["n_face"],
                np.random.rand(n_faces),
                {
                    "mesh": "mesh_topology",
                    "location": "face",
                    "standard_name": "air_temperature",
                },
            ),
        },
        coords={
            "node_lon": (
                ["n_node"],
                node_x,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "node_lat": (
                ["n_node"],
                node_y,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "face_lon": (
                ["n_face"],
                face_x,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "face_lat": (
                ["n_face"],
                face_y,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
    )
    return ds


def test_ugrid_discovery_and_regrid():
    """Verify UGRID discovery and regridding (Eager and Lazy)."""
    src_ds = create_mock_ugrid(n_nodes=25, n_faces=16)
    tgt_grid = create_global_grid(30, 30)

    # Test that Regridder can handle the UGRID dataset
    # We use conservative regridding to test triangulation logic
    regridder = Regridder(src_ds, tgt_grid, method="conservative")

    # 1. Eager test
    res_eager = regridder(src_ds.temp)

    assert "lat" in res_eager.coords
    assert "lon" in res_eager.coords
    # Verify metadata removal as target is not UGRID
    assert "mesh" not in res_eager.attrs

    # 2. Lazy test
    da_lazy = src_ds.temp.chunk({"n_face": 5})
    res_lazy = regridder(da_lazy).compute()

    xr.testing.assert_allclose(res_eager, res_lazy)


def test_ugrid_scientific_hygiene():
    """Verify UGRID metadata propagation to UGRID target."""
    src_ds = create_mock_ugrid(n_nodes=16, n_faces=9)
    tgt_ds = create_mock_ugrid(n_nodes=25, n_faces=16)

    # For simplicity, use nearest_s2d which doesn't require complex connectivity for target
    regridder = Regridder(src_ds, tgt_ds, method="nearest_s2d")

    res = regridder(src_ds.temp)

    # Scientific Hygiene: target mesh should be attached
    assert res.attrs["mesh"] == "mesh_topology"
    assert "location" in res.attrs
    assert "mesh_topology" in res.coords or "mesh_topology" in res.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
