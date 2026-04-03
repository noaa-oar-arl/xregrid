import numpy as np
import pytest
import xarray as xr
from xregrid import Regridder
from xregrid.utils import create_global_grid


def test_musica_cesm_regrid_aero():
    """Verify MUSICA/CESM (ncol) grid discovery and regridding (Aero Protocol)."""
    # 1. Create MUSICA-like source grid (unstructured ncol)
    n_col = 100
    ds_src = xr.Dataset(
        data_vars={"temp": (["ncol"], np.random.rand(n_col))},
        coords={
            "lat": (["ncol"], np.linspace(-90, 90, n_col)),
            "lon": (["ncol"], np.linspace(0, 350, n_col)),
        },
    )

    # 2. Create target rectilinear grid
    ds_tgt = create_global_grid(10, 20)

    # 3. Initialize Regridder (should detect unstructured ncol)
    # Using nearest_s2d because ncol has no connectivity info
    regridder = Regridder(ds_src, ds_tgt, method="nearest_s2d")
    assert regridder._is_unstructured_src
    assert regridder._dims_source == ("ncol",)

    # 4. Double-Check: Eager
    out_eager = regridder(ds_src)
    assert isinstance(out_eager, xr.Dataset)
    assert "temp" in out_eager
    assert out_eager.temp.dims == ("lat", "lon")

    # 5. Double-Check: Lazy
    ds_src_lazy = ds_src.chunk({"ncol": 10})
    out_lazy = regridder(ds_src_lazy)
    assert hasattr(out_lazy.temp.data, "dask")

    xr.testing.assert_allclose(out_eager, out_lazy.compute())


def test_mpas_regrid_aero():
    """Verify MPAS (nCells) grid discovery and regridding (Aero Protocol)."""
    n_cells = 100
    ds_src = xr.Dataset(
        data_vars={"temp": (["nCells"], np.random.rand(n_cells))},
        coords={
            "latCell": (["nCells"], np.linspace(-90, 90, n_cells)),
            "lonCell": (["nCells"], np.linspace(0, 350, n_cells)),
        },
    )
    # CF-Xarray might not know latCell/lonCell without attributes,
    # but xregrid fallback should find them.
    ds_src.latCell.attrs["standard_name"] = "latitude"
    ds_src.lonCell.attrs["standard_name"] = "longitude"

    ds_tgt = create_global_grid(10, 20)

    # Using nearest_s2d because nCells has no connectivity info in this test
    regridder = Regridder(ds_src, ds_tgt, method="nearest_s2d")
    assert regridder._is_unstructured_src
    assert regridder._dims_source == ("nCells",)

    out_eager = regridder(ds_src)
    assert out_eager.temp.dims == ("lat", "lon")

    ds_src_lazy = ds_src.chunk({"nCells": 10})
    out_lazy = regridder(ds_src_lazy)
    xr.testing.assert_allclose(out_eager, out_lazy.compute())


def test_ugrid_discovery_aero():
    """Verify UGRID-compliant discovery with explicit mesh topology."""
    n_nodes = 50
    ds = xr.Dataset(
        data_vars={
            "temp": (["node"], np.random.rand(n_nodes)),
            "mesh": (
                [],
                0,
                {"cf_role": "mesh_topology", "node_coordinates": "lon_node lat_node"},
            ),
        },
        coords={
            "lat_node": (
                ["node"],
                np.linspace(-90, 90, n_nodes),
                {"standard_name": "latitude"},
            ),
            "lon_node": (
                ["node"],
                np.linspace(0, 360, n_nodes),
                {"standard_name": "longitude"},
            ),
        },
    )
    # The new logic should prefer lat_node/lon_node because they are linked in 'mesh'
    # OR because they have standard names and 'node' in name.

    ds_tgt = create_global_grid(10, 20)
    # Using nearest_s2d because this UGRID has no connectivity info
    regridder = Regridder(ds, ds_tgt, method="nearest_s2d")
    assert regridder._is_unstructured_src
    assert regridder._dims_source == ("node",)

    out = regridder(ds)
    assert out.temp.dims == ("lat", "lon")


def test_scrip_conservative_regrid_aero():
    """Verify SCRIP-style unstructured grid handles conservative regridding via derived connectivity."""
    n_cells = 10
    # Create SCRIP-like 2D bounds (n_cells, 4 corners)
    lat_b = np.array(
        [
            [-10, -10, 10, 10],
            [-10, -10, 10, 10],
            # ... just a few for testing
        ]
    )
    lat_b = np.repeat(lat_b, n_cells // 2, axis=0)
    lon_b = np.array(
        [
            [0, 10, 10, 0],
            [10, 20, 20, 10],
        ]
    )
    lon_b = np.repeat(lon_b, n_cells // 2, axis=0)

    ds_src = xr.Dataset(
        data_vars={"temp": (["grid_size"], np.random.rand(n_cells))},
        coords={
            "lat": (["grid_size"], np.zeros(n_cells)),
            "lon": (["grid_size"], np.zeros(n_cells)),
            "lat_b": (["grid_size", "nv"], lat_b),
            "lon_b": (["grid_size", "nv"], lon_b),
        },
    )

    ds_tgt = create_global_grid(10, 10)

    # conservative requires bounds
    regridder = Regridder(ds_src, ds_tgt, method="conservative")
    assert regridder._is_unstructured_src

    out = regridder(ds_src)
    assert "temp" in out


def test_mpas_non_conservative_discovery_aero():
    """Verify MPAS (nCells) non-conservative discovery (triggers optimized path)."""
    n_cells = 50
    ds_src = xr.Dataset(
        data_vars={"temp": (["nCells"], np.random.rand(n_cells))},
        coords={
            "lat": (["nCells"], np.linspace(-90, 90, n_cells)),
            "lon": (["nCells"], np.linspace(0, 350, n_cells)),
        },
    )
    ds_tgt = create_global_grid(10, 20)

    # This should trigger the optimized section 2 in _get_unstructured_mesh_info
    regridder = Regridder(ds_src, ds_tgt, method="nearest_s2d")
    assert regridder._is_unstructured_src
    assert regridder._dims_source == ("nCells",)

    out = regridder(ds_src)
    assert out.temp.dims == ("lat", "lon")


def test_mpas_to_scrip_regrid_aero():
    """Verify MPAS to SCRIP conversion and native regridding."""
    from xregrid.utils import mpas_to_scrip

    n_cells = 4
    # Create a minimal valid MPAS-like grid
    ds_mpas = xr.Dataset(
        data_vars={
            "temp": (["nCells"], np.random.rand(n_cells)),
            "verticesOnCell": (
                ["nCells", "maxEdges"],
                np.array([[1, 2, 3], [1, 3, 4], [1, 2, 4], [2, 3, 4]]),
            ),
            "nEdgesOnCell": (["nCells"], [3, 3, 3, 3]),
        },
        coords={
            "latCell": (["nCells"], np.linspace(-45, 45, n_cells)),
            "lonCell": (["nCells"], np.linspace(0, 90, n_cells)),
            "latVertex": (["nVertices"], np.linspace(-90, 90, 5)),
            "lonVertex": (["nVertices"], np.linspace(0, 360, 5)),
        },
    )

    # 1. Convert
    ds_scrip = mpas_to_scrip(ds_mpas)
    assert "lat_b" in ds_scrip.coords
    assert ds_scrip.lat.dims == ("grid_size",)

    # 2. Regrid
    ds_tgt = create_global_grid(10, 20)
    regridder = Regridder(ds_scrip, ds_tgt, method="bilinear")
    assert regridder._is_unstructured_src

    # Data to regrid must match the new grid_size dimension if using the scrip grid as source
    # Only use coordinates that are compatible with the 1D grid_size dimension
    compatible_coords = {
        c: ds_scrip.coords[c]
        for c in ds_scrip.coords
        if set(ds_scrip.coords[c].dims).issubset({"grid_size"})
    }
    da_src = xr.DataArray(
        np.random.rand(len(ds_scrip.grid_size)),
        dims=["grid_size"],
        coords=compatible_coords,
    )
    out = regridder(da_src)
    assert out.dims == ("lat", "lon")


if __name__ == "__main__":
    pytest.main([__file__])
