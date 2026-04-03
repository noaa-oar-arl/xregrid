import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from xregrid.viz import plot_static, plot_interactive


def create_unstructured_da(lazy: bool = False) -> xr.DataArray:
    """
    Create a 1D unstructured DataArray for testing.

    Parameters
    ----------
    lazy : bool, default False
        Whether to chunk the DataArray to make it Dask-backed.

    Returns
    -------
    xr.DataArray
        The 1D unstructured DataArray.
    """
    n = 100
    lat = np.linspace(-90, 90, n)
    lon = np.linspace(0, 360, n)
    data = np.random.rand(n)

    da = xr.DataArray(
        data,
        dims=["ncol"],
        coords={
            "lat": (
                ["ncol"],
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                ["ncol"],
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        name="test_data",
    )

    if lazy:
        da = da.chunk({"ncol": 10})

    return da


@patch("matplotlib.pyplot.axes")
@patch("matplotlib.pyplot.gca")
@patch("xarray.plot.accessor.DataArrayPlotAccessor.scatter")
def test_plot_static_unstructured(
    mock_scatter: MagicMock, mock_gca: MagicMock, mock_axes: MagicMock
) -> None:
    """
    Verify plot_static uses scatter for 1D unstructured data.

    Following the Aero Protocol's "Double-Check" rule, this test verifies
    logic with both Eager (NumPy) and Lazy (Dask) data backends.

    Parameters
    ----------
    mock_scatter : MagicMock
        Mock for xarray's scatter plot accessor.
    mock_gca : MagicMock
        Mock for plt.gca().
    mock_axes : MagicMock
        Mock for plt.axes().
    """
    # 1. Eager (NumPy)
    da_eager = create_unstructured_da(lazy=False)

    plot_static(da_eager)

    # Check if scatter was called
    mock_scatter.assert_called()
    args, kwargs = mock_scatter.call_args
    assert kwargs["x"] == "lon"
    assert kwargs["y"] == "lat"

    # 2. Lazy (Dask)
    da_lazy = create_unstructured_da(lazy=True)

    plot_static(da_lazy)

    assert mock_scatter.call_count == 2
    args, kwargs = mock_scatter.call_args
    assert kwargs["x"] == "lon"
    assert kwargs["y"] == "lat"


@patch("xarray.DataArray.hvplot")
def test_plot_interactive_unstructured(mock_hvplot: MagicMock) -> None:
    """
    Verify plot_interactive uses kind='points' for 1D unstructured data.

    Parameters
    ----------
    mock_hvplot : MagicMock
        Mock for the hvplot accessor.
    """
    # 1. Eager
    da_eager = create_unstructured_da(lazy=False)
    plot_interactive(da_eager)

    mock_hvplot.assert_called_with(
        rasterize=True, title="Interactive Map", kind="points", x="lon", y="lat"
    )

    # 2. Lazy
    da_lazy = create_unstructured_da(lazy=True)
    plot_interactive(da_lazy)

    # Check if last call was for lazy data with correct parameters
    mock_hvplot.assert_called_with(
        rasterize=True, title="Interactive Map", kind="points", x="lon", y="lat"
    )


def test_find_coord_unstructured() -> None:
    """
    Verify coordinate discovery for unstructured data.
    """
    da = create_unstructured_da()
    from xregrid.utils import _find_coord

    lat_da = _find_coord(da, "latitude")
    lon_da = _find_coord(da, "longitude")

    assert lat_da.name == "lat"
    assert lon_da.name == "lon"
    assert lat_da.ndim == 1
    assert lon_da.ndim == 1
