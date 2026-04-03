import numpy as np
import xarray as xr

from unittest.mock import patch

import xregrid.viz
from xregrid.viz import plot_interactive

xregrid.viz.hvplot = True  # Force enable for testing


def test_plot_interactive_smart_crs():
    """
    Verify that plot_interactive discovers CRS and sets geo=True.
    """
    da = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, 10),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, 20),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
        name="test_data",
    )
    # Add CRS info
    da.attrs["crs"] = "EPSG:4326"

    # Mock the hvplot accessor call
    with patch.object(xr.DataArray, "hvplot", create=True) as mock_hvplot:
        plot_interactive(da)
        # Check that geo=True was passed in kwargs
        args, kwargs = mock_hvplot.call_args
        assert kwargs.get("geo") is True
        assert kwargs.get("title") == "Interactive Map"


def test_plot_interactive_no_crs():
    """
    Verify that plot_interactive does not set geo=True if no CRS is found.
    """
    da = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"], name="test_data")

    with patch.object(xr.DataArray, "hvplot", create=True) as mock_hvplot:
        plot_interactive(da)
        args, kwargs = mock_hvplot.call_args
        assert "geo" not in kwargs
