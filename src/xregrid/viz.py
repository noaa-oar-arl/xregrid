from __future__ import annotations

from typing import Any

import xarray as xr

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None

try:
    import hvplot.xarray  # noqa: F401
except ImportError:
    hvplot = None
else:
    hvplot = True


def plot_static(
    da: xr.DataArray,
    projection: Any = None,
    transform: Any = None,
    title: str = "Static Map",
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality static plot using Matplotlib and Cartopy.

    Parameters
    ----------
    da : xr.DataArray
        The 2D DataArray to plot.
    projection : cartopy.crs.Projection, optional
        The projection to use for the axes. Defaults to ccrs.PlateCarree() if cartopy is available.
    transform : cartopy.crs.Projection, optional
        The transform to use for the plot call. Defaults to ccrs.PlateCarree() if cartopy is available.
    title : str, default 'Static Map'
        The plot title.
    **kwargs : Any
        Additional arguments passed to da.plot().

    Returns
    -------
    matplotlib.collections.QuadMesh or similar
        The plot object.
    """
    if plt is None:
        raise ImportError(
            "Matplotlib is required for plot_static. "
            "Install it with `pip install matplotlib`."
        )

    if ccrs is None:
        # Fallback to standard matplotlib if cartopy is missing
        ax = plt.gca()
        im = da.plot(ax=ax, **kwargs)
        ax.set_title(title)
        return im

    if projection is None:
        projection = ccrs.PlateCarree()
    if transform is None:
        transform = ccrs.PlateCarree()

    plt.gcf()
    ax = plt.subplot(1, 1, 1, projection=projection)

    im = da.plot(ax=ax, transform=transform, **kwargs)
    ax.coastlines()
    ax.set_title(title)

    return im


def plot_interactive(
    da: xr.DataArray,
    rasterize: bool = True,
    title: str = "Interactive Map",
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive plot using HvPlot.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to plot.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets (Aero Protocol requirement).
    title : str, default 'Interactive Map'
        The plot title.
    **kwargs : Any
        Additional arguments passed to da.hvplot().

    Returns
    -------
    hvplot.Interactive
        The interactive plot object.
    """
    if not hvplot:
        raise ImportError(
            "HvPlot is required for plot_interactive. "
            "Install it with `pip install hvplot`."
        )
    return da.hvplot(rasterize=rasterize, title=title, **kwargs)
