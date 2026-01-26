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
    import pyproj
except ImportError:
    pyproj = None

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

    if transform is None and ccrs is not None:
        # Try to detect CRS from attributes (Aero Protocol)
        crs_wkt = da.attrs.get("crs") or da.attrs.get("grid_mapping")
        # Check encoding as well
        if crs_wkt is None:
            crs_wkt = da.encoding.get("crs") or da.encoding.get("grid_mapping")

        if crs_wkt and pyproj is not None:
            try:
                # Use pyproj to identify the CRS
                proj_crs = pyproj.CRS(crs_wkt)

                # Try to find a matching Cartopy projection
                if proj_crs.is_geographic:
                    transform = ccrs.PlateCarree()
                elif proj_crs.is_projected:
                    # Attempt UTM detection
                    if proj_crs.utm_zone:
                        transform = ccrs.UTM(
                            zone=int(proj_crs.utm_zone[:-1]),
                            southern_hemisphere="S" in proj_crs.utm_zone,
                        )
                    # Generic fallback for other projected CRS if cartopy supports it
                    # (Simplified for this implementation)
            except Exception:
                pass

    if projection is None:
        projection = ccrs.PlateCarree()
    if transform is None:
        transform = ccrs.PlateCarree()

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        # Ensure the existing axes is a GeoAxes if we are using cartopy
        is_geoaxes = False
        try:
            import cartopy.mpl.geoaxes as geoaxes

            is_geoaxes = isinstance(ax, geoaxes.GeoAxes)
        except ImportError:
            is_geoaxes = hasattr(ax, "projection")

        if not is_geoaxes:
            import warnings

            warnings.warn(
                "The provided axes does not appear to be a Cartopy GeoAxes. "
                "Geospatial plotting may not work as expected. "
                "Ensure your axes was created with a projection (e.g., plt.axes(projection=...))."
            )
    else:
        ax = plt.axes(projection=projection)

    # Enforce transform for geospatial accuracy (Aero Protocol)
    if "transform" not in kwargs:
        kwargs["transform"] = transform

    im = da.plot(ax=ax, **kwargs)

    if hasattr(ax, "coastlines"):
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
