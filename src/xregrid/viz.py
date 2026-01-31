from __future__ import annotations

import warnings
from typing import Any, Optional

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
    title: Optional[str] = None,
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

        # Try cf-xarray if available
        if crs_wkt is None:
            try:
                # Use cf-xarray to find the grid mapping variable
                gm_var = da.cf.get_grid_mapping()
                if gm_var is not None:
                    crs_wkt = gm_var.attrs.get("crs_wkt")
            except (AttributeError, KeyError, ImportError):
                pass

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

    # Handle axes and faceting
    is_faceted = "col" in kwargs or "row" in kwargs
    ax = kwargs.pop("ax", None)

    if ax is not None:
        if is_faceted:
            warnings.warn(
                "Providing an 'ax' with faceting ('col' or 'row') is not supported by xarray and will be ignored."
            )
            ax = None
        else:
            # Ensure the existing axes is a GeoAxes if we are using cartopy
            is_geoaxes = False
            try:
                import cartopy.mpl.geoaxes as geoaxes

                is_geoaxes = isinstance(ax, geoaxes.GeoAxes)
            except ImportError:
                is_geoaxes = hasattr(ax, "projection")

            if not is_geoaxes:
                warnings.warn(
                    "The provided axes does not appear to be a Cartopy GeoAxes. "
                    "Geospatial plotting may not work as expected. "
                    "Ensure your axes was created with a projection (e.g., plt.axes(projection=...))."
                )

    if ax is None and not is_faceted:
        ax = plt.axes(projection=projection)

    # Enforce transform for geospatial accuracy (Aero Protocol)
    if "transform" not in kwargs:
        kwargs["transform"] = transform

    if is_faceted and "subplot_kws" not in kwargs:
        kwargs["subplot_kws"] = {"projection": projection}

    # Aero Protocol: No Ambiguous Plots.
    # Identify spatial and faceting dimensions to slice away everything else.

    # Identify spatial dimensions using cf-xarray for robust slicing
    try:
        # We look for dimensions associated with latitude and longitude
        lat_dims = da.cf["latitude"].dims
        lon_dims = da.cf["longitude"].dims
        spatial_dims = set(lat_dims) | set(lon_dims)
    except (KeyError, AttributeError, ImportError):
        # Fallback to assuming the last two dimensions are spatial
        spatial_dims = set(da.dims[-2:])

    # Identify dimensions used for faceting
    facet_dims = {kwargs.get("col"), kwargs.get("row")} - {None}

    # Dimensions that are neither spatial nor used for faceting
    extra_dims = [d for d in da.dims if d not in spatial_dims and d not in facet_dims]

    if extra_dims:
        first_slice = {d: 0 for d in extra_dims}
        warnings.warn(
            f"DataArray has {da.ndim} dimensions, but only 2 spatial dimensions "
            f"(plus optional faceting) are supported for static plots. "
            f"Automatically selecting the first slice along {extra_dims}: {first_slice}. "
            "To plot other slices, subset your data before calling plot_static."
        )
        da = da.isel(first_slice)

    im = da.plot(ax=ax, **kwargs)

    if is_faceted:
        # im is a FacetGrid
        if hasattr(im, "axes"):
            for a in im.axes.flat:
                if hasattr(a, "coastlines"):
                    a.coastlines()
        if title:
            plt.suptitle(title, y=1.02)
    else:
        # im is a QuadMesh or similar
        if hasattr(ax, "coastlines"):
            ax.coastlines()

        if title is None:
            title = da.name if da.name else "Static Map"
        ax.set_title(title)

    return im


def plot(
    da: xr.DataArray,
    mode: str = "static",
    **kwargs: Any,
) -> Any:
    """
    Unified entry point for xregrid plotting following the Two-Track Rule.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to plot.
    mode : str, default 'static'
        The plotting mode: 'static' (Track A: Publication) or
        'interactive' (Track B: Exploration).
    **kwargs : Any
        Additional arguments passed to plot_static or plot_interactive.

    Returns
    -------
    Any
        The plot object (Matplotlib artist or HvPlot object).
    """
    if mode == "static":
        return plot_static(da, **kwargs)
    elif mode == "interactive":
        return plot_interactive(da, **kwargs)
    else:
        raise ValueError(
            f"Unknown plotting mode: '{mode}'. Must be 'static' or 'interactive'."
        )


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


def plot_comparison(
    da_src: xr.DataArray,
    da_tgt: xr.DataArray,
    projection: Any = None,
    transform: Any = None,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality comparison plot (Source, Target, Difference).

    Parameters
    ----------
    da_src : xr.DataArray
        The source DataArray.
    da_tgt : xr.DataArray
        The target (regridded) DataArray.
    projection : cartopy.crs.Projection, optional
        The projection for the axes.
    transform : cartopy.crs.Projection, optional
        The transform for the plot call.
    cmap : str, default 'viridis'
        Colormap for the data plots.
    diff_cmap : str, default 'RdBu_r'
        Colormap for the difference plot.
    title : str, optional
        Overall figure title.
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plot_comparison.")

    if projection is None and ccrs is not None:
        projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        subplot_kw={"projection": projection} if projection else None,
    )

    # 1. Source Plot
    plot_static(
        da_src,
        ax=axes[0],
        projection=projection,
        transform=transform,
        cmap=cmap,
        title="Source Grid",
        **kwargs,
    )

    # 2. Target Plot
    plot_static(
        da_tgt,
        ax=axes[1],
        projection=projection,
        transform=transform,
        cmap=cmap,
        title="Target Grid",
        **kwargs,
    )

    # 3. Difference Plot
    # Interpolate source to target grid for difference calculation
    try:
        da_src_interp = da_src.interp_like(da_tgt, method="linear")
        diff = da_tgt - da_src_interp
        plot_static(
            diff,
            ax=axes[2],
            projection=projection,
            transform=transform,
            cmap=diff_cmap,
            title="Difference (Tgt - Src_interp)",
            **kwargs,
        )
    except Exception as e:
        axes[2].text(
            0.5,
            0.5,
            f"Could not compute difference:\n{e}",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Difference")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig
