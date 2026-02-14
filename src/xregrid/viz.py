from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import xarray as xr

from xregrid.utils import get_crs_info

if TYPE_CHECKING:
    from xregrid.regridder import Regridder

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
    import holoviews as hv
except ImportError:
    hvplot = None
    hv = None
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
    title : str, optional
        The plot title.
    **kwargs : Any
        Additional arguments passed to da.plot().

    Returns
    -------
    Any
        The plot object (e.g., matplotlib QuadMesh or FacetGrid).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "Matplotlib is required for plot_static. "
            "Install it with `pip install matplotlib`."
        )

    # Handle axes and faceting early to avoid multiple 'ax' arguments (Aero Protocol: Robustness)
    ax = kwargs.pop("ax", None)
    is_faceted = "col" in kwargs or "row" in kwargs

    # Aero Protocol: No Ambiguous Plots.
    # Identify spatial and faceting dimensions to slice away everything else.
    # We do this early so it applies even if cartopy is missing.

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

    if ccrs is None:
        # Fallback to standard matplotlib if cartopy is missing
        if ax is None:
            ax = plt.gca()
        im = da.plot(ax=ax, **kwargs)
        if title:
            ax.set_title(title)
        return im

    if transform is None and ccrs is not None:
        proj_crs = get_crs_info(da)

        if proj_crs:
            try:
                # Map pyproj CRS to Cartopy projections
                if proj_crs.is_geographic:
                    transform = ccrs.PlateCarree()
                elif proj_crs.is_projected:
                    # Attempt robust projection detection
                    # UTM detection
                    if proj_crs.utm_zone:
                        transform = ccrs.UTM(
                            zone=int(proj_crs.utm_zone[:-1]),
                            southern_hemisphere="S" in proj_crs.utm_zone,
                        )
                    # Mercator
                    elif "merc" in proj_crs.to_dict().get("proj", ""):
                        transform = ccrs.Mercator()
                    # Lambert Conformal
                    elif "lcc" in proj_crs.to_dict().get("proj", ""):
                        transform = ccrs.LambertConformal(
                            central_longitude=proj_crs.to_dict().get("lon_0", 0.0),
                            central_latitude=proj_crs.to_dict().get("lat_0", 0.0),
                        )
            except Exception:
                pass

    if projection is None:
        projection = ccrs.PlateCarree()
    if transform is None:
        transform = ccrs.PlateCarree()

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
        # Strictly enforce projection in axes creation (Aero Protocol)
        if projection is None and ccrs is not None:
            projection = ccrs.PlateCarree()
        ax = plt.axes(projection=projection)

    # Enforce transform for geospatial accuracy (Aero Protocol)
    if transform is None and ccrs is not None:
        transform = ccrs.PlateCarree()

    if "transform" not in kwargs:
        kwargs["transform"] = transform

    if is_faceted and "subplot_kws" not in kwargs:
        kwargs["subplot_kws"] = {"projection": projection}

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

    Raises
    ------
    ValueError
        If an unknown plotting mode is provided.
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
    Any
        The interactive plot object (HvPlot/HoloViews).

    Raises
    ------
    ImportError
        If HvPlot is not installed.
    """
    if not hvplot:
        raise ImportError(
            "HvPlot is required for plot_interactive. "
            "Install it with `pip install hvplot`."
        )
    return da.hvplot(rasterize=rasterize, title=title, **kwargs)


def plot_diagnostics(
    regridder: "Regridder",
    projection: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Plot spatial diagnostics for a Regridder.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance to diagnose.
    projection : Any, optional
        The projection for the axes. Defaults to ccrs.PlateCarree() if available.
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plot_diagnostics.")

    # Aero Protocol: Automated projection discovery (No Ambiguous Plots)
    if projection is None and ccrs is not None:
        # Attempt to discover projection from target grid
        target_crs = get_crs_info(regridder.target_grid_ds)
        if target_crs:
            if target_crs.is_geographic:
                projection = ccrs.PlateCarree()
            elif target_crs.is_projected:
                # Basic mapping to common projections
                if target_crs.utm_zone:
                    projection = ccrs.UTM(
                        zone=int(target_crs.utm_zone[:-1]),
                        southern_hemisphere="S" in target_crs.utm_zone,
                    )
                elif "merc" in target_crs.to_dict().get("proj", ""):
                    projection = ccrs.Mercator()
                else:
                    projection = ccrs.PlateCarree()
        else:
            projection = ccrs.PlateCarree()

    ds_diag = regridder.diagnostics()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        subplot_kw={"projection": projection},
    )

    plot_static(
        ds_diag.weight_sum,
        ax=axes[0],
        title="Weight Sum",
        cmap="viridis",
        **kwargs,
    )

    plot_static(
        ds_diag.unmapped_mask,
        ax=axes[1],
        title="Unmapped Mask (1=Unmapped)",
        cmap="Reds",
        **kwargs,
    )

    fig.suptitle(f"Regridder Diagnostics ({regridder.method})", fontsize=16)
    plt.tight_layout()

    return fig


def plot_diagnostics_interactive(
    regridder: "Regridder",
    rasterize: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive diagnostic plot.

    Uses HvPlot and HoloViews to provide a side-by-side interactive view
    of weight_sum and unmapped_mask.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance to diagnose.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets (Aero Protocol requirement).
    title : str, optional
        Overall plot title.
    **kwargs : Any
        Additional arguments passed to hvplot calls.

    Returns
    -------
    Any
        The composed HoloViews object (Layout).

    Raises
    ------
    ImportError
        If HvPlot or HoloViews is not installed.
    """
    if not hvplot or hv is None:
        raise ImportError(
            "HvPlot and HoloViews are required for plot_diagnostics_interactive. "
            "Install them with `pip install hvplot holoviews`."
        )

    ds_diag = regridder.diagnostics()

    # 1. Weight Sum Plot
    p_sum = ds_diag.weight_sum.hvplot(
        rasterize=rasterize, cmap="viridis", title="Weight Sum", **kwargs
    )

    # 2. Unmapped Mask Plot
    p_mask = ds_diag.unmapped_mask.hvplot(
        rasterize=rasterize, cmap="Reds", title="Unmapped Mask (1=Unmapped)", **kwargs
    )

    layout = (p_sum + p_mask).cols(2)

    if title is None:
        title = f"Regridder Diagnostics ({regridder.method})"

    layout = layout.opts(title=title)

    return layout


def plot_comparison(
    da_src: xr.DataArray,
    da_tgt: xr.DataArray,
    regridder: Optional[Any] = None,
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
    regridder : Regridder, optional
        The regridder used to transform da_src to da_tgt.
        If provided, it will be used to calculate the difference plot correctly.
    projection : Any, optional
        The projection for the axes.
    transform : Any, optional
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

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plot_comparison.")

    if projection is None and ccrs is not None:
        projection = ccrs.PlateCarree()

    # Enforce projection on all subplots for comparison consistency (Aero Protocol)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        subplot_kw={"projection": projection},
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
    # Use Regridder if provided for exact difference, otherwise fallback to interp_like
    try:
        if regridder is not None:
            da_src_interp = regridder(da_src)
        else:
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


def plot_comparison_interactive(
    da_src: xr.DataArray,
    da_tgt: xr.DataArray,
    regridder: Optional[Any] = None,
    rasterize: bool = True,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive comparison plot (Source, Target, Difference).

    Uses HvPlot and HoloViews to provide a side-by-side interactive view.

    Parameters
    ----------
    da_src : xr.DataArray
        The source DataArray.
    da_tgt : xr.DataArray
        The target (regridded) DataArray.
    regridder : Regridder, optional
        The regridder used to transform da_src to da_tgt.
        If provided, it will be used to calculate the difference plot correctly.
    rasterize : bool, default True
        Whether to rasterize the grid for large datasets (Aero Protocol requirement).
    cmap : str, default 'viridis'
        Colormap for the data plots.
    diff_cmap : str, default 'RdBu_r'
        Colormap for the difference plot.
    title : str, optional
        Overall plot title.
    **kwargs : Any
        Additional arguments passed to hvplot calls.

    Returns
    -------
    Any
        The composed HoloViews object (Layout).

    Raises
    ------
    ImportError
        If HvPlot or HoloViews is not installed.
    """
    if not hvplot or hv is None:
        raise ImportError(
            "HvPlot and HoloViews are required for plot_comparison_interactive. "
            "Install them with `pip install hvplot holoviews`."
        )

    # 1. Source Plot
    p_src = da_src.hvplot(rasterize=rasterize, cmap=cmap, title="Source Grid", **kwargs)

    # 2. Target Plot
    p_tgt = da_tgt.hvplot(rasterize=rasterize, cmap=cmap, title="Target Grid", **kwargs)

    # 3. Difference Plot
    try:
        if regridder is not None:
            da_src_interp = regridder(da_src)
        else:
            da_src_interp = da_src.interp_like(da_tgt, method="linear")

        diff = da_tgt - da_src_interp
        p_diff = diff.hvplot(
            rasterize=rasterize,
            cmap=diff_cmap,
            title="Difference (Tgt - Src_interp)",
            **kwargs,
        )
    except Exception as e:
        # Fallback to a placeholder if difference computation fails
        p_diff = hv.Text(0.5, 0.5, f"Could not compute difference:\n{e}")

    layout = (p_src + p_tgt + p_diff).cols(3)

    if title:
        layout = layout.opts(title=title)

    return layout


def plot_weights(
    regridder: "Regridder",
    row_idx: int,
    **kwargs: Any,
) -> Any:
    """
    Track A: Visualize source points contributing to a specific destination point.

    Parameters
    ----------
    regridder : Regridder
        The Regridder instance.
    row_idx : int
        The index of the destination point (0-based).
    **kwargs : Any
        Additional arguments passed to plot_static.

    Returns
    -------
    Any
        The plot object.
    """
    # Use weights property to ensure they are gathered if remote
    matrix = regridder.weights
    row = matrix.getrow(row_idx).toarray().flatten()

    # Reconstruct 2D/1D array on source grid
    da_weights = xr.DataArray(
        row.reshape(regridder._shape_source),
        dims=regridder._dims_source,
        coords={
            c: regridder.source_grid_ds.coords[c]
            for c in regridder.source_grid_ds.coords
            if set(regridder.source_grid_ds.coords[c].dims).issubset(
                set(regridder._dims_source)
            )
        },
        name="weights",
    )

    return plot_static(
        da_weights, title=f"Weights for Destination Point {row_idx}", **kwargs
    )
