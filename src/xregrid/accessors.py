from __future__ import annotations

from typing import Any

import xarray as xr

from xregrid.regridder import Regridder


@xr.register_dataarray_accessor("regrid")
class RegridDataArrayAccessor:
    """
    Xarray Accessor for regridding DataArrays.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the DataArray regrid accessor.

        Parameters
        ----------
        xarray_obj : xr.DataArray
            The DataArray to regrid.
        """
        self._obj = xarray_obj

    def to(self, target_grid: xr.Dataset, **kwargs: Any) -> xr.DataArray:
        """
        Regrid the DataArray to a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        xr.DataArray
            The regridded DataArray.
        """
        # Convert DataArray to Dataset to ensure compatibility with Regridder
        source_ds = self._obj.to_dataset(name="_tmp_data")
        regridder = Regridder(source_ds, target_grid, **kwargs)
        return regridder(self._obj)


@xr.register_dataset_accessor("regrid")
class RegridDatasetAccessor:
    """
    Xarray Accessor for regridding Datasets.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the Dataset regrid accessor.

        Parameters
        ----------
        xarray_obj : xr.Dataset
            The Dataset to regrid.
        """
        self._obj = xarray_obj

    def to(self, target_grid: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """
        Regrid the Dataset to a target grid.

        Parameters
        ----------
        target_grid : xr.Dataset
            The target grid dataset.
        **kwargs : Any
            Arguments passed to the Regridder constructor.

        Returns
        -------
        xr.Dataset
            The regridded Dataset.
        """
        regridder = Regridder(self._obj, target_grid, **kwargs)
        return regridder(self._obj)
