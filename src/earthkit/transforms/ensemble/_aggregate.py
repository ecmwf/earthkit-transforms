# Copyright 2024-, European Centre for Medium Range Weather Forecasts.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Union

import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms import _tools
from earthkit.transforms._aggregate import reduce as _reduce


@format_handler()
def reduce(
    dataarray: xr.DataArray | xr.Dataset,
    how: Union[str, Callable] = "mean",
    dim: Optional[str] = None,
):
    """Reduce data over the ensemble dimension.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to reduce. Must contain an ensemble dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, an earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an numpy.ndarray over an integer valued axis.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced over the ensemble dimension.

    """
    if dim is None:
        dim = _tools.get_dim_key(dataarray, "realization", raise_error=True)
    return _reduce(dataarray, how=how, dim=dim)


def mean(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble mean.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to calculate the ensemble mean. Must contain
        an ensemble dimension.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced to the ensemble mean.

    """
    kwargs["how"] = "mean"
    return reduce(*args, **kwargs)


def std(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble standard deviation.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to calculate the ensemble standard deviation.
        Must contain an ensemble dimension.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced to the ensemble standard deviation.

    """
    kwargs["how"] = "std"
    return reduce(*args, **kwargs)


def min(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble minimum.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to calculate the ensemble minimum. Must contain
        an ensemble dimension.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced to the ensemble minimum.

    """
    kwargs["how"] = "min"
    return reduce(*args, **kwargs)


def max(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble maximum.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to calculate the ensemble maximum. Must contain
        an ensemble dimension.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced to the ensemble maximum.

    """
    kwargs["how"] = "max"
    return reduce(*args, **kwargs)


def sum(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble sum.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        The DataArray over which to calculate the ensemble sum. Must contain
        an ensemble dimension.
    dim : str, optional
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        Data reduced to the ensemble sum.

    """
    kwargs["how"] = "sum"
    return reduce(*args, **kwargs)
