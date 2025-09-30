# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any, Callable, Optional, Union

import xarray as xr
from earthkit.transforms import _tools
from earthkit.transforms._aggregate import reduce as _reduce


@_tools.transform_inputs_decorator()
def reduce(
    dataarray: xr.DataArray | xr.Dataset,
    how: Union[str, Callable] = "mean",
    dim: Optional[str] = None,
):
    """Reduce data over the ensemble dimension.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, an earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an np.ndarray over an integer valued axis.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    """
    if dim is None:
        dim = _tools.get_dim_key(dataarray, "realization", raise_error=True)
    return _reduce(dataarray, how=how, dim=dim)


def mean(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble mean.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.
    """
    kwargs["how"] = "mean"
    return reduce(*args, **kwargs)


def std(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.
    """
    kwargs["how"] = "std"
    return reduce(*args, **kwargs)


def min(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble minimum.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.
    """
    kwargs["how"] = "min"
    return reduce(*args, **kwargs)


def max(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble maximum.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.
    """
    kwargs["how"] = "max"
    return reduce(*args, **kwargs)


def sum(*args: Any, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """Calculate the ensemble sum.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        The DataArray over which to calculate the climatological sum. Must
        contain a `time` dimension.
    dim : str (optional)
        Name of the ensemble dimension in the data object, default behaviour is to detect the
        ensemble dimension from the input object.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the underlying reduce function.
    """
    kwargs["how"] = "sum"
    return reduce(*args, **kwargs)
