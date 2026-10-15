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

import typing as T

import numpy as np
import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms import _tools
from earthkit.transforms._aggregate import reduce as _reduce
from earthkit.transforms._tools import groupby_time
from earthkit.transforms.temporal import reduce as _temporal_reduce


@format_handler()
@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator(climatology=True)
@_tools.season_order_decorator
def reduce(
    dataarray: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    how: str | T.Callable | None = "mean",
    groupby_kwargs: dict | None = None,
    climatology_range: tuple | list | None = None,
    **reduce_kwargs,
):
    """Group data annually over a given `frequency` and reduce using the specified `how` method.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, an earthkit how method or
        any method compatible with the array namespace of the data.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit,
        array_namespace.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an array over an integer valued axis
    frequency : str (optional)
        Frequency used for grouping the data in climatology mode. Typical values include
        `dayofyear`, `weekofyear`, `month`, `year`, etc. The full set of accepted options
        matches those supported by `earthkit.transforms._tools.groupby_time`. If not
        provided, the climatology is calculated over the entire period.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on the
        frequency provided by `frequency`. If `bin_widths` is a sequence it defines the
        edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    climatology_range : (list or tuple, optional)
        Start and end year of the period to be used for the reference climatology. Default
        is to use the entire time-series.
    groupby_kwargs : dict
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.groupby_time`
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    # Validate and normalize climatology_range if provided
    if climatology_range is not None:
        try:
            start, end = climatology_range  # expect exactly two items
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "climatology_range must be a sequence of exactly two items (start, end), or None."
            ) from exc
        climatology_range = (start, end)

    # If climate range is defined, use it
    if climatology_range is not None and all(c_r is not None for c_r in climatology_range):
        selection = dataarray.sel({time_dim: slice(*climatology_range)})
    else:
        selection = dataarray

    groupby_kwargs = groupby_kwargs or {}
    if groupby_kwargs.get("frequency") is not None:
        grouped_data = groupby_time(
            selection,
            time_dim=time_dim,
            **groupby_kwargs,
        )
        return _reduce(grouped_data, how=how, dim=time_dim, **reduce_kwargs)

    return _reduce(selection, how=how, dim=time_dim, **reduce_kwargs)


def mean(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological mean.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "mean"
    return reduce(*_args, **_kwargs)


def median(*_args, **_kwargs) -> xr.DataArray:
    """Calculate the climatological median.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological median. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    result = quantiles(*_args, q=[0.5], **_kwargs)
    return result.isel(quantile=0)


def min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological minimum.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological minimum. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "min"
    return reduce(*_args, **_kwargs)


def max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological maximum.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological maximum. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "max"
    return reduce(*_args, **_kwargs)


def std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological standard deviation.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological standard deviation.
        Must contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)


    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "std"
    return reduce(*_args, **_kwargs)


def daily_reduce(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Reduce the data to the daily climatology of the provided "how" method.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, an earthkit how method or
        any method compatible with the array namespace of the data.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit,
        array_namespace.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an array over an integer valued axis
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["frequency"] = "dayofyear"
    return reduce(*_args, **_kwargs)


def daily_mean(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological mean.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "mean"
    return daily_reduce(*_args, **_kwargs)


def daily_median(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological median.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological median. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "median"
    return daily_reduce(*_args, **_kwargs)


def daily_min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological min.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological min. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "min"
    return daily_reduce(*_args, **_kwargs)


def daily_max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological max.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological max. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "max"
    return daily_reduce(*_args, **_kwargs)


def daily_std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological standard deviation.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological standard deviation.
        Must contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)


    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "std"
    return daily_reduce(*_args, **_kwargs)


def monthly_reduce(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Reduce the data to the monthly climatology of the provided "how" method.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, an earthkit how method or
        any method compatible with the array namespace of the data.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit,
        array_namespace.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an array over an integer valued axis
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["frequency"] = "month"
    return reduce(*_args, **_kwargs)


def monthly_mean(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological mean.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "mean"
    return monthly_reduce(*_args, **_kwargs)


def monthly_median(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological median.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological median. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "median"
    return monthly_reduce(*_args, **_kwargs)


def monthly_min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological min.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological min. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "min"
    return monthly_reduce(*_args, **_kwargs)


def monthly_max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological max.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological max. Must
        contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "max"
    return monthly_reduce(*_args, **_kwargs)


def monthly_std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological standard deviation.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological standard deviation.
        Must contain a `time` dimension.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)


    Returns
    -------
    xarray.DataArray

    """
    _kwargs["how"] = "std"
    return monthly_reduce(*_args, **_kwargs)


@format_handler()
@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator(climatology=True)
@_tools.season_order_decorator
def quantiles(
    dataarray: xr.Dataset | xr.DataArray,
    q: float | list,
    time_dim: str | None = None,
    groupby_kwargs: dict | None = None,
    climatology_range: tuple | list | None = None,
    **reduce_kwargs,
) -> xr.DataArray:
    """Calculate a set of climatological quantiles.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological quantiles. Must
        contain a `time` dimension.
    q : float | list
        The quantile, or list of quantiles, to calculate the climatology.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is `year`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    climatology_range : (list or tuple, optional)
        Start and end year of the period to be used for the reference climatology. Default
        is to use the entire time-series.
    groupby_kwargs : dict
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.groupby_time`
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    # Validate and normalize climatology_range if provided
    if climatology_range is not None:
        try:
            start, end = climatology_range  # expect exactly two items
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "climatology_range must be a sequence of exactly two items (start, end), or None."
            ) from exc
        climatology_range = (start, end)

    # If climate range is defined, use it
    if climatology_range is not None and all(c_r is not None for c_r in climatology_range):
        selection = dataarray.sel({time_dim: slice(*climatology_range)})
    else:
        selection = dataarray

    groupby_kwargs = groupby_kwargs or {}
    groupby_kwargs.setdefault("frequency", "year")
    grouped_data = groupby_time(selection.chunk({time_dim: -1}), time_dim=time_dim, **groupby_kwargs)
    results = []
    if not isinstance(q, (list, tuple)):
        q = [q]
    for quantile in q:
        results.append(
            grouped_data.quantile(
                q=quantile,
                dim=time_dim,
                **reduce_kwargs,
            )
        )
    result = xr.concat(results, dim="quantile")
    return result


def percentiles(
    dataarray: xr.Dataset | xr.DataArray,
    p: float | list,
    **_kwargs,
) -> xr.DataArray:
    """Calculate a set of climatological percentiles.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the climatological percentiles. Must
        contain a `time` dimension.
    p : float | list
        The pecentile, or list of percentiles, to calculate the climatology.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xarray.DataArray

    """
    if not isinstance(p, (list, tuple)):
        p = [p]
    q = [_p * 1e-2 for _p in p]
    quantile_data = quantiles(
        dataarray,
        q,
        **_kwargs,
    )
    result = quantile_data.assign_coords(percentile=("quantile", p))
    result = result.swap_dims({"quantile": "percentile"})
    result = result.drop_vars("quantile")
    return result


@format_handler()
def anomaly(
    dataarray: xr.Dataset | xr.DataArray,
    climatology: xr.Dataset | xr.DataArray,
    **_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Calculate the anomaly from a reference climatology.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a time dimension indicated by time_dim.
    climatology :  (xarray.DataArray, optional)
        Reference climatology data against which the anomaly is to be calculated.
        If not provided then the climatological mean is calculated from dataarray.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    relative : bool (optional)
        Return the relative anomaly, i.e. the percentage change w.r.t the climatological period
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.climatology.mean`

    Returns
    -------
    xarray.DataArray

    """
    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _anomaly_dataarray(dataarray[var], climatology, **_kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _anomaly_dataarray(dataarray, climatology, **_kwargs)


@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator(climatology=True)
@_tools.season_order_decorator
def _anomaly_dataarray(
    dataarray: xr.DataArray,
    climatology: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    groupby_kwargs: dict | None = None,
    relative: bool = False,
    climatology_how_tag: str = "",
    how_label: str | None = None,
    **reduce_kwargs,
) -> xr.DataArray:
    """Calculate the anomaly from a reference climatology.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a time dimension indicated by time_dim.
    climatology :  (xarray.DataArray)
        Reference climatology data against which the anomaly is to be calculated.
        If not provided then the climatological mean is calculated from dataarray.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    relative : bool (optional)
        Return the relative anomaly, i.e. the percentage change w.r.t the climatological period
    climatology_how_tag : str (optional)
        Tag to identify the climatology variable in the climatology dataset
    how_label : str (optional)
        Label to append to the variable name of the anomaly dataarray
    groupby_kwargs : dict
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.groupby_time`
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.climatology.mean`

    Returns
    -------
    xarray.DataArray

    """
    reduce_kwargs.setdefault("how", "mean")
    var_name = dataarray.name
    climatology_da = _climatology_dataarray(climatology, var_name, climatology_how_tag)

    # If frequency not defined, it is deduced from the climatology.
    # This is somewhat hardcoded, but it is best practice, so for now it can stay here
    for clim_freq in _tools.VALID_CLIMATOLOGY_FREQUENCIES:
        if clim_freq in climatology_da.dims:
            break
    else:
        clim_freq = "year"

    groupby_kwargs = groupby_kwargs or {}
    if groupby_kwargs.get("frequency") == "climatology":
        groupby_kwargs["frequency"] = clim_freq

    # Annual anomalies are simpler and do not need to be subtracted from before resampling
    frequency = groupby_kwargs.get("frequency")
    clim_groupby_kwargs = {k: v for k, v in groupby_kwargs.items() if k != "frequency"}

    if frequency is None:
        if clim_freq == "year":
            # If frequency is None, and clim frequency is year, then we can just take the difference
            anomaly_array = dataarray - climatology_da
            if relative:
                anomaly_array = (anomaly_array / climatology_da) * 100.0
        else:
            # If clim_freq is not year, then we need to groupby the dataarray before taking
            # the difference
            anomaly_array = (
                groupby_time(dataarray, time_dim=time_dim, frequency=clim_freq, **clim_groupby_kwargs) - climatology_da
            )
            if relative:
                anomaly_array = (
                    groupby_time(anomaly_array, time_dim=time_dim, frequency=clim_freq, **clim_groupby_kwargs)
                    / climatology_da
                    * 100.0
                )
            anomaly_array = anomaly_array.broadcast_like(dataarray)

    elif frequency == "year":
        anomaly_array = (
            _temporal_reduce(dataarray, time_dim=time_dim, **groupby_kwargs, **reduce_kwargs) - climatology_da
        )

        if relative:
            anomaly_array = (anomaly_array / climatology_da) * 100.0

    else:
        if clim_freq == "year":
            anomaly_array = dataarray - climatology_da
            if relative:
                anomaly_array = (anomaly_array / climatology_da) * 100.0
        else:
            # Need to group the dataarray to the same frequency as the climatology before taking the difference,
            # and then broadcast back to the original dataarray dimensions
            anomaly_array = (
                groupby_time(dataarray, time_dim=time_dim, frequency=clim_freq, **clim_groupby_kwargs) - climatology_da
            )

            if relative:
                anomaly_array = (
                    groupby_time(anomaly_array, time_dim=time_dim, frequency=clim_freq, **clim_groupby_kwargs)
                    / climatology_da
                    * 100.0
                )

            # The broadcast_like is probably not necessary as the _temporal_reduce should take care
            # of things, but it is a useful safeguard against any potential changes in downstream processing
            anomaly_array = anomaly_array.broadcast_like(dataarray)

        anomaly_array = _temporal_reduce(anomaly_array, time_dim=time_dim, **groupby_kwargs, **reduce_kwargs)

    if relative:
        name_tag = "relative anomaly"
        update_attrs = {"units": "%"}
    else:
        name_tag = "anomaly"
        update_attrs = {}

    return _update_anomaly_array(anomaly_array, dataarray, var_name, name_tag, update_attrs, how_label=how_label)


def _climatology_dataarray(
    climatology: xr.Dataset | xr.DataArray,
    var_name: T.Hashable,
    climatology_how_tag: str = "",
) -> xr.DataArray:
    """Return the climatology variable that matches the data variable `var_name`."""
    if isinstance(climatology, xr.Dataset):
        if var_name in climatology:
            climatology_da = climatology[var_name]
        else:
            potential_clim_vars = [c_var for c_var in climatology.data_vars if str(var_name) in str(c_var)]
            if len(potential_clim_vars) == 1:
                climatology_da = climatology[potential_clim_vars[0]]
            elif f"{var_name}_{climatology_how_tag}" in potential_clim_vars:
                climatology_da = climatology[f"{var_name}_{climatology_how_tag}"]
            elif len(potential_clim_vars) > 1:
                raise KeyError(
                    "Multiple potential climatologies found in climatology dataset, "
                    "please identify appropriate statistic with `climatology_how_tag`.\n"
                    f"Potential climatology variables found: {potential_clim_vars}"
                )
            else:
                raise ValueError(
                    "Could not find a variable in the climatology dataset that matches "
                    f"the name of the dataarray: {var_name}"
                )
    else:
        climatology_da = climatology
    return climatology_da


def _climatology_of_time_steps(
    dataarray: xr.DataArray,
    climatology_da: xr.DataArray,
    clim_freq: str,
    time_dim: T.Hashable,
    chunks: dict | None = None,
) -> xr.DataArray:
    """Return the climatology of each time step of the data, from a climatology grouped by `clim_freq`.

    The climatology of each time step's group is selected in one step, as groupby arithmetic with lazily
    loaded data or a lazily loaded climatology creates a task for every few time steps. Points and time steps
    whose group the climatology does not cover are given missing values. If either is lazily loaded, `chunks`
    gives the chunks of the climatology's dimensions that the data does not have, which keep theirs otherwise.
    """
    _, climatology_da = xr.align(dataarray, climatology_da, join="left", exclude=[clim_freq], copy=False)
    groups = dataarray[f"{time_dim}.{clim_freq}"]
    climatology_da = climatology_da.reindex({clim_freq: np.unique(groups.values)})
    # If either is lazily loaded, the selection is too, with the chunks of the data. An eager selection
    # for lazily loaded data would hold a climatology the size of the data in memory.
    lazy = dataarray.chunks is not None or climatology_da.chunks is not None
    if lazy:
        data_chunks = {
            dim: dataarray.chunksizes[dim]
            for dim in climatology_da.dims
            if dim in dataarray.chunksizes and dim != clim_freq
        }
        # The selection has time chunks the size of the climatology's chunks of groups, so these are made
        # as large as the time chunks of the data, rather than leaving a chunk for every few time steps
        time_chunks = [size for dim in groups.dims for size in dataarray.chunksizes.get(dim, ())]
        group_chunks = int(np.min([np.max(time_chunks), climatology_da.sizes[clim_freq]])) if time_chunks else -1
        climatology_da = climatology_da.chunk({**(chunks or {}), **data_chunks, clim_freq: group_chunks or -1})
    # Select with the group labels alone and give the selection the time coordinate of the data,
    # so that other coordinates, such as the scalar quantile of an ensemble median, cannot conflict
    if time_dim not in climatology_da.dims:
        climatology_da = climatology_da.drop_vars([time_dim], errors="ignore")
    climatology_da = climatology_da.sel({clim_freq: xr.Variable(groups.dims, groups.data)})
    climatology_da = climatology_da.assign_coords({time_dim: dataarray[time_dim].variable})
    if groups.isnull().any():
        # Time steps without a group, at missing times, have no climatology
        climatology_da = climatology_da.where(groups.notnull().reset_coords(drop=True))
    if lazy:
        # The time coordinate need not be a dimension, its groups have the dimensions of the time steps
        climatology_da = climatology_da.chunk({dim: dataarray.chunksizes.get(dim, "auto") for dim in groups.dims})
    return climatology_da


def _update_anomaly_array(anomaly_array, original_array, var_name, name_tag, update_attrs, how_label=None):
    if how_label is not None:
        var_name = f"{var_name}_{how_label}"
    anomaly_array = anomaly_array.rename(f"{var_name}")
    update_attrs = {**original_array.attrs, **update_attrs}
    if "standard_name" in update_attrs:
        update_attrs["standard_name"] += f"_{name_tag.replace(' ', '_')}"
    if "long_name" in update_attrs:
        update_attrs["long_name"] += f" {name_tag}"
    anomaly_array = anomaly_array.assign_attrs(update_attrs)
    return anomaly_array


@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator(climatology=True)
@_tools.season_order_decorator
def relative_anomaly(*_args, **_kwargs):
    """Calculate the relative anomaly from a reference climatology, i.e. percentage change.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a `time` dimension.
    climatology :  (xarray.DataArray, optional)
        Reference climatology data against which the anomaly is to be calculated.
        If not provided then the climatological mean is calculated from dataarray.
    climatology_range : (list or tuple, optional)
        Start and end year of the period to be used for the reference climatology. Default
        is to use the entire time-series.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.climatology.mean`

    Returns
    -------
    xarray.DataArray

    """
    anomaly_xarray = anomaly(*_args, relative=True, **_kwargs)

    return anomaly_xarray


@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator(climatology=True)
@format_handler()
def auto_anomaly(
    dataarray: xr.Dataset | xr.DataArray,
    *_args,
    climatology_range: tuple | None = None,
    climatology_how: str = "mean",
    climatology_frequency: str | None = None,
    relative: bool = False,
    **_kwargs,
):
    """Calculate the anomaly from a reference climatology.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a `time` dimension.
    climatology_range : (list or tuple, optional)
        Start and end year of the period to be used for the reference climatology. Default
        is to use the entire time-series.
    climatology_how : string
        Method used to calculate climatology, default is "mean". Accepted values are "median", "min", "max"
    climatology_frequency : str (optional)
        Valid options are None (default), `dayofyear`, `weekofyear` and `month`. If None,
        the climatology is calculated over all time-steps
        and the anomaly is returned on the same frequency as the input data.
    frequency : str (optional)
        Valid options are `day`, `week`, `month` and `year`. The default is to return the anomaly on the
        same frequency as the input data.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    relative : bool (optional)
        Return the relative anomaly, i.e. the percentage change w.r.t the climatological period
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.resample`

    Returns
    -------
    xarray.DataArray

    """
    climatology = reduce(
        dataarray,
        *_args,
        how=climatology_how,
        climatology_range=climatology_range,
        frequency=climatology_frequency,
    )

    return anomaly(dataarray, climatology, *_args, relative=relative, **_kwargs)


@format_handler()
def quantile_categories(
    dataarray: xr.Dataset | xr.DataArray,
    climatology: xr.Dataset | xr.DataArray,
    **_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Categorise data by the climatological quantiles it falls between.

    Each value is assigned the number of climatological quantiles that it is greater than or equal to,
    so `N` quantiles give the categories `0`, below the lowest quantile, to `N`, at or above the highest
    quantile. A value equal to a quantile falls in the category above it. For example, the terciles
    give the below-normal, near-normal and above-normal categories `0`, `1` and `2`.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The DataArray to categorise. Must contain a time dimension indicated by time_dim
        if the climatology is grouped by a time frequency.
    climatology : xarray.DataArray
        Reference climatological quantiles against which the data is categorised, with a
        `quantile` or `percentile` dimension, as returned by
        `earthkit.transforms.climatology.quantiles` and `earthkit.transforms.climatology.percentiles`.
        If the climatology is grouped by `dayofyear`, `week`, `weekofyear` or `month`, each time
        step is compared with the quantiles of its group.
    quantile_dim : str (optional)
        Name of the quantile dimension in the climatology, default behaviour is to use
        `quantile` or `percentile`.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behaviour is to detect the
        time dimension from the input object
    climatology_how_tag : str (optional)
        Tag to identify the climatology variable in the climatology dataset
    how_label : str (optional)
        Label to append to the variable name of the categories dataarray

    Returns
    -------
    xarray.DataArray
        The categories, with the dimensions of the input data. Values are missing where the data,
        or any of the quantiles it is compared with, are missing.

    """
    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _quantile_categories_dataarray(dataarray[var], climatology, **_kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _quantile_categories_dataarray(dataarray, climatology, **_kwargs)


@_tools.time_dim_decorator
def _quantile_categories_dataarray(
    dataarray: xr.DataArray,
    climatology: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    quantile_dim: str | None = None,
    climatology_how_tag: str = "",
    how_label: str | None = None,
) -> xr.DataArray:
    """Categorise data by the climatological quantiles it falls between.

    See :func:`quantile_categories` for a description of the parameters.
    """
    var_name = dataarray.name
    climatology_da = _climatology_dataarray(climatology, var_name, climatology_how_tag)

    if quantile_dim is None:
        quantile_dim = next((dim for dim in ["quantile", "percentile"] if dim in climatology_da.dims), None)
    if quantile_dim not in climatology_da.dims:
        raise ValueError(
            "Could not find the quantile dimension in the climatology, please identify it with `quantile_dim`. "
            f"Climatology dimensions: {climatology_da.dims}"
        )
    if quantile_dim in dataarray.dims:
        raise ValueError(f"The quantile dimension of the climatology is also in the dataarray: {quantile_dim}")

    # The climatology frequency is deduced from the climatology, as for anomalies
    clim_freq = next(
        (freq for freq in _tools.VALID_CLIMATOLOGY_FREQUENCIES if freq in climatology_da.dims and freq != quantile_dim),
        None,
    )
    extra_dims = set(climatology_da.dims) - set(dataarray.dims) - {quantile_dim, clim_freq}
    if extra_dims:
        raise ValueError(
            f"The climatology has dimensions which are not in the dataarray: {sorted(map(str, extra_dims))}. "
            f"Climatologies grouped by time must be grouped by one of: {_tools.VALID_CLIMATOLOGY_FREQUENCIES}"
        )

    # Keep every point of the data, points without quantiles are categorised as missing. The groups of the
    # climatology are matched to the time steps, not to a dimension of the data with the same name.
    _, climatology_da = xr.align(dataarray, climatology_da, join="left", exclude=[clim_freq], copy=False)

    if clim_freq is not None:
        # Groups missing from the climatology are categorised as missing, and the quantiles of each value
        # are compared in one chunk
        climatology_da = _climatology_of_time_steps(
            dataarray, climatology_da, clim_freq, time_dim, chunks={quantile_dim: -1}
        )

    at_or_above = dataarray >= climatology_da
    categories = at_or_above.sum(quantile_dim).where(dataarray.notnull() & climatology_da.notnull().all(quantile_dim))

    if how_label is not None:
        var_name = f"{var_name}_{how_label}"
    # Categories have no units, and are no longer the quantity described by the standard_name
    update_attrs = {k: v for k, v in dataarray.attrs.items() if k not in ["units", "standard_name"]}
    if "long_name" in update_attrs:
        update_attrs["long_name"] += " quantile category"
    categories = categories.rename(var_name)
    categories.attrs = update_attrs
    return categories
