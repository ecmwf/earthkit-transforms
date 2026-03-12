import typing as T

import xarray as xr

from earthkit.transforms import _tools
from earthkit.transforms._aggregate import reduce as _reduce
from earthkit.transforms._aggregate import resample
from earthkit.transforms._tools import groupby_time
from earthkit.transforms.temporal import reduce as _temporal_reduce


@_tools.transform_inputs_decorator()
@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator
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
    dataarray : xr.DataArray
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
        Valid options are `day`, `week` and `month`. If not provided the climatology is
        calculated for the period.
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
    xr.DataArray

    """
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
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "mean"
    return reduce(*_args, **_kwargs)


def median(*_args, **_kwargs) -> xr.DataArray:
    """Calculate the climatological median.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    result = quantiles(*_args, q=[0.5], **_kwargs)
    return result.isel(quantile=0)


def min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological minimum.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "max"
    return reduce(*_args, **_kwargs)


def max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological maximum.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "max"
    return reduce(*_args, **_kwargs)


def std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the climatological standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "std"
    return reduce(*_args, **_kwargs)


def daily_reduce(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Reduce the data to the daily climatology of the provided "how" method.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["frequency"] = "dayofyear"
    return reduce(*_args, **_kwargs)


def daily_mean(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological mean.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "mean"
    return daily_reduce(*_args, **_kwargs)


def daily_median(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological median.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "median"
    return daily_reduce(*_args, **_kwargs)


def daily_min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological min.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "min"
    return daily_reduce(*_args, **_kwargs)


def daily_max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological max.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "max"
    return daily_reduce(*_args, **_kwargs)


def daily_std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the daily climatological standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "std"
    return daily_reduce(*_args, **_kwargs)


def monthly_reduce(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Reduce the data to the monthly climatology of the provided "how" method.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["frequency"] = "month"
    return reduce(*_args, **_kwargs)


def monthly_mean(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological mean.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "mean"
    return monthly_reduce(*_args, **_kwargs)


def monthly_median(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological median.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "median"
    return monthly_reduce(*_args, **_kwargs)


def monthly_min(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological min.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "min"
    return monthly_reduce(*_args, **_kwargs)


def monthly_max(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological max.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "max"
    return monthly_reduce(*_args, **_kwargs)


def monthly_std(*_args, **_kwargs) -> xr.Dataset | xr.DataArray:
    """Calculate the monthly climatological standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    xr.DataArray

    """
    _kwargs["how"] = "std"
    return monthly_reduce(*_args, **_kwargs)


@_tools.transform_inputs_decorator()
@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator
@_tools.season_order_decorator
def quantiles(
    dataarray: xr.Dataset | xr.DataArray,
    q: float | list,
    time_dim: str | None = None,
    groupby_kwargs: dict | None = None,
    **reduce_kwargs,
) -> xr.DataArray:
    """Calculate a set of climatological quantiles.

    Parameters
    ----------
    dataarray : xr.DataArray
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
    groupby_kwargs : dict
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.groupby_time`
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.transforms.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray

    """
    groupby_kwargs = groupby_kwargs or {}
    groupby_kwargs.setdefault("frequency", "year")
    grouped_data = groupby_time(dataarray.chunk({time_dim: -1}), time_dim=time_dim, **groupby_kwargs)
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
    dataarray : xr.DataArray
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
    xr.DataArray

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
    result = result.drop("quantile")
    return result


@_tools.transform_inputs_decorator()
def anomaly(
    dataarray: xr.Dataset | xr.DataArray,
    climatology: xr.Dataset | xr.DataArray,
    **_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Calculate the anomaly from a reference climatology.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a time dimension indicated by time_dim.
    climatology :  (xr.DataArray, optional)
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
    xr.DataArray

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
@_tools.groupby_kwargs_decorator
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
    dataarray : xr.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a time dimension indicated by time_dim.
    climatology :  (xr.DataArray)
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
    xr.DataArray

    """
    reduce_kwargs.setdefault("how", "mean")
    var_name = dataarray.name
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
                    f"the name of the anomaly dataarray: {var_name}"
                )
    else:
        climatology_da = climatology

    # If frequency not defined, it is deduced from the climatology.
    # This is somewhat hardcoded, but it is best practice, so for now it can stay here
    groupby_kwargs = groupby_kwargs or {}
    if groupby_kwargs.get("frequency") is None:
        for freq in ["dayofyear", "week", "month"]:
            if freq in climatology_da.dims:
                groupby_kwargs["frequency"] = freq
                break
        else:
            groupby_kwargs["frequency"] = "year"

    # Annual anomalies are simpler and do not need to be subtracted from before resampling
    if groupby_kwargs["frequency"] == "year":
        anomaly_array = (
            _temporal_reduce(dataarray, time_dim=time_dim, **groupby_kwargs, **reduce_kwargs) - climatology_da
        )

        if relative:
            anomaly_array = (anomaly_array / climatology_da) * 100.0

    else:
        anomaly_array = groupby_time(dataarray, time_dim=time_dim, **groupby_kwargs) - climatology_da

        if relative:
            anomaly_array = (
                groupby_time(anomaly_array, time_dim=time_dim, **groupby_kwargs) / climatology_da
            ) * 100.0
        anomaly_array = resample(anomaly_array, **reduce_kwargs, **groupby_kwargs, dim=time_dim)

    if relative:
        name_tag = "relative anomaly"
        update_attrs = {"units": "%"}
    else:
        name_tag = "anomaly"
        update_attrs = {}

    return _update_anomaly_array(
        anomaly_array, dataarray, var_name, name_tag, update_attrs, how_label=how_label
    )


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
@_tools.groupby_kwargs_decorator
@_tools.season_order_decorator
def relative_anomaly(*_args, **_kwargs):
    """Calculate the relative anomaly from a reference climatology, i.e. percentage change.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a `time` dimension.
    climatology :  (xr.DataArray, optional)
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
    xr.DataArray

    """
    anomaly_xarray = anomaly(*_args, relative=True, **_kwargs)

    return anomaly_xarray


@_tools.time_dim_decorator
@_tools.groupby_kwargs_decorator
@_tools.transform_inputs_decorator()
def auto_anomaly(
    dataarray: xr.Dataset | xr.DataArray,
    *_args,
    climatology_range: tuple | None = None,
    climatology_how: str = "mean",
    relative: bool = False,
    **_kwargs,
):
    """Calculate the anomaly from a reference climatology.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the anomaly from the reference
        climatology. Must contain a `time` dimension.
    climatology :  (xr.DataArray, optional)
        Reference climatology data against which the anomaly is to be calculated.
        If not provided then the climatological mean is calculated from dataarray.
    climatology_range : (list or tuple, optional)
        Start and end year of the period to be used for the reference climatology. Default
        is to use the entire time-series.
    climatology_how : string
        Method used to calculate climatology, default is "mean". Accepted values are "median", "min", "max"
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
        Any other kwargs that are accepted by `earthkit.transforms.resample`

    Returns
    -------
    xr.DataArray

    """
    clim_kwargs = {k: v for k, v in _kwargs.items() if k not in ["how"]}
    climatology = reduce(
        dataarray, *_args, how=climatology_how, climatology_range=climatology_range, **clim_kwargs
    )

    return anomaly(dataarray, climatology, *_args, relative=relative, **_kwargs)
