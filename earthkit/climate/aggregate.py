import typing as T
from copy import deepcopy

import numpy as np
import xarray as xr

from earthkit.climate import tools

#: Mapping from pandas frequency strings to xarray time groups
_PANDAS_FREQUENCIES = {
    "D": "dayofyear",
    "W": "weekofyear",
    "M": "month",
    "H": "hour",
}
_PANDAS_FREQUENCIES_R = {v: k for k, v in _PANDAS_FREQUENCIES.items()}

#: The maximum limit of climatology time groups
_BIN_MAXES = {
    "hour": 24,
    "dayofyear": 366,
    "weekofyear": 53,
    "month": 12,
    "season": 4,
}


@tools.time_dim_decorator
def daily_mean(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the daily mean.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="D", dim=time_dim, how="mean", **kwargs)


@tools.time_dim_decorator
def daily_max(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the daily max.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="D", dim=time_dim, how="max", **kwargs)


@tools.time_dim_decorator
def daily_min(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the daily min.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="D", dim=time_dim, how="min", **kwargs)


@tools.time_dim_decorator
def monthly_mean(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the monthly mean.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="M", dim=time_dim, how="mean", **kwargs)


@tools.time_dim_decorator
def monthly_max(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the monthly max.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="M", dim=time_dim, how="max", **kwargs)


@tools.time_dim_decorator
def monthly_min(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Calculate the monthly min.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="M", dim=time_dim, how="min", **kwargs)


def resample(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    frequency: str or int or float,
    dim: str = "time",
    how: str = "mean",
    closed: str = "left",
    label: str = "left",
    skipna: bool = True,
    **kwargs,
) -> xr.DataArray:
    """
    Resample dataarray to a user-defined frequency using a user-defined "how" method.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray to be resampled.
    frequency : str, int, float
        The frequency at which to resample the chosen dimension. The format must be applicable
        to the chosen dimension.
    dim: str
        The dimension to resample along, default is `time`
    how: str
        The reduction method for resampling, default is `mean`
    **kwargs
        Keyword arguments to be passed to :func:`resample`. Defaults have been set as:
        `{"closed": "left", "skipna": True, "label": "left"}`

    Returns
    -------
    xr.DataArray
    """
    # Translate and xarray frequencies to pandas language:
    frequency = _PANDAS_FREQUENCIES_R.get(frequency, frequency)
    resample = dataarray.resample(
        label=label, closed=closed, skipna=skipna, **{dim: frequency}, **kwargs
    )
    result = resample.__getattribute__(how)(dim)
    return result


def _groupby_time(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    frequency: str = None,
    bin_widths: T.Union[int, None] = None,
    squeeze: bool = True,
    time_dim: str = "time",
):
    if frequency is None:
        try:
            frequency = xr.infer_freq(dataarray.time)
        except:  # noqa: E722
            raise ValueError(
                "Unable to infer time frequency from data; please pass the "
                "'frequency' argument explicitly"
            )
        frequency, possible_bins = _pandas_frequency_and_bins(frequency)
        bin_widths = bin_widths or possible_bins

    if bin_widths is not None:
        grouped_data = _groupby_bins(
            dataarray, frequency, bin_widths, squeeze, time_dim=time_dim
        )
    else:
        try:
            grouped_data = dataarray.groupby(f"{time_dim}.{frequency}", squeeze=squeeze)
        except AttributeError:
            raise ValueError(
                f"Invalid frequency '{frequency}' - see xarray documentation for "
                f"a full list of valid frequencies."
            )

    return grouped_data


def _groupby_bins(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    frequency: str,
    bin_widths: int = 1,
    squeeze: bool = False,
    time_dim: str = "time",
):
    if not isinstance(bin_widths, (list, tuple)):
        max_value = _BIN_MAXES[frequency]
        bin_widths = list(range(0, max_value + 1, bin_widths))
    try:
        grouped_data = dataarray.groupby_bins(
            f"{time_dim}.{frequency}", bin_widths, squeeze=squeeze
        )
    except AttributeError:
        raise ValueError(
            f"Invalid frequency '{frequency}' - see xarray documentation for "
            f"a full list of valid frequencies."
        )
    return grouped_data


def _pandas_frequency_and_bins(
    frequency: str,
) -> tuple:
    freq = frequency.lstrip("0123456789")
    bins = int(frequency[: -len(freq)]) or None
    freq = _PANDAS_FREQUENCIES.get(freq.lstrip(" "), frequency)
    return freq, bins


def _reduce_dataarray(
    dataarray: xr.DataArray,
    how: T.Union[T.Callable, str] = "mean",
    weights: T.Union[None, str, np.ndarray] = None,
    how_label: str = "",
    how_dropna=False,
    **kwargs,
):
    """
    Reduce an xarray.dataarray or xarray.dataset using a specified `how` method.

    With the option to apply weights either directly or using a specified
    `weights` method.

    Parameters
    ----------
    dataarray : xr.DataArray or xr.Dataset
        Data object to reduce
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, a earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an np.ndarray over an integer valued axis
    weights : str
        Choose a recognised method to apply weighting. Currently availble methods are; 'latitude'
    how_dropna : str
        Choose how to drop nan values.
        Default is None and na values are preserved. Options are 'any' and 'all'.
    **kwargs :
        kwargs recognised by the how :func: `reduce`

    Returns
    -------
        A data array with reduce dimensions removed.

    """
    # If weighted, use xarray weighted methods
    if weights is not None:
        # Create any standard weights, i.e. latitude
        if isinstance(weights, str):
            weights = tools.WEIGHTS_DICT[weights](dataarray)
        # We ensure the callable is always a string
        if callable(how):
            how = how.__name__
        # map any alias methods:
        how = tools.WEIGHTED_HOW_METHODS.get(how, how)

        dataarray = dataarray.weighted(weights)

        red_array = dataarray.__getattribute__(how)(**kwargs)

    else:
        # If how is string, fetch function from dictionary:
        if isinstance(how, str) and how in dir(dataarray):
            red_array = dataarray.__getattribute__(how)(**kwargs)
        else:
            if isinstance(how, str):
                how_label = deepcopy(how)
                how = tools.get_how(how)
            assert isinstance(how, T.Callable), f"how method not recognised: {how}"

            red_array = dataarray.reduce(how, **kwargs)

    if how_label:
        red_array = red_array.rename(f"{red_array.name}_{how_label}")

    if how_dropna:
        red_array = red_array.dropna(how_dropna)

    return red_array


def reduce(
    dataarray: T.Union[xr.DataArray, xr.Dataset],
    *args,
    **kwargs,
):
    """
    Reduce an xarray.dataarray or xarray.dataset using a specified `how` method.

    With the option to apply weights either directly or using a specified
    `weights` method.

    Parameters
    ----------
    dataarray : xr.DataArray or xr.Dataset
        Data object to reduce
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, a earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an np.ndarray over an integer valued axis
    weights : str
        Choose a recognised method to apply weighting. Currently availble methods are; 'latitude'
    how_label : str
        Label to append to the name of the variable in the reduced object
    how_dropna : str
        Choose how to drop nan values.
        Default is None and na values are preserved. Options are 'any' and 'all'.
    **kwargs :
        kwargs recognised by the how :func: `reduce`

    Returns
    -------
        A data array with reduce dimensions removed.

    """
    if isinstance(dataarray, (xr.Dataset)):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _reduce_dataarray(dataarray[var], *args, **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _reduce_dataarray(dataarray, *args, **kwargs)


def rolling_reduce(
    dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs
) -> xr.DataArray:
    """Return reduced data using a moving window over which to apply the reduction.

    Parameters
    ----------
    dataarray : xr.DataArray or xr.Dataset
        Data over which the moving window is applied according to the reduction method.
    windows :
        windows for the rolling groups, for example `time=10` to perform a reduction
        in the time dimension with a bin size of 10. the rolling groups can be defined
        over any number of dimensions. **see documentation for xarray.dataarray.rolling**.
    min_periods : integer
        The minimum number of observations in the window required to have a value
        (otherwise result is NaN). Default is to set **min_periods** equal to the size of the window.
        **see documentation for xarray.dataarray.rolling**
    center : bool
        Set the labels at the centre of the window, **see documentation for xarray.dataarray.rolling**.
    how_reduce : str,
        Function to be applied for reduction. Default is 'mean'.
    how_dropna : str
        Determine if dimension is removed from the output when we have at least one NaN or
        all NaN. **how_dropna** can be 'None', 'any' or 'all'. Default is 'any'.
    **kwargs :
        Any kwargs that are compatible with the select `how_reduce` method.

    Returns
    -------
    xr.DataArray or xr.Dataset (as provided)
    """
    if isinstance(dataarray, (xr.Dataset)):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _rolling_reduce_dataarray(dataarray[var], *args, **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _rolling_reduce_dataarray(dataarray, *args, **kwargs)


def _rolling_reduce_dataarray(
    dataarray: xr.DataArray, how_reduce="mean", how_dropna="any", **kwargs
) -> xr.DataArray:
    """Return reduced data using a moving window over which to apply the reduction.

    Parameters
    ----------
    dataarray : xr.DataArray
        Data over which the moving window is applied according to the reduction method.
    windows :
        windows for the rolling groups, for example `time=10` to perform a reduction
        in the time dimension with a bin size of 10. the rolling groups can be defined
        over any number of dimensions. **see documentation for xarray.dataarray.rolling**.
    min_periods : integer
        The minimum number of observations in the window required to have a value
        (otherwise result is NaN). Default is to set **min_periods** equal to the size of the window.
        **see documentation for xarray.dataarray.rolling**
    center : bool
        Set the labels at the centre of the window, **see documentation for xarray.dataarray.rolling**.
    how_reduce : str,
        Function to be applied for reduction. Default is 'mean'.
    how_dropna : str
        Determine if dimension is removed from the output when we have at least one NaN or
        all NaN. **how_dropna** can be 'None', 'any' or 'all'. Default is 'any'.
    **kwargs :
        Any kwargs that are compatible with the select `how_reduce` method.

    Returns
    -------
    xr.DataArray
    """
    # Expand dim kwarg to individual kwargs
    if isinstance(kwargs.get("dim"), dict):
        kwargs.update(kwargs.pop("dim"))

    window_dims = [_dim for _dim in list(dataarray.dims) if _dim in list(kwargs)]
    rolling_kwargs_keys = ["min_periods", "center"] + window_dims
    rolling_kwargs_keys = [_kwarg for _kwarg in kwargs if _kwarg in rolling_kwargs_keys]
    rolling_kwargs = {_kwarg: kwargs.pop(_kwarg) for _kwarg in rolling_kwargs_keys}

    # Any kwargs left after above reductions are kwargs for reduction method
    reduce_kwargs = kwargs
    # Create rolling groups:
    data_rolling = dataarray.rolling(**rolling_kwargs)

    data_windowed = _reduce_dataarray(data_rolling, how=how_reduce, **reduce_kwargs)

    data_windowed = _dropna(data_windowed, window_dims, how_dropna)

    data_windowed.attrs.update(dataarray.attrs)

    return data_windowed


def _dropna(data, dims, how):
    """Method for drop nan values."""
    if how in [None, "None", "none"]:
        return data

    for dim in dims:
        data = data.dropna(dim, how=how)
    return data
