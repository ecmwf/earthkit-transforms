"""
Module that contains generalised methods for aggregating xarray objects
"""

import xarray as xr

from ._options import ALLOWED_LIBS, HOW_DICT, WEIGHT_DICT

#: Mapping from pandas frequency strings to xarray time groups
_PANDAS_FREQUENCIES = {
    "D": "dayofyear",
    "W": "weekofyear",
    "M": "month",
    "H": "hour",
}

#: The maximum limit of climatology time groups
_BIN_MAXES = {
    "hour": 24,
    "dayofyear": 366,
    "weekofyear": 53,
    "month": 12,
    "season": 4,
}


def daily_mean(dataarray, **kwargs):
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
    return resample(dataarray, frequency="D", dim="time", how="mean", **kwargs)


def daily_max(dataarray, **kwargs):
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
    return resample(dataarray, frequency="D", dim="time", how="max", **kwargs)


def daily_min(dataarray, **kwargs):
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
    return resample(dataarray, frequency="D", dim="time", how="min", **kwargs)


def monthly_mean(dataarray, **kwargs):
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
    return resample(dataarray, frequency="M", dim="time", how="mean", **kwargs)


def resample(
    dataarray: xr.DataArray,
    frequency: str or int or float,
    dim: str = "time",
    how: str = "mean",
    closed: str = "left",
    label: str = "left",
    skipna: bool = True,
    **kwargs,
) -> xr.DataArray:
    """
    Resample dataarray to a user-defined frequency using a user-defined "how" method

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
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
    resample = dataarray.resample(
        label=label, closed=closed, skipna=skipna, **{dim: frequency}, **kwargs
    )
    result = resample.__getattribute__(how)(dim)
    return result


def _groupby_time(
    dataarray: xr.DataArray,
    frequency: str = None,
    bin_widths: int = None,
    squeeze: bool = True,
    time_dim="time",
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

    try:
        grouped_data = dataarray.groupby(f"{time_dim}.{frequency}", squeeze=squeeze)
    except AttributeError:
        raise ValueError(
            f"Invalid frequency '{frequency}' - see xarray documentation for "
            f"a full list of valid frequencies."
        )
    if frequency in ["season"]:
        grouped_data = grouped_data.reindex(season=["DJF", "MAM", "JJA", "SON"])
    return grouped_data


def _groupby_bins(
    dataarray: xr.DataArray,
    frequency: str,
    bin_widths: int,
    squeeze: bool,
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


def reduce(data, how="mean", how_weights=None, how_dropna=False, **kwargs):
    """
    Reduce an xarray.dataarray or xarray.dataset using a specified `how` method
    with the option to apply weights either directly or using a specified
    `how_weights` method.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data object to reduce
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, a coucal how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, coucal, numpy.
        Otherwise it can be any function which can be called in the form f(x, axis=axis, **kwargs)
        to return the result of reducing an np.ndarray over an integer valued axis
    how_weights (optional): str
        Choose a recognised method to apply weighting. Currently availble methods are; ['latitude']
    how_dropna (optional): str
        Choose how to drop nan values.
        Default is None and na values are preserved. Options are 'any' and 'all'.
    **kwargs:
        kwargs recognised by the how :func: `reduce`

    Returns:
        A data array with dimensions [features] + [data.dims not in ['lat','lon']].
        Each slice of layer corresponds to a feature in layer.

    """

    # If latitude_weighted, build array of weights based on latitude.
    if how_weights is not None:
        weights = WEIGHT_DICT.get(how_weights)(data)
        kwargs.update(dict(weights=weights))

    in_built_how_methods = [
        method for method in dir(data) if not method.startswith("_")
    ]
    # If how is string, fetch function from dictionary:
    if isinstance(how, str):
        if how in in_built_how_methods:
            return data.__getattribute__(how)(**kwargs)
        else:
            try:
                how_method = HOW_DICT[how]
            except KeyError:
                try:
                    module, function = how.split(".")
                    how_method = getattr(globals()[ALLOWED_LIBS[module]], function)
                except KeyError:
                    raise ValueError(f"method must come from one of {ALLOWED_LIBS}")
                except AttributeError:
                    raise AttributeError(
                        f"module '{module}' has no attribute " f"'{function}'"
                    )
    else:
        how_method = how

    reduced = data.reduce(how_method, **kwargs)

    return reduced


def rolling_reduce(
    dataarray: xr.DataArray, how_reduce="mean", how_dropna="any", **kwargs
) -> xr.DataArray:
    """Return reduced data using a moving window over which to apply the reduction.

    Parameters
    ----------
    dataarray : xr.DataArray
        Data over which the moving window is applied according to the reduction method.
    **windows : (see documentation for xarray.dataarray.rolling)
        windows for the rolling groups, for example `time=10` to perform a reduction
        in the time dimension with a bin size of 10. the rolling groups can be defined
        over any number of dimensions.
    min_periods (optional) : integer (see documentation for xarray.dataarray.rolling)
        The minimum number of observations in the window required to have a value
        (otherwise result is NaN). Default is to set **min_periods** equal to the size of the window.
    center (optional): bool (see documentation for xarray.dataarray.rolling)
        Set the labels at the centre of the window.
    how_reduce (optional) : str,
        Function to be applied for reduction. Default is 'mean'.
    how_dropna (optional): str
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
    print("rolling kwargs: ", rolling_kwargs)
    # Create rolling groups:
    data_rolling = dataarray.rolling(**rolling_kwargs)
    print("reduce kwargs: ", reduce_kwargs)

    data_windowed = reduce(data_rolling, how=how_reduce, **reduce_kwargs)

    data_windowed = _dropna(data_windowed, window_dims, how_dropna)

    data_windowed.attrs.update(dataarray.attrs)

    return data_windowed


def _dropna(data, dims, how):
    """
    Method for drop nan values
    """
    if how in [None, "None", "none"]:
        return data

    for dim in dims:
        data = data.dropna(dim, how=how)
    return data
