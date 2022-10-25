import xarray as xr

from .options import HOW_DICT

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


def daily_mean(datarray, **kwargs):
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
    return resample(datarray, frequency="D", dim="time", how="mean", **kwargs)


def daily_max(datarray, **kwargs):
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
    return resample(datarray, frequency="D", dim="time", how="max", **kwargs)


def daily_min(datarray, **kwargs):
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
    return resample(datarray, frequency="D", dim="time", how="min", **kwargs)


def monthly_mean(datarray, **kwargs):
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
    return resample(datarray, frequency="M", dim="time", how="mean", **kwargs)


def resample(
    dataarray: xr.DataArray,
    frequency: str,
    dim: str = "time",
    how: str = "mean",
    closed: str = "left",
    label: str = "left",
    skipna: bool = True,
    **kwargs,
) -> xr.DataArray:
    resample = dataarray.resample(
        label=label, closed=closed, skipna=skipna, **{dim: frequency}, **kwargs
    )
    result = resample.__getattribute__(how)(dim)
    return result


def groupby_time(
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
        return _groupby_bins(
            dataarray, frequency, bin_widths, squeeze, time_dim=time_dim
        )

    try:
        grouped_data = dataarray.groupby(f"{time_dim}.{frequency}", squeeze=squeeze)
    except AttributeError:
        raise ValueError(
            f"Invalid frequency '{frequency}' - see xarray documentation for "
            f"a full list of valid frequencies."
        )
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

    # Create rolling groups:
    data_rolling = dataarray.rolling(**rolling_kwargs)

    in_built_how_methods = [
        method for method in dir(data_rolling) if not method.startswith("_")
    ]
    if how_reduce in in_built_how_methods:
        data_windowed = data_rolling.__getattribute__(how_reduce)(**reduce_kwargs)
    else:  # Check for tyto HOW methods
        data_windowed = data_rolling.reduce(HOW_DICT[how_reduce], reduce_kwargs)

    if how_dropna not in [None, "None", "none"]:
        for dim in window_dims:
            data_windowed = data_windowed.dropna(dim, how=how_dropna)

    data_windowed.attrs.update(dataarray.attrs)

    return data_windowed
