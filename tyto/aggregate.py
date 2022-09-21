
import xarray as xr


#: Mapping from pandas frequency strings to xarray time groups
_PANDAS_FREQUENCIES = {
    "D": "dayofyear",
    "W": "weekofyear",
    "M": "month",
}

#: The maximum limit of climatology time groups
_BIN_MAXES = {
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
    dim: str="time",
    how: str="mean",
    closed: str="left",
    label: str="left",
    skipna: bool=True,
    **kwargs,
) -> xr.DataArray:
    resample = dataarray.resample(
        label=label, closed=closed, skipna=skipna,
        **{dim: frequency}, **kwargs
    )
    result = resample.__getattribute__(how)(dim)
    return result

def groupby(
    dataarray: xr.DataArray,
    frequency: str=None,
    bin_widths: int=None,
    squeeze: bool=True,
) -> xr.core.groupby.DataArrayGroupBy:
    if frequency is None:
        try:
            frequency = xr.infer_freq(dataarray.time)
        except:
            raise ValueError(
                f"Unable to infer time frequency from data; please pass the "
                f"'frequency' argument explicitly"
            )
        frequency, possible_bins = _pandas_frequency_and_bins(frequency)
        bin_widths = bin_widths or possible_bins

    if bin_widths is not None:
        return _groupby_bins(dataarray, frequency, bin_widths, squeeze)

    try:
        grouped_data = dataarray.groupby(f"time.{frequency}", squeeze=squeeze)
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
) -> xr.core.groupby.DataArrayGroupBy:
    if not isinstance(bin_widths, (list, tuple)):
        max_value = _BIN_MAXES[frequency]
        bin_widths = list(range(0, max_value+1, bin_widths))
    try:
        grouped_data = dataarray.groupby_bins(
            f"time.{frequency}", bin_widths, squeeze=squeeze)
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
    bins = frequency[:-len(freq)] or None
    freq = _PANDAS_FREQUENCIES.get(freq.lstrip(" "), frequency)
    return freq, bins