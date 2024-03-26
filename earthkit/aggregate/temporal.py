import logging
import typing as T
from copy import deepcopy

import xarray as xr

from earthkit.aggregate import tools
from earthkit.aggregate.general import reduce as g_reduce
from earthkit.aggregate.general import resample
from earthkit.aggregate.general import rolling_reduce as _rolling_reduce
from earthkit.aggregate.tools import groupby_time

logger = logging.getLogger(__name__)


@tools.time_dim_decorator
def reduce(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    *args,
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Reduce an xarray.dataarray or xarray.dataset along the time/date dimension using a specified `how` method.

    With the option to apply weights either directly or using a specified
    `weights` method.

    Parameters
    ----------
    dataarray : xr.DataArray or xr.Dataset
        Data object to reduce
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
        If you do not want to aggregate along the time dimension use earthkit.aggregate.reduce
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
        kwargs recognised by the how :func: `xr.reduce`

    Returns
    -------
        A data array with reduce dimensions removed.

    """
    reduce_dims = tools.ensure_list(kwargs.get("dim", []))
    if time_dim is not None and time_dim not in reduce_dims:
        reduce_dims.append(time_dim)
    kwargs["dim"] = reduce_dims

    return g_reduce(dataarray, *args, **kwargs)


@tools.time_dim_decorator
def rolling_reduce(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    window_length: int | None = None,
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """Return reduced data using a moving window over the time dimension.

    Parameters
    ----------
    dataarray : xr.DataArray or xr.Dataset
        Data over which the moving window is applied according to the reduction method.
    window_length :
        Length of window for the rolling groups along the time dimension.
        **see documentation for xarray.dataarray.rolling**.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
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
    windows : dict[str, int]
        Any other windows to apply to other dimensions in the dataset/dataarray
    **kwargs :
        Any kwargs that are compatible with the select `how_reduce` method.

    Returns
    -------
    xr.DataArray or xr.Dataset (as provided)
    """
    if window_length is not None:
        kwargs.update({time_dim: window_length})
    return _rolling_reduce(dataarray, **kwargs)


@tools.time_dim_decorator
def daily_reduce(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    how: str | T.Callable = "mean",
    time_dim: T.Union[str, None] = None,
    **kwargs,
):
    """
    Group data by day and reduce using the given how method.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, a earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an np.ndarray over an integer valued axis
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    # If time_dim in dimensions then use resample, this should be faster.
    #  At present, performance differences are small, but resampling can be improved by handling as
    #  a pandas dataframes. reample function should be updated to do this.
    #  NOTE: force_groupby is an undocumented kwarg for debug purposes
    if time_dim in dataarray.dims and not kwargs.pop("force_groupby", False):
        return resample(dataarray, frequency="D", dim=time_dim, how=how, **kwargs)

    # Otherwise, we groupby, with specifics set up for daily and handling both datetimes and timedeltas
    if dataarray[time_dim].dtype in ["<M8[ns]"]:  # datetime
        group_key = "date"
    elif dataarray[time_dim].dtype in ["<m8[ns]"]:  # timedelta
        group_key = "days"
    else:
        raise TypeError(f"Invalid type for time dimension ({time_dim}): {dataarray[time_dim].dtype}")

    grouped_data = groupby_time(dataarray, time_dim=time_dim, frequency=group_key)
    # If how is string and inbuilt method of grouped_data, we apply
    if isinstance(how, str) and how in dir(grouped_data):
        how_label = deepcopy(how)
        red_array = grouped_data.__getattribute__(how)(**kwargs)
    else:
        # If how is string, fetch function from dictionary:
        if isinstance(how, str):
            how_label = deepcopy(how)
            how = tools.get_how(how)
        assert isinstance(how, T.Callable), f"how method not recognised: {how}"

        red_array = grouped_data.reduce(how, **kwargs)

    # Update variable names, depends on dataset or dataarray format
    if isinstance(dataarray, (xr.Dataset)):
        red_array = red_array.rename(
            {**{data_arr: f"{data_arr}_{how_label}" for data_arr in red_array}, **{group_key: time_dim}}
        )
    else:
        red_array = red_array.rename(f"{red_array.name}_{how_label}", **{group_key: time_dim})

    # Revert group dimension name to time_dim
    red_array = red_array.rename()
    return red_array


def daily_mean(dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs):
    """
    Return the daily mean of the datacube.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    return daily_reduce(dataarray, *args, how="mean", **kwargs)


def daily_max(dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs):
    """
    Calculate the daily maximum.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    return daily_reduce(dataarray, *args, how="max", **kwargs)


def daily_min(dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs):
    """
    Calculate the daily minimum.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    return daily_reduce(dataarray, *args, how="min", **kwargs)


def daily_std(dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs):
    """
    Calculate the daily standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    return daily_reduce(dataarray, *args, how="std", **kwargs)


def daily_sum(dataarray: T.Union[xr.Dataset, xr.DataArray], *args, **kwargs):
    """
    Calculate the daily sum (accumulation).

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray containing a `time` dimension.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`reduce`.

    Returns
    -------
    xr.DataArray
    """
    return daily_reduce(dataarray, *args, how="sum", **kwargs)


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
    time_dim : str
        Name of the time dimension in the xarray object, default is `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
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
    time_dim : str
        Name of the time dimension in the xarray object, default is `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
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
    time_dim : str
        Name of the time dimension in the xarray object, default is `"time"`.
    time_shift : (optional) timedelta or dict
        A time shift to apply to the data prior to calculation, e.g. to change the local time zone.
        It can be provided as any object that can be understood by `pandas.Timedelta`, a dictonary is passed
        as kwargs to `pandas.Timedelta`
    **kwargs
        Keyword arguments to be passed to :func:`resample`.

    Returns
    -------
    xr.DataArray
    """
    return resample(dataarray, frequency="M", dim=time_dim, how="min", **kwargs)
