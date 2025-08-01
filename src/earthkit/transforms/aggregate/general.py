import typing as T

import xarray as xr
from earthkit.transforms import tools

# Use numpy.ndarray for Array type hinting
from numpy import ndarray as Array


def how_label_rename(
    dataarray: xr.Dataset | xr.DataArray,
    how_label: str | None = None,
) -> xr.Dataset | xr.DataArray:
    if how_label is not None:
        # Update variable names, depends on dataset or dataarray format
        if isinstance(dataarray, xr.Dataset):
            renames = {data_arr: f"{data_arr}_{how_label}" for data_arr in dataarray}
            dataarray = dataarray.rename(renames)
        else:
            dataarray = dataarray.rename(f"{dataarray.name}_{how_label}")

    return dataarray


def _reduce_dataarray(
    dataarray: xr.DataArray,
    how: T.Callable | str = "mean",
    weights: None | str | Array = None,
    how_label: str | None = None,
    how_dropna: bool = False,
    xp: T.Any = None,
    **kwargs,
):
    """Reduce an xarray.dataarray or xarray.dataset using a specified `how` method.

    With the option to apply weights either directly or using a specified
    `weights` method.

    Parameters
    ----------
    dataarray : xr.DataArray
        Data object to reduce
    how: str or callable
        Method used to reduce data. Default='mean', which will implement the xarray in-built mean.
        If string, it must be an in-built xarray reduce method, a earthkit how method or any numpy method.
        In the case of duplicate names, method selection is first in the order: xarray, earthkit, numpy.
        Otherwise it can be any function which can be called in the form `f(x, axis=axis, **kwargs)`
        to return the result of reducing an xp.ndarray over an integer valued axis
    weights : str
        Choose a recognised method to apply weighting. Currently available methods are; 'latitude'
    how_dropna : str
        Choose how to drop nan values.
        Default is None and na values are preserved. Options are 'any' and 'all'.
    how_label : str
        Label to append to the name of the variable in the reduced object, default is nothing
    xp : T.Any
        The array namespace to use for the reduction. If None, it will be inferred from the dataarray.
    **kwargs :
        kwargs recognised by the how :func: `reduce`

    Returns
    -------
    A data array with reduce dimensions removed.

    """
    if xp is None:
        xp = tools.array_namespace_from_object(dataarray)
    # If weighted, use xarray weighted methods
    if weights is not None:
        # Create any standard weights, e.g. latitude
        if isinstance(weights, str):
            _weights = tools.standard_weights(dataarray, weights, **kwargs)
        else:
            _weights = weights
        # We ensure the callable is always a string
        if callable(how):
            how = weighted_how = how.__name__
        # map any alias methods:
        weighted_how = tools.WEIGHTED_HOW_METHODS.get(how, how)

        red_array = dataarray.weighted(_weights).__getattribute__(weighted_how)(**kwargs)

    else:
        # If how is string, fetch function from dictionary:
        if isinstance(how, str) and how in dir(dataarray):
            red_array = dataarray.__getattribute__(how)(**kwargs)
        else:
            if isinstance(how, str):
                how = tools.get_how_xp(how, xp=xp)
            assert callable(how), f"how method not recognised: {how}"

            red_array = dataarray.reduce(how, **kwargs)

    red_array = how_label_rename(red_array, how_label=how_label)

    if how_dropna:
        red_array = red_array.dropna(how_dropna)

    return red_array


def reduce(
    dataarray: xr.Dataset | xr.DataArray,
    *_args,
    **kwargs,
):
    """Reduce an xarray.dataarray or xarray.dataset using a specified `how` method.

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
        to return the result of reducing an xp.ndarray over an integer valued axis
    weights : str
        Choose a recognised method to apply weighting. Currently available methods are; 'latitude'
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
    # handle how as arg or kwarg
    kwargs["how"] = _args[0] if _args else kwargs.get("how", "mean")

    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _reduce_dataarray(dataarray[var], **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds

    out = _reduce_dataarray(dataarray, **kwargs)
    return out


def rolling_reduce(dataarray: xr.Dataset | xr.DataArray, *_args, **kwargs) -> xr.Dataset | xr.DataArray:
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
            out_da = _rolling_reduce_dataarray(dataarray[var], *_args, **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _rolling_reduce_dataarray(dataarray, *_args, **kwargs)


def _rolling_reduce_dataarray(
    dataarray: xr.DataArray, how_reduce="mean", how_dropna=None, chunk=True, **kwargs
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
        all NaN. **how_dropna** can be 'None', 'any' or 'all'. Default is None.
    chunk: bool
        If True, the dataarray is chunked before the rolling operation.
    **kwargs :
        Any kwargs that are compatible with the select `how_reduce` method.

    Returns
    -------
    xr.DataArray
    """
    xp = tools.array_namespace_from_object(dataarray)

    if chunk:
        dataarray = dataarray.chunk()
    # Expand dim kwarg to individual kwargs
    if isinstance(kwargs.get("dim"), dict):
        kwargs.update(kwargs.pop("dim"))

    window_dims = [str(_dim) for _dim in list(dataarray.dims) if _dim in list(kwargs)]
    rolling_kwargs_keys = ["min_periods", "center"] + window_dims
    rolling_kwargs_keys = [_kwarg for _kwarg in kwargs if _kwarg in rolling_kwargs_keys]
    rolling_kwargs = {_kwarg: kwargs.pop(_kwarg) for _kwarg in rolling_kwargs_keys}

    # Any kwargs left after above reductions are kwargs for reduction method
    reduce_kwargs = kwargs
    # Create rolling groups:
    data_rolling = dataarray.rolling(**rolling_kwargs)

    reduce_kwargs.setdefault("how", how_reduce)
    # TODO: remove type ignore when xarray puts types in stable location
    data_windowed = _reduce_dataarray(data_rolling, xp=xp, **reduce_kwargs)  # type: ignore

    data_windowed = _dropna(data_windowed, window_dims, how_dropna)

    data_windowed.attrs.update(dataarray.attrs)

    return data_windowed


def _dropna(data, dims, how):
    """Drop nan values from data."""
    if how in [None, "None", "none"]:
        return data

    for dim in dims:
        data = data.dropna(dim, how=how)
    return data


@tools.time_dim_decorator
def resample(
    dataarray: xr.Dataset | xr.DataArray,
    frequency: str | int | float,
    time_dim: str = "time",
    how: str = "mean",
    skipna: bool = True,
    how_args: list[T.Any] = [],
    how_kwargs: dict[str, T.Any] = {},
    how_label: str | None = None,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Resample dataarray to a user-defined frequency using a user-defined "how" method.

    Parameters
    ----------
    dataarray : xr.DataArray
        DataArray to be resampled.
    frequency : str, int, float
        The frequency at which to resample the chosen dimension. The format must be applicable
        to the chosen dimension.
    time_dim: str
        The dimension to resample along, default is `time`
    how: str
        The reduction method for resampling, default is `mean`
    how_label : str
        Label to append to the name of the variable in the reduced object, default is nothing
    skipna : bool
        If True, exclude missing values (na values) from the reduction.
    how_args : list
        List of arguments to be passed to the reduction method.
    how_kwargs : dict
        Dictionary of keyword arguments to be passed to the reduction method.
    **kwargs
        Keyword arguments to be passed to :func:`resample`. Defaults have been set as:
        `{"skipna": True}`

    Returns
    -------
    xr.Dataset | xr.DataArray
    """
    # Handle legacy API instances:
    time_dim = kwargs.pop("dim", time_dim)

    # Get any how kwargs into appropriate dictionary:
    for _k in ["q", "p"]:
        if _k in kwargs:
            how_kwargs[_k] = kwargs.pop(_k)
    # Translate and xarray frequencies to pandas language:
    frequency = tools._PANDAS_FREQUENCIES_R.get(frequency, frequency)
    kwargs[time_dim] = frequency
    resample = dataarray.resample(skipna=skipna, **kwargs)
    result = resample.__getattribute__(how)(*how_args, dim=time_dim, **how_kwargs)

    result = how_label_rename(result, how_label=how_label)

    return result
