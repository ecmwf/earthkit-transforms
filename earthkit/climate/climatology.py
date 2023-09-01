import typing as T

import xarray as xr

from earthkit.climate import aggregate, tools


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def mean(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    **reduce_kwargs,
) -> xr.DataArray:
    """
    Calculate the climatological mean.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate._groupby_time(
        dataarray,
        time_dim=time_dim,
        **groupby_kwargs,
    )
    return aggregate.reduce(grouped_data, how="mean", dim=time_dim, **reduce_kwargs)


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def stdev(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    **reduce_kwargs,
) -> xr.DataArray:
    """
    Calculate of the climatological standard deviation.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological standard deviation.
        Must contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate._groupby_time(
        dataarray,
        time_dim=time_dim,
        **groupby_kwargs,
    )
    return aggregate.reduce(grouped_data, how="std", dim=time_dim, **reduce_kwargs)


def median(dataarray: T.Union[xr.Dataset, xr.DataArray], **kwargs) -> xr.DataArray:
    """
    Calculate the climatological median.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological median. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    result = quantiles(dataarray, [0.5], **kwargs)
    return result.isel(quantile=0)


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def max(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    **reduce_kwargs,
) -> xr.DataArray:
    """
    Calculate the climatological maximum.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate._groupby_time(
        dataarray,
        time_dim=time_dim,
        **groupby_kwargs,
    )
    return aggregate.reduce(grouped_data, how="max", dim=time_dim, **reduce_kwargs)


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def min(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    **reduce_kwargs,
) -> xr.DataArray:
    """
    Calculate the climatological minimum.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate._groupby_time(
        dataarray,
        time_dim=time_dim,
        **groupby_kwargs,
    )
    return aggregate.reduce(grouped_data, how="min", dim=time_dim, **reduce_kwargs)


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def quantiles(
    dataarray: xr.DataArray,
    quantiles: list,
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    **reduce_kwargs,
) -> xr.DataArray:
    """
    Calculate a set of climatological quantiles.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological quantiles. Must
        contain a `time` dimension.
    quantiles : list
        The list of climatological quantiles to calculate.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate._groupby_time(
        dataarray.chunk({time_dim: -1}), time_dim=time_dim, **groupby_kwargs
    )
    results = []
    for quantile in quantiles:
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
    dataarray: xr.DataArray,
    percentiles: list,
    **kwargs,
) -> xr.DataArray:
    """
    Calculate a set of climatological percentiles.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological percentiles. Must
        contain a `time` dimension.
    percentiles : list
        The list of climatological percentiles to calculate.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    time_dim : str (optional)
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.aggregate.reduce` (except how)

    Returns
    -------
    xr.DataArray
    """
    quantiles = [p * 1e-2 for p in percentiles]
    quantile_data = quantiles(
        dataarray,
        quantiles,
        **kwargs,
    )
    result = quantile_data.assign_coords(percentile=("quantile", percentiles))
    result = result.swap_dims({"quantile": "percentile"})
    result = result.drop("quantile")
    return result


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def anomaly(
    dataarray: xr.DataArray,
    climatology: xr.DataArray = None,
    climatology_range: tuple = (None, None),
    time_dim: T.Union[str, None] = None,
    groupby_kwargs: dict = {},
    relative: bool = False,
    **reduce_kwargs,
):
    """
    Calculate the anomaly from a reference climatology.

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
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    relative : bool (optional)
        Return the relative anomaly, i.e. the percentage change w.r.t the climatological period
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.climatology.mean`

    Returns
    -------
    xr.DataArray
    """
    if climatology is None:
        if all(c_r is not None for c_r in climatology_range):
            selection = dataarray.sel(time=slice(*climatology_range))
        else:
            selection = dataarray
        climatology = mean(
            selection,
            groupby_kwargs=groupby_kwargs,
            **reduce_kwargs,
            time_dim=time_dim,
        )
    anomaly_array = (
        aggregate._groupby_time(dataarray, time_dim=time_dim, **groupby_kwargs)
        - climatology
    )
    if relative:
        anomaly_array = (
            aggregate._groupby_time(anomaly_array, time_dim=time_dim, **groupby_kwargs)
            / climatology
        )
        name_tag = "relative anomaly"
        update_attrs = {"units": "%"}
    else:
        name_tag = "anomaly"
        update_attrs = {}

    anomaly_array = aggregate.resample(
        anomaly_array, how="mean", **reduce_kwargs, **groupby_kwargs, dim=time_dim
    )
    update_attrs = {**dataarray.attrs, **update_attrs}
    if "standard_name" in update_attrs:
        update_attrs["standard_name"] += f"_{name_tag.replace(' ', '_')}"
    if "long_name" in anomaly_array.attrs:
        update_attrs["long_name"] += f" {name_tag}"

    anomaly_array = anomaly_array.assign_attrs(update_attrs)

    return anomaly_array


@tools.time_dim_decorator
@tools.groupby_kwargs_decorator
@tools.season_order_decorator
def relative_anomaly(dataarray: xr.DataArray, *args, **kwargs):
    """
    Calculate the relative anomaly from a reference climatology, i.e. percentage change.

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
        Name of the time dimension in the data object, default behviour is to detect the
        time dimension from the input object
    **reduce_kwargs :
        Any other kwargs that are accepted by `earthkit.climate.climatology.mean`

    Returns
    -------
    xr.DataArray
    """
    anomaly_xarray = anomaly(dataarray, *args, relative=True, **kwargs)

    return anomaly_xarray
