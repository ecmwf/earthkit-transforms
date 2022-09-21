
import xarray as xr

from . import aggregate


def climate_mean(
    dataarray: xr.DataArray,
    frequency: str=None,
    bin_widths: int=None,
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
    
    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate.groupby(dataarray, frequency, bin_widths)
    return grouped_data.mean("time")


def climate_max(
    dataarray: xr.DataArray,
    frequency: str=None,
    bin_widths: int=None,
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
    
    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate.groupby(dataarray, frequency, bin_widths)
    return grouped_data.max("time")


def climate_min(
    dataarray: xr.DataArray,
    frequency: str=None,
    bin_widths: int=None,
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
    
    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate.groupby(dataarray, frequency, bin_widths)
    return grouped_data.min("time")


def climate_quantiles(
    dataarray: xr.DataArray,
    quantiles: list,
    frequency: str=None,
    bin_widths: int=None,
    skipna: bool=False,
    **kwargs,
) -> xr.DataArray:
    """
    Calculate a set of climatological quantiles.

    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray over which to calculate the climatological mean. Must
        contain a `time` dimension.
    quantiles : list
        The list of climatological quantiles to calculate.
    frequency : str (optional)
        Valid options are `day`, `week` and `month`.
    bin_widths : int or list (optional)
        If `bin_widths` is an `int`, it defines the width of each group bin on
        the frequency provided by `frequency`. If `bin_widths` is a sequence
        it defines the edges of each bin, allowing for non-uniform bin widths.
    
    Returns
    -------
    xr.DataArray
    """
    grouped_data = aggregate.groupby(dataarray, frequency, bin_widths)
    results = []
    for quantile in quantiles:
        results.append(
            grouped_data.quantile(
                q=quantile, dim="time", skipna=skipna, **kwargs,
            )
        )
    result = xr.concat(results, dim="quantile")
    return result
