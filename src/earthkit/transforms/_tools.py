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

import functools
import importlib
import logging
import posixpath
import types
import typing as T

import numpy as np
import pandas as pd
import xarray as xr
from earthkit.utils.array import array_namespace

logger = logging.getLogger(__name__)


# Define a function to construct remote test data file paths
_REMOTE_TEST_DATA_URL = "https://sites.ecmwf.int/repository/earthkit-data/test-data/"


def earthkit_remote_test_data_file(*args):
    # Use POSIX-style joining to build URLs safely across platforms.
    # Strip leading slashes so that no argument is treated as absolute.
    parts = [str(a).lstrip("/") for a in args]
    return posixpath.join(_REMOTE_TEST_DATA_URL, *parts)


#: Mapping from pandas frequency strings to xarray time groups
_PANDAS_FREQUENCIES = {
    "D": "dayofyear",
    "W": "weekofyear",
    "M": "month",
    "ME": "month",
    "MS": "month",
    "H": "hour",
    "YE": "year",
}
# Note this is not 100% reversible, 3 pandas freqs map to xarray's "month",
# but "month" will only map to "MS"
_PANDAS_FREQUENCIES_R = {
    **{v: k for k, v in _PANDAS_FREQUENCIES.items()},
    # Add some additional aliases for reverse mapping
    **{"month_start": "MS", "month_end": "ME", "week": "W", "day": "D"},
}

#: The maximum limit of climatology time groups
_BIN_MAXES = {
    "hour": 24,
    "dayofyear": 366,
    "weekofyear": 53,
    "month": 12,
    "season": 4,
    "year": 1,
}


def ensure_list(thing) -> list[T.Any]:
    if isinstance(thing, list):
        return thing
    try:
        return thing.to_list()
    except AttributeError:
        return [thing]


def normalize_dims(dims: str | list[str] | None) -> list[str]:
    """Normalise a dimension argument to a list of strings.

    ``None`` is treated as "no dimensions" and returns an empty list.
    A bare string is wrapped in a list. Any other sequence is converted to a list.
    """
    if dims is None:
        return []
    if isinstance(dims, str):
        return [dims]
    return list(dims)


def time_dim_decorator(func):
    @functools.wraps(func)
    def wrapper(
        dataarray: xr.Dataset | xr.DataArray,
        *args,
        time_dim: str | None = None,
        time_shift: dict | str | pd.Timedelta | None = None,
        remove_partial_periods: bool = False,
        **kwargs,
    ):
        if time_dim is None:
            try:
                time_dim = get_dim_key(dataarray, "t")
            except Exception:
                # Not able to find time dimension in object so let fail its own way
                func(dataarray, *args, **kwargs)

        if time_shift is not None:
            # Create timedelta from dict
            if isinstance(time_shift, dict):
                time_shift = pd.Timedelta(**time_shift)
            else:
                time_shift = pd.Timedelta(time_shift)

            # Convert timedelta to timedelta64 (TODO: may need to be more robust here)
            time_coord = dataarray.coords[time_dim] + time_shift  # type: ignore
            time_coord = time_coord.assign_attrs({"time_shift": f"{time_shift}"})

            dataarray = dataarray.assign_coords({time_dim: time_coord})

        result = func(dataarray, *args, time_dim=time_dim, **kwargs)

        # If we want only full days then chop off the first and last day if we have time_shift
        if remove_partial_periods and time_shift and isinstance(result, (xr.Dataset, xr.DataArray)):
            return result.isel({time_dim: slice(1, -1)})

        return result

    return wrapper


GROUPBY_KWARGS = ["frequency", "bin_widths"]

_INVALID_CLIMATOLOGY_FREQUENCIES = ["day"]
VALID_CLIMATOLOGY_FREQUENCIES = ["dayofyear", "week", "weekofyear", "month"]


def groupby_kwargs_decorator(func: T.Callable | None = None, *, climatology: bool = False):
    """Collect groupby kwargs and optionally validates climatology frequencies.

    Can be used either as a simple decorator or as a decorator factory::

        @groupby_kwargs_decorator
        def f(...): ...

        @groupby_kwargs_decorator()
        def f(...): ...

        @groupby_kwargs_decorator(climatology=True)
        def f(...): ...

    Parameters
    ----------
    climatology : bool
        When True, raises a ValueError if the resolved ``frequency`` value is
        listed in ``_INVALID_CLIMATOLOGY_FREQUENCIES``.

    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, groupby_kwargs: dict | None = None, **kwargs):
            groupby_kwargs = groupby_kwargs or {}
            new_kwargs = {}
            for k, v in kwargs.copy().items():
                if k in GROUPBY_KWARGS:
                    groupby_kwargs.setdefault(k, v)
                else:
                    new_kwargs[k] = v
            if climatology:
                freq = groupby_kwargs.get("frequency")
                if freq in _INVALID_CLIMATOLOGY_FREQUENCIES:
                    valid = VALID_CLIMATOLOGY_FREQUENCIES + [None]
                    raise ValueError(
                        f"frequency={freq!r} is not accepted for climatology calculations. "
                        f"Please select one of: {valid}"
                    )
            return f(*args, groupby_kwargs=groupby_kwargs, **new_kwargs)

        return wrapper

    if func is None:
        # Called as a decorator factory: @groupby_kwargs_decorator(...) or groupby_kwargs_decorator(...)(func)
        return decorator

    # Called directly as a decorator: @groupby_kwargs_decorator or groupby_kwargs_decorator(func)
    return decorator(func)


def season_order_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if kwargs.get("frequency", "NOTseason") in ["season"]:
            result = result.reindex(season=["DJF", "MAM", "JJA", "SON"])
        return result

    return wrapper


def array_namespace_from_object(data_object: T.Any) -> types.ModuleType:
    """Attempt to infer the array namespace from the data object.

    Parameters
    ----------
    data_object : T.Any
        The data object from which to infer the array namespace.

    Returns
    -------
    types.ModuleType
        The inferred array namespace.

    Raises
    ------
    TypeError
        If the input data_object contains an incompatible array interface,
        e.g. a xarray.Dataset with mixed array namespaces.

    """
    if isinstance(data_object, xr.DataArray):
        return array_namespace(data_object.data)
    elif isinstance(data_object, xr.Dataset):
        data_vars = list(data_object.data_vars)
        xps = [array_namespace(data_object[var].data) for var in data_vars]
        if len(set(xps)) == 1:
            return xps[0]
        elif len(set(xps)) > 1:
            raise TypeError(
                "Data object contains variables with different array namespaces, "
                "cannot infer a single xp for computation."
            )
        else:
            raise TypeError("data_object must contain at least one data_variable to infer xp.")

    try:
        return array_namespace(data_object)
    except Exception:
        logger.warning(
            "Unable to infer array namespace from data_object, defaulting to numpy. "
            "If you are using a custom data object, please ensure it has a compatible array interface.",
        )
        return np


def nanaverage(data, weights=None, **kwargs):
    """Calculate the average of data ignoring nan-values.

    Parameters
    ----------
    data : array-like
        Data to average.
    weights : array-like or None, optional
        Weights to apply to the data for averaging.
        Weights will be normalised and must correspond to the
        shape of the data array and axis/axes that is/are
        averaged over.
    axis : int or tuple of int, optional
        axis/axes to compute the nanaverage over.
    kwargs:
        any other xp.nansum kwargs

    Returns
    -------
    Array mean of data (along axis) where nan-values are ignored
    and weights applied if provided.

    """
    xp = array_namespace_from_object(data)
    if weights is not None:
        # set weights to nan where data is nan:
        this_weights = xp.ones(data.shape) * weights
        this_weights[xp.isnan(data)] = xp.nan
        # Weights must be scaled to the sum of valid
        #  weights for each relevant axis:
        this_denom = xp.nansum(this_weights, **kwargs)
        # If averaging over an axis then we must add dummy
        # dimension[s] to the denominator to make compatible
        # with the weights.
        if kwargs.get("axis", None) is not None:
            reshape = list(this_weights.shape)
            reshape[kwargs.get("axis")] = 1
            this_denom = this_denom.reshape(reshape)

        # Scale weights to mean of valid weights:
        this_weights = this_weights / this_denom
        # Apply weights to data:
        _nanaverage = xp.nansum(data * this_weights, **kwargs)
    else:
        # If no weights, then nanmean will suffice
        _nanaverage = xp.nanmean(data, **kwargs)

    return _nanaverage


def standard_weights(dataarray: xr.DataArray, weights: str, **kwargs):
    """Return a weights DataArray for a recognised weighting method.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The data object for which to compute weights.
    weights : str
        Name of the weighting method. Currently accepted values: ``'latitude'`` or ``'lat'``.
    **kwargs
        Additional keyword arguments passed to the underlying weighting function
        (e.g. ``lat_key`` to override the detected latitude coordinate name).

    Returns
    -------
    xarray.DataArray
        Weights array compatible with the spatial coordinates of ``dataarray``.

    Raises
    ------
    NotImplementedError
        If ``weights`` is not a recognised weighting method.

    """
    if weights in ["latitude", "lat"]:
        lat_weight_kwargs = {key: value for key, value in kwargs.items() if key in ["lat_key"]}
        return latitude_weights(dataarray, **lat_weight_kwargs)

    raise NotImplementedError(f"The selected weights method is not recognised or implemented yet: {weights}.")


def latitude_weights(dataarray: xr.DataArray, lat_key: str | None = None):
    """Return cosine-of-latitude weights for the given DataArray.

    Detects the spatial dimensions; latitude must be a coordinate of the dataarray.

    Parameters
    ----------
    dataarray : xarray.DataArray
        The data object whose latitude coordinate is used to compute weights.
    lat_key : str, optional
        Name of the latitude coordinate. If not provided, the coordinate is
        detected automatically using CF conventions and known key names.

    Returns
    -------
    xarray.DataArray
        Array of cosine(latitude) weights with the same latitude coordinates
        as ``dataarray``.

    Raises
    ------
    KeyError
        If the latitude coordinate cannot be found in the dataarray.

    """
    if lat_key is None:
        lat_key = get_dim_key(dataarray, "y", check_coords=True)

    lat_array = dataarray.coords.get(lat_key)
    if lat_array is not None:
        xp = array_namespace_from_object(lat_array[lat_key])
        return xp.cos(xp.radians(lat_array[lat_key]))

    raise KeyError("Latitude variable name not detected or found in the dataarray. Please provide the correct key.")


HOW_METHODS = {
    "average": nanaverage,
    "nanaverage": nanaverage,
    "mean": np.nanmean,
    "stddev": np.nanstd,
    "std": np.nanstd,
    "stdev": np.nanstd,
    "sum": np.nansum,
    "max": np.nanmax,
    "min": np.nanmin,
    "median": np.nanmedian,
    "q": np.nanquantile,
    "quantile": np.nanquantile,
    "percentile": np.nanpercentile,
    "p": np.nanpercentile,
}

HOW_METHODS_MAPPING = {
    "average": "nanaverage",
    "mean": "nanmean",
    "stddev": "nanstd",
    "std": "nanstd",
    "stdev": "nanstd",
    "sum": "nansum",
    "max": "nanmax",
    "min": "nanmin",
    "median": "nanmedian",
    "q": "nanquantile",
    "quantile": "nanquantile",
    "percentile": "nanpercentile",
    "p": "nanpercentile",
}

WEIGHTED_HOW_METHODS = {
    "average": "mean",
    # "mean": "mean",
    "nanmean": "mean",
    "stddev": "std",
    # "std": "std",
    "stdev": "std",
    # "sum": "sum",
    # "sum_of_squares": "sum_of_squares",
    # "sum_of_weights": "sum_of_weights",
    "q": "quantile",
    # "quantile": "quantile",
    # "percentile": np.nanpercentile,
    # "p": np.nanpercentile,
}


# Libraries which are usable with reduce
ALLOWED_LIBS = {
    "numpy": "np",
    "cupy": "cp",
}

# Dictionary containing recognised weight functions.
WEIGHTS_DICT = {
    "latitude": latitude_weights,
    "lat": latitude_weights,
}


def get_how(how: str, how_methods=HOW_METHODS):
    try:
        how = how_methods[how]
    except KeyError:
        try:
            module, function = how.split(".")
        except Exception:
            raise ValueError(f"how method not recognised or found: how={how}")

        try:
            how = getattr(globals()[ALLOWED_LIBS[module]], function)
        except KeyError:
            raise ValueError(f"method must come from one of {ALLOWED_LIBS}")
        except AttributeError:
            raise AttributeError(f"module '{module}' has no attribute '{function}'")

    return how


def resolve_function_from_path(path: str) -> T.Callable:
    """Given a string like 'numpy.mean' or 'some.module.func', dynamically import module and return callable.

    Parameters
    ----------
    path : str
        Fully qualified path to a function (e.g., 'numpy.mean').

    Returns
    -------
    T.Callable
        The resolved function object.


    Raises
    ------
    ValueError
        If the path is invalid or the function/module is not found.

    """
    if "." not in path:
        raise ValueError(f"Invalid path '{path}'. Must be in the form 'module.func'.")

    module_path, func_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"Module '{module_path}' could not be imported: {e}") from e

    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        raise ValueError(f"Function '{func_name}' not found in module '{module_path}'.") from e

    if not callable(func):
        raise ValueError(f"Resolved attribute '{func_name}' in '{module_path}' is not callable.")

    return func


def get_how_xp(
    how_str: str,
    xp: types.ModuleType | None = None,
    how_methods_mapping: dict[str, str] = HOW_METHODS_MAPPING,
    data_object: T.Any = None,
) -> T.Callable:
    """Resolve a method name to a callable from the given module (xp), using an optional mapping for aliases.

    Parameters
    ----------
    how_str : str
        The method name or alias.

    xp : module, optional
        The array API module (e.g., numpy). Defaults to numpy.

    how_methods_mapping : dict
        Mapping of aliases to method names.

    data_object : Any, optional
        The data object to infer the array API from, if xp is not provided explicitly

    Returns
    -------
    T.Callable
        The resolved method.

    Raises
    ------
    ValueError
        If the method cannot be found in xp.

    """
    # First check if the "how_str" has a `.`, if so, we assume it is a full path to the method
    if "." in how_str:
        return resolve_function_from_path(how_str)

    resolved_name = how_methods_mapping.get(how_str, how_str)

    # Check if the resolved name is in this script's globals
    if resolved_name in globals():
        return globals()[resolved_name]

    if xp is None:
        if data_object is not None:
            try:
                xp = array_namespace_from_object(data_object)
            except Exception:
                logger.warning("Unable to infer array namespace from data_object, defaulting to numpy.")
                xp = np
        else:
            # Default to numpy if no xp or data_object is provided
            xp = np

    for name in (resolved_name, how_str):
        if hasattr(xp, name):
            return getattr(xp, name)

    raise ValueError(f"how method not recognised or found: how={how_str} for xp={xp.__name__}.")


STANDARD_AXIS_KEYS: dict[str, list[str]] = {
    "y": ["lat", "latitude"],
    "x": ["lon", "long", "longitude"],
    "t": ["time", "valid_time", "forecast_reference_time"],
    "realization": ["ensemble_member", "ensemble", "member", "number", "realization", "realisation"],
}

STANDARD_AXIS_CF_NAMES: dict[str, list[str]] = {
    "y": ["projection_y_coordinate", "latitude", "grid_latitude"],
    "x": ["projection_x_coordinate", "longitude", "grid_longitude"],
    "t": ["time", "valid_time", "forecast_reference_time"],
    "realization": ["realization"],
}


def get_dim_key(
    dataarray: xr.Dataset | xr.DataArray,
    axis: str,
    raise_error: bool = False,
    check_coords: bool = False,
) -> str:
    """Return the key of the dimension.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        The data to search for the dimension in.
    axis : str
        The axis to search for. This should be a CF standard axis key like 'x', 'y', 'z' or 't',
        or one of the earthkit-transforms recognised axis, e.g. "member".
    raise_error : bool
        If True, raises an error if the axis is not found in the dataarray.

    Returns
    -------
    str
        The key of the dimension in the dataarray that matches the axis.

    """
    # First check if the axis value is in any dim:
    for dim in dataarray.dims:
        if "axis" in dataarray[dim].attrs and dataarray[dim].attrs["axis"].lower() == axis.lower():
            return str(dim)

    # Then check if any dims have CF recognised standard names,
    #  Prioritised in order of the STANDARD_AXIS_CF_NAMES list order
    for standard_name in STANDARD_AXIS_CF_NAMES.get(axis.lower(), []):
        for dim in dataarray.dims:
            if dataarray[dim].attrs.get("standard_name") == standard_name:
                return str(dim)

    # Then check if any dims match our "standard" axis,
    #  Prioritised in order of the STANDARD_AXIS_KEYS list order
    for standard_axis_key in STANDARD_AXIS_KEYS.get(axis.lower(), []):
        if standard_axis_key in dataarray.dims:
            return standard_axis_key

    if check_coords:
        # Now repeat with coordinates
        # First check if the axis value is in any coord:
        for coord in dataarray.coords:
            if "axis" in dataarray[coord].attrs and dataarray[coord].attrs["axis"].lower() == axis.lower():
                return str(coord)

        # Then check if any dims have CF recognised standard names,
        #  Prioritised in order of the STANDARD_AXIS_CF_NAMES list order
        for standard_name in STANDARD_AXIS_CF_NAMES.get(axis.lower(), []):
            for dim in dataarray.coords:
                if dataarray[dim].attrs.get("standard_name") == standard_name:
                    return str(dim)

        # Then check if any dims match our "standard" axis,
        #  Prioritised in order of the STANDARD_AXIS_KEYS list order
        for standard_axis_key in STANDARD_AXIS_KEYS.get(axis.lower(), []):
            if standard_axis_key in dataarray.dims:
                return standard_axis_key

    # We have not been able to detect, so raise an error
    if raise_error:
        raise ValueError(f"Unable to find dimension key for axis '{axis}' in dataarray with dims: {dataarray.dims}.")

    return axis


def get_spatial_info(dataarray, lat_key=None, lon_key=None):
    """Return a dictionary of spatial metadata for a DataArray.

    Detects latitude and longitude coordinate names, their associated
    dimensions, and whether the grid is regular (1-D lat/lon coordinates)
    or irregular (shared dimensions, e.g. obs or curvilinear grids).

    Parameters
    ----------
    dataarray : xarray.DataArray or xarray.Dataset
        The data object to inspect.
    lat_key : str, optional
        Name of the latitude coordinate. If not provided, it is detected
        automatically.
    lon_key : str, optional
        Name of the longitude coordinate. If not provided, it is detected
        automatically.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``'lat_key'`` (str): name of the latitude coordinate.
        - ``'lon_key'`` (str): name of the longitude coordinate.
        - ``'regular'`` (bool): ``True`` if the grid is regular, ``False`` if irregular.
        - ``'spatial_dims'`` (list[str]): list of spatial dimension names.

    """
    if lat_key is None:
        lat_key = get_dim_key(dataarray, "y", check_coords=True)
    if lon_key is None:
        lon_key = get_dim_key(dataarray, "x")

    # Get the geospatial dimensions of the data. In the case of regular data this
    #  will be 'lat' and 'lon'. For irregular data it could be any dimensions
    lat_dims = dataarray.coords[lat_key].dims
    lon_dims = dataarray.coords[lon_key].dims
    spatial_dims = [dim for dim in lat_dims] + [dim for dim in lon_dims if dim not in lat_dims]

    # Assert that latitude and longitude have the same dimensions
    #   (irregular data, e.g. x&y or obs)
    # or the dimensions are themselves (regular data, 'lat'&'lon')
    assert (lat_dims == lon_dims) or ((lat_dims == (lat_key,)) and (lon_dims) == (lon_key,))
    if lat_dims == lon_dims:
        regular = False
    elif (lat_dims == (lat_key,)) and (lon_dims) == (lon_key,):
        regular = True
    else:
        raise ValueError(
            "The geospatial dimensions have not not been correctly detected:\n"
            f"lat_key: {lat_key}; lat_dims: {lat_dims}\n"
            f"lon_key: {lon_key}; lon_dims: {lon_dims}\n"
        )
    spatial_info = {
        "lat_key": lat_key,
        "lon_key": lon_key,
        "regular": regular,
        "spatial_dims": spatial_dims,
    }
    return spatial_info


def _pandas_frequency_and_bins(
    frequency: str,
) -> tuple[str, int | None]:
    freq = frequency.lstrip("0123456789")
    prefix = frequency[: -len(freq)] if len(freq) < len(frequency) else ""
    bins = int(prefix) if prefix else None
    freq = _PANDAS_FREQUENCIES.get(freq.lstrip(" "), frequency)
    return freq, bins


def groupby_time(
    dataarray: xr.Dataset | xr.DataArray,
    frequency: str | None = None,
    bin_widths: int | None = None,
    time_dim: str = "time",
):
    """Group a DataArray or Dataset by a time frequency.

    Parameters
    ----------
    dataarray : xarray.DataArray or xarray.Dataset
        The data object to group.
    frequency : str, optional
        The time frequency to group by (e.g. ``'month'``, ``'dayofyear'``,
        ``'weekofyear'``). If not provided, it is inferred from the data.
    bin_widths : int, optional
        Width of bins to use when grouping. If provided, :func:`groupby_bins`
        is used instead of a simple ``groupby``.
    time_dim : str, optional
        Name of the time dimension. Defaults to ``'time'``.

    Returns
    -------
    xr.core.groupby.DataArrayGroupBy or xr.core.groupby.DatasetGroupBy
        A grouped data object ready for reduction.

    Raises
    ------
    ValueError
        If the frequency cannot be inferred from the data or is not valid.

    """
    if frequency is None:
        try:
            frequency = xr.infer_freq(dataarray.time)
        except:  # noqa: E722
            raise ValueError(
                "Unable to infer time frequency from data; please pass the 'frequency' argument explicitly"
            )
        frequency, possible_bins = _pandas_frequency_and_bins(frequency)
        bin_widths = bin_widths or possible_bins
    _frequency = _PANDAS_FREQUENCIES.get(frequency, frequency)
    if bin_widths is not None:
        grouped_data = groupby_bins(dataarray, _frequency, bin_widths, time_dim=time_dim)
    else:
        try:
            grouped_data = dataarray.groupby(f"{time_dim}.{_frequency}")
        except AttributeError:
            raise ValueError(
                f"Invalid frequency '{frequency}' - see xarray documentation for a full list of valid frequencies."
            )

    return grouped_data


def groupby_bins(
    dataarray: xr.Dataset | xr.DataArray,
    frequency: str,
    bin_widths: list[int] | int = 1,
    time_dim: str = "time",
):
    """Group a DataArray or Dataset by binned time values.

    Parameters
    ----------
    dataarray : xarray.DataArray or xarray.Dataset
        The data object to group.
    frequency : str
        The time component to bin (e.g. ``'month'``, ``'dayofyear'``).
    bin_widths : list of int or int, optional
        If an ``int``, defines the uniform width of each bin (edges are
        generated from 0 to the maximum value for ``frequency``). If a list
        or tuple, it is used directly as the bin edges. Defaults to ``1``.
    time_dim : str, optional
        Name of the time dimension. Defaults to ``'time'``.

    Returns
    -------
    xr.core.groupby.DataArrayGroupBy or xr.core.groupby.DatasetGroupBy
        A grouped data object ready for reduction.

    Raises
    ------
    ValueError
        If the frequency is not valid.

    """
    if not isinstance(bin_widths, (list, tuple)):
        max_value = _BIN_MAXES[frequency]
        bin_widths = list(range(0, max_value + 1, bin_widths))
    try:
        grouped_data = dataarray.groupby_bins(f"{time_dim}.{frequency}", bin_widths)
    except AttributeError:
        raise ValueError(
            f"Invalid frequency '{frequency}' - see xarray documentation for a full list of valid frequencies."
        )
    return grouped_data


def timedelta_to_largest_unit(td: pd.Timedelta) -> str:
    """Convert a pandas Timedelta to a decimal value string in the largest non-zero unit."""
    # Define units in decreasing order
    units = [
        ("days", td.total_seconds() / 86400),  # 1 day = 86400 seconds
        ("hours", td.total_seconds() / 3600),
        ("minutes", td.total_seconds() / 60),
        ("seconds", td.total_seconds()),
    ]

    for unit_name, value in units:
        if abs(value) >= 1:
            # Round to 6 decimals to avoid floating-point noise
            value = round(value, 6)
            if np.isclose(value, 1):
                return unit_name
            if value.is_integer():
                value = int(value)
            return f"({value} {unit_name})"

    # Fallback (less than 1 second)
    milliseconds = td.total_seconds() * 1000
    return f"{round(milliseconds, 6)} milliseconds"
