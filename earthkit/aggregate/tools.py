import functools
import typing as T

import numpy as np
import pandas as pd
import xarray as xr

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


def ensure_list(thing):
    if isinstance(thing, list):
        return thing
    try:
        return thing.to_list()
    except AttributeError:
        return [thing]


def time_dim_decorator(func):
    @functools.wraps(func)
    def wrapper(
        dataarray: T.Union[xr.Dataset, xr.DataArray],
        *args,
        time_dim: T.Union[str, None] = None,
        time_shift: T.Union[None, dict, str, pd.Timedelta] = None,
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
            time_coord = dataarray.coords[time_dim] + time_shift
            time_coord = time_coord.assign_attrs({"time_shift": f"{time_shift}"})

            dataarray = dataarray.assign_coords({time_dim: time_coord})

        return func(dataarray, *args, time_dim=time_dim, **kwargs)

    return wrapper


GROUPBY_KWARGS = ["frequency", "bin_widths", "squeeze"]


def groupby_kwargs_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, groupby_kwargs: dict = None, **kwargs):
        groupby_kwargs = groupby_kwargs or {}
        new_kwargs = {}
        for k, v in kwargs.copy().items():
            if k in GROUPBY_KWARGS:
                groupby_kwargs.setdefault(k, v)
            else:
                new_kwargs[k] = v
        return func(*args, groupby_kwargs=groupby_kwargs, **new_kwargs)

    return wrapper


def season_order_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if kwargs.get("frequency", "NOTseason") in ["season"]:
            result.reindex(season=["DJF", "MAM", "JJA", "SON"])
        return result

    return wrapper


# TODO: Replace with method from meteokit
def nanaverage(data, weights=None, **kwargs):
    """A merge of the functionality of np.nanmean and np.average.

    Parameters
    ----------
    data : numpy array
    weights: Weights to apply to the data for averaging.
            Weights will be normalised and must correspond to the
            shape of the numpy data array and axis/axes that is/are
            averaged over.
    axis: axis/axes to compute the nanaverage over.
    kwargs: any other np.nansum kwargs

    Returns
    -------
    numpy array
        mean of data (along axis) where nan-values are ignored
        and weights applied if provided.
    """
    if weights is not None:
        # set weights to nan where data is nan:
        this_weights = np.ones(data.shape) * weights
        this_weights[np.isnan(data)] = np.nan
        # Weights must be scaled to the sum of valid
        #  weights for each relevant axis:
        this_denom = np.nansum(this_weights, **kwargs)
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
        nanaverage = np.nansum(data * this_weights, **kwargs)
    else:
        # If no weights, then nanmean will suffice
        nanaverage = np.nanmean(data, **kwargs)

    return nanaverage


# TODO: Replace with method from meteokit
def latitude_weights(latitudes, data_shape=None, lat_dims=None):
    """Function to return latitude weights.

    This is a very basic latitudinal
    weights function where weights are the normalised cosine of latitude,
    i.e. weight = cosine(latitude) / SUM(cosine(latitude)).

    Parameters
    ----------
    latitudes: numpy.array
        Latitude values to calculate weights
    data_shape (optional): list
        The shape of the data which the weights apply to,
        default is the shape of `latitudes`
    lat_dims (optional): integer or list
        The dimension indices that corresponde to the latitude data,
        default is the shape of the latitudes array. If latitudes is a multi-dimensional,
        then order of latitudes must be in the same order as the lat_dims.

    Returns
    -------
    numpy.array
        weights equal to cosine of latitude coordinate in the shape of latitudes,
        or a user defined data_shape
    """
    # Calculate the weights
    weights = np.cos(np.radians(latitudes))
    weights = weights / np.nanmean(weights)

    if data_shape is None:
        data_shape = latitudes.shape
    ndims = len(data_shape)

    # Treat lat_dim as a list so we can handle irregular data
    if lat_dims is None:
        lat_dims = latitudes.shape
    elif isinstance(lat_dims, int):
        lat_dims = [lat_dims]

    # create shape for weights, where latitude dependant take
    # appropriate weight shape, where not fill with ones.
    i_w = 0
    w_shape = []
    for i_d in range(ndims):
        if i_d in lat_dims:
            w_shape.append(weights.shape[i_w])
            i_w += 1
        else:
            w_shape.append(1)

    ones = np.ones(data_shape)
    return ones * weights.reshape(w_shape)


def _latitude_weights(dataarray: xr.DataArray, lat_dim_names=["latitude", "lat"]):
    """
    xarray.DataArray wrapper for latitude_weights.

    Detects the spatial dimensions latitude must be a coordinate of the dataarray.
    """
    # data_shape = dataarray.shape
    for lat in lat_dim_names:
        lat_array = dataarray.coords.get(lat)
        if lat_array is not None:
            return np.cos(np.radians(lat_array.latitude))
    #         break
    # lat_dim_indices = [dataarray.dims.index(dim) for dim in lat_array.dims]
    # return latitude_weights(
    #     lat_array.values, data_shape=data_shape, lat_dims=lat_dim_indices
    # )


HOW_METHODS = {
    "average": nanaverage,
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
}

# Dictionary containing recognised weight functions.
WEIGHTS_DICT = {
    "latitude": _latitude_weights,
}


def get_how(how: str, how_methods=HOW_METHODS):
    try:
        how = how_methods[how]
    except KeyError:
        try:
            module, function = how.split(".")
            how = getattr(globals()[ALLOWED_LIBS[module]], function)
        except KeyError:
            raise ValueError(f"method must come from one of {ALLOWED_LIBS}")
        except AttributeError:
            raise AttributeError(f"module '{module}' has no attribute " f"'{function}'")
    return how


STANDARD_AXIS_KEYS = {
    "y": ["lat", "latitude"],
    "x": ["lon", "long", "longitude"],
    "t": ["time", "valid_time"],
}


def get_dim_key(
    dataarray: T.Union[xr.DataArray, xr.Dataset],
    axis: str,
):
    """Return the key of the dimension."""
    # First check if the axis value is in any dim:
    for dim in dataarray.dims:
        if "axis" in dataarray[dim].attrs and dataarray[dim].attrs["axis"].lower() == axis.lower():
            return dim

    # Then check if any dims match our "standard" axis
    for dim in dataarray.dims:
        if dim in STANDARD_AXIS_KEYS.get(axis.lower()):
            return dim

    # We have not been able to detect, so return the axis key
    return axis


def get_spatial_info(dataarray, lat_key=None, lon_key=None):
    # Figure out the keys for the latitude and longitude variables
    if lat_key is None:
        lat_key = get_dim_key(dataarray, "y")
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
