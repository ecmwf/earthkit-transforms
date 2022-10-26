import numpy as np
import xarray as xr


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


def latitude_weights(latitudes, data_shape=None, lat_dims=0):
    """
    Function to return latitude weights. This is a very basic latitudinal
    weights function where weights are the normalised cosine of latitude.
    i.e. weight = cosine(latitude) / SUM(cosine(latitude))

    Args:
        latitude: numpy.array
            Latitude values to calculate weights
        data_shape (optional): list
            The shape of the data which the weights apply to,
            default is the shape of `latitudes`
        lat_dims (optional): integer or list
            The latitude dimension indices, default=0. If multi-dimensional then latitudes
            must be in the same order
    Returns:
        weights equal to cosine of latitude coordinate
    """
    # Calculate the weights
    weights = np.cos(np.radians(latitudes))
    weights = weights / np.nanmean(weights)

    if data_shape is None:
        data_shape = latitudes.shape
    ndims = len(data_shape)

    if not isinstance(lat_dims, list):
        lat_dims = list(lat_dims)

    i_w = 0
    w_shape = []
    for i_d in range(ndims):
        if i_d in lat_dims:
            w_shape.append(weights.shape[i_w])
            i_w += 1
        else:
            w_shape.append(1)

    # Is expanding to full data shape required?
    ones = np.ones(data_shape)
    return ones * weights.reshape(w_shape)


def _latitude_weights(dataarray: xr.DataArray, lat_dim_names=["latitude", "lat"]):
    """
    xarray.DataArray wrapper for latitude_weights which detects to the spatial dimensions
    latitude must be a coordinate of the dataarray
    """
    data_shape = dataarray.shape
    for lat in lat_dim_names:
        lat_array = dataarray.coords.get(lat, None)
        if lat_array is not None:
            break
    lat_dim_indices = [dataarray.dims.index(dim) for dim in lat_array.dims]
    return latitude_weights(
        lat_array.values, data_shape=data_shape, lat_dims=lat_dim_indices
    )


HOW_DICT = {
    # 'latitude_weighted_average': latitude_weighted_average,
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


# Libraries which are usable with reduce
ALLOWED_LIBS = {
    "numpy": "np",
}


# Dictionary containing recognised weight functions.
WEIGHT_DICT = {
    "latitude": _latitude_weights,
}
