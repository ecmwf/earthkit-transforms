import typing as T
from copy import deepcopy

import geopandas as gpd
import numpy as np
import xarray as xr

from earthkit.climate.tools import (
    WEIGHTS_DICT,
    get_dim_key,
    get_how,
    get_spatial_dims,
    nanaverage,
)


def transform_from_latlon(lat, lon):
    """
    Return an Affine transformation of input 1D arrays of lat and lon.

    This assumes that both lat and lon are regular and contiguous.

    Parameters
    ----------
    lat/lon : arrays or lists of latitude and longitude
    """
    from affine import Affine

    trans = Affine.translation(
        lon[0] - (lon[1] - lon[0]) / 2, lat[0] - (lat[1] - lat[0]) / 2
    )
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])

    return trans * scale


def rasterize(
    shape_list: T.List,
    coords: xr.core.coordinates.Coordinates,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    dtype: type = int,
    **kwargs,
):
    """
    Rasterize a list of geometries onto the given xarray coordinates.
    This only works for regular and contiguous latitude and longitude grids.

    Parameters
    ----------
    shape_list (affine.Affine): List of geometries
    coords (xarray.coords): Coordinates of dataarray to be masked

    lat_key/lon_key: name of the latitude/longitude variables in the coordinates object

    fill: value to fill points which are not within the shape_list, default is 0
    dtype: datatype of the returned mask, default is `int`

    kwargs: Any other kwargs accepted by rasterio.features.rasterize

    Returns
    -------
    xr.DataArray mask where points not inside the shape_list are set to `fill` value


    """
    from rasterio import features

    transform = transform_from_latlon(coords[lat_key], coords[lon_key])
    out_shape = (len(coords[lat_key]), len(coords[lon_key]))
    raster = features.rasterize(
        shape_list, out_shape=out_shape, transform=transform, dtype=dtype, **kwargs
    )
    spatial_coords = {lat_key: coords[lat_key], lon_key: coords[lon_key]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(lat_key, lon_key))


def mask_contains_points(shape_list, coords, lat_key="lat", lon_key="lon", **kwargs):
    """
    Return a mask array for the spatial points of data that lie within shapes in shape_list.


    Function uses matplotlib.Path so can accept a list of points,
    this is much faster than shapely.
    It was initially included for use with irregular data but has been
    constructed to also accept regular data and return in the same
    format as the rasterize function.
    """
    import matplotlib.path as mpltPath

    lat_dims = coords[lat_key].dims
    lon_dims = coords[lon_key].dims
    # Assert that latitude and longitude have the same dimensions
    #   (irregular data, e.g. x,y or obs)
    # or the dimensions are themselves (regular data) but we will probably
    # just use the rasterize function for the regular case
    assert (lat_dims == lon_dims) or (lat_dims == (lat_key,) and lon_dims == (lon_key,))
    if lat_dims == (lat_key,) and lon_dims == (lon_key,):
        lon_full, lat_full = np.meshgrid(
            coords[lon_key].values,
            coords[lat_key].values,
        )
    else:
        lon_full, lat_full = (
            coords[lon_key].values,
            coords[lat_key].values,
        )
    # convert lat lon pairs to to points:
    points = list(
        zip(
            lon_full.flat,
            lat_full.flat,
        )
    )

    # get spatial dims and create output array:
    spatial_dims = list(set(lat_dims + lon_dims))
    outdata_shape = [len(coords[dim]) for dim in spatial_dims]
    outdata = np.zeros(outdata_shape).astype(bool) * np.nan
    # loop over shapes and mask any point that is in the shape
    for shape in shape_list:
        for shp in shape[0]:
            shape_exterior = shp.exterior.coords.xy
            shape_exterior = list(
                zip(
                    list(shape_exterior[0]),  # longitudes
                    list(shape_exterior[1]),  # latitudes
                )
            )
            path = mpltPath.Path(shape_exterior)
            outdata.flat[path.contains_points(points)] = True

    out_coords = {coord: coords[coord] for coord in spatial_dims}
    outarray = xr.DataArray(outdata, coords=out_coords, dims=spatial_dims)

    return outarray


def geopandas_to_shape_list(geodataframe):
    """Iterate over rows of a geodataframe."""
    return [row[1]["geometry"] for row in geodataframe.iterrows()]


def _shape_mask_iterator(shapes, target, regular_grid=True, **kwargs):
    """Method which iterates over shape mask methods."""
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = geopandas_to_shape_list(shapes)
    if regular_grid:
        mask_function = rasterize
    else:
        mask_function = mask_contains_points
    for shape in shapes:
        shape_da = mask_function([shape], target.coords, **kwargs)
        yield shape_da


def shapes_to_mask(shapes, target, regular_grid=True, **kwargs):
    """
    Method which creates a list of mask dataarrays.

    If possible use the shape_mask_iterator.
    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = geopandas_to_shape_list(shapes)
    if regular_grid:
        mask_function = rasterize
    else:
        mask_function = mask_contains_points

    return [mask_function([shape], target.coords, **kwargs) for shape in shapes]


def masks(
    dataarray: T.Union[xr.DataArray, xr.Dataset],
    geodataframe: gpd.GeoDataFrame,
    mask_dim: str = "FID",
    # regular_grid: bool = True,
    **kwargs,
):
    """
    Apply multiple shape masks to some gridded data.

    Each feauture in shape is treated as an individual mask to apply to
    data. The data provided is returned with an additional dimension equal in
    length to the number of features in the shape object, this can result in very
    large files which will slow down your workflow. It may be better to loop
    over individual features, or directly apply the mask with the ct.shapes.average
    or ct.shapes.reduce functions.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    how :
        method used to apply mask. Default='mean', which calls np.nanmean
    weights :
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'

    Returns
    -------
        A masked data array with dimensions [feautre_id] + [data.dims].
        Each slice of layer corresponds to a feature in layer.
    """
    masked_arrays = []
    for mask in _shape_mask_iterator(geodataframe, dataarray, **kwargs):
        masked_arrays.append(dataarray.where(mask))

    if isinstance(mask_dim, str):
        mask_dim_values = geodataframe.get(
            mask_dim, np.arange(len(masked_arrays))
        ).to_numpy()
    elif isinstance(mask_dim, dict):
        assert (
            len(mask_dim) == 1
        ), "If provided as a dictionary, mask_dim should have onlly one key value pair"
        mask_dim, mask_dim_values = mask_dim.items()
    else:
        raise ValueError(
            "Unrecognised format for mask_dim, should be a string or length one dictionary"
        )

    out = xr.concat(masked_arrays, dim=mask_dim)
    out = out.assign_coords({mask_dim: mask_dim_values})

    out.attrs.update(geodataframe.attrs)

    return out


def reduce(
    dataarray: T.Union[xr.DataArray, xr.Dataset],
    geodataframe: gpd.GeoDataFrame,
    **kwargs,
):
    """
    Apply a shape object to an xarray.DataArray object using the specified 'how' method.

    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    how :
        method used to apply mask. Default='mean', which calls np.nanmean
    weights :
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key/lon_key :
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims :
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim :
        dimension that will be created after the reduction of the spatial dimensions, default = `FID`
    return_as :
        what format to return the data object, `pandas` or `xarray`. Work In Progress
    how_label :
        label to append to variable name in returned object, default is `how`
    kwargs :
        kwargs recognised by the how function

    Returns
    -------
        A data array with dimensions `features` + `data.dims not in 'lat','lon'`.
        Each slice of layer corresponds to a feature in layer.

    """
    if isinstance(dataarray, xr.DataArray):
        return _reduce_dataarray(dataarray, geodataframe, **kwargs)
    else:
        if kwargs.get("return_as", "pandas") in ["xarray"]:
            return xr.Dataset(
                [
                    _reduce_dataarray(dataarray[var], geodataframe, **kwargs)
                    for var in dataarray.data_vars
                ]
            )
        else:
            out = geodataframe
            for var in dataarray.data_vars:
                out = _reduce_dataarray(dataarray[var], geodataframe, **kwargs)
            return out


def _reduce_dataarray(
    dataarray: xr.DataArray,
    geodataframe: gpd.GeoDataFrame,
    how: T.Union[T.Callable, str] = nanaverage,
    weights: T.Union[None, str, np.ndarray] = None,
    lat_key: T.Union[None, str] = None,
    lon_key: T.Union[None, str] = None,
    extra_reduce_dims: T.Union[list, str] = [],
    mask_dim: str = "FID",
    return_as: str = "pandas",
    how_label: T.Union[str, None] = None,
    **kwargs,
):
    """
    Apply a shape object to an xarray.DataArray object using the specified 'how' method.

    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    how :
        method used to apply mask. Default='mean', which calls np.nanmean
    weights :
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key/lon_key :
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims :
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim :
        dimension that will be created after the reduction of the spatial dimensions, default = `"FID"`
    return_as :
        what format to return the data object, `"pandas"` or `"xarray"`. Work In Progress
    how_label :
        label to append to variable name in returned object, default is `how`
    kwargs :
        kwargs recognised by the how function

    Returns
    -------
        A data array with dimensions [features] + [data.dims not in ['lat','lon']].
        Each slice of layer corresponds to a feature in layer.

    """
    # If how is string, fetch function from dictionary:
    if isinstance(how, str):
        how_label = deepcopy(how)
        how = get_how(how)

    if isinstance(extra_reduce_dims, str):
        extra_reduce_dims = [extra_reduce_dims]

    assert isinstance(how, T.Callable), "how must be a callable"

    if lat_key is None:
        lat_key = get_dim_key(dataarray, "y")
    if lon_key is None:
        lon_key = get_dim_key(dataarray, "x")

    spatial_dims = get_spatial_dims(dataarray, lat_key, lon_key)

    # Create any standard weights, i.e. latitude
    if isinstance(weights, str):
        weights = WEIGHTS_DICT[weights](dataarray)

    red_kwargs = {}
    reduced_list = []
    for mask in _shape_mask_iterator(geodataframe, dataarray, **kwargs):
        this = dataarray.where(mask, other=np.nan)

        # If weighted, use xarray weighted arrays which correctly handle missing values etc.
        if weights is not None:
            dataarray.weighted(weights)

        reduced = this.reduce(
            how, dim=spatial_dims + extra_reduce_dims, **red_kwargs
        ).compute()
        reduced = reduced.assign_attrs(dataarray.attrs)
        reduced_list.append(reduced)
        # context.debug(f"Shapes.average reduced ({i}): {reduced} \n{i}")

    if isinstance(mask_dim, str):
        mask_dim_values = geodataframe.get(
            mask_dim, np.arange(len(reduced_list))
        ).to_numpy()
    elif isinstance(mask_dim, dict):
        assert (
            len(mask_dim) == 1
        ), "If provided as a dictionary, mask_dim should have onlly one key value pair"
        mask_dim, mask_dim_values = mask_dim.items()
    else:
        raise ValueError(
            "Unrecognised format for mask_dim, should be a string or length one dictionary"
        )

    # TODO: Maybe this could be handled more succinctly by making better use of xarray/pandas interoperability
    if return_as in ["xarray"]:
        out = xr.concat(reduced_list, dim=mask_dim)
        out = out.assign_coords(
            **{
                mask_dim: (mask_dim, mask_dim_values),
                "geometry": (mask_dim, [geom for geom in geodataframe["geometry"]]),
            }
        )
        out = out.assign_attrs(geodataframe.attrs)
    else:
        how_label = f"{dataarray.name}_{how_label or how.__name__}"
        if how_label in geodataframe:
            how_label += "_reduced"
        # If all dataarrays are single valued, convert to integer values
        if all([not red.shape for red in reduced_list]):
            reduced_list = [red.values for red in reduced_list]

        out = geodataframe.assign(**{how_label: reduced_list})
        out.attrs.update(dataarray.attrs)

    return out
