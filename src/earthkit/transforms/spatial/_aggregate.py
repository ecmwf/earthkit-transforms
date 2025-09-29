import logging
import typing as T
from copy import deepcopy

import geopandas as gpd
import pandas as pd
import xarray as xr
from earthkit.transforms._tools import (
    array_namespace_from_object,
    ensure_list,
    get_how_xp,
    get_spatial_info,
    standard_weights,
    transform_inputs_decorator,
)
from numpy import ndarray

logger = logging.getLogger(__name__)


def _transform_from_latlon(lat, lon):
    """Return an Affine transformation of input 1D arrays of lat and lon.

    This assumes that both lat and lon are regular and contiguous.

    Parameters
    ----------
    lat : list
        arrays or lists of latitude and longitude

    lon : list
        arrays or lists of latitude and longitude
    """
    from affine import Affine

    trans = Affine.translation(lon[0] - (lon[1] - lon[0]) / 2, lat[0] - (lat[1] - lat[0]) / 2)
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])

    return trans * scale


def rasterize(
    shape_list: T.List,
    coords: xr.core.coordinates.Coordinates,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    **kwargs,
) -> xr.DataArray:
    """Rasterize a list of geometries onto the given xarray coordinates.

    This only works for regular and contiguous latitude and longitude grids.

    Parameters
    ----------
    shape_list : affine.Affine
        List of geometries
    coords : xarray.coords
        Coordinates of dataarray to be masked
    lat_key :
        name of the latitude variables in the coordinates object
    lon_key :
        name of the longitude variables in the coordinates object
    dtype:
        datatype of the returned mask, default is `int`

    kwargs:
        Any other kwargs accepted by rasterio.features.rasterize

    Returns
    -------
    xr.DataArray
        A mask where points not inside the shape_list are set to `fill` value

    """
    from rasterio import features

    transform = _transform_from_latlon(coords[lat_key], coords[lon_key])
    out_shape = (len(coords[lat_key]), len(coords[lon_key]))
    raster = features.rasterize(shape_list, out_shape=out_shape, transform=transform, **kwargs)
    spatial_coords = {lat_key: coords[lat_key], lon_key: coords[lon_key]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(lat_key, lon_key))


def mask_contains_points(
    shape_list: T.List,
    coords: xr.core.coordinates.Coordinates,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    **_kwargs,
) -> xr.DataArray:
    """Return a mask array for the spatial points of data that lie within shapes in shape_list.

    Function uses matplotlib.Path so can accept a list of points, this is much faster than shapely.
    It was initially included for use with irregular data but has been constructed to also accept
    regular data and return in the same format as the rasterize function.

    Parameters
    ----------
    shape_list :
        List of geometries
    coords : xarray.coords
        Coordinates of dataarray to be masked
    lat_key:
        name of the latitude variables in the coordinates object
    lon_key:
        name of the longitude variables in the coordinates object
    dtype:
        datatype of the returned mask, default is `int`

    Returns
    -------
    xr.DataArray
        A mask where points not inside the shape_list are set to `fill` value

    """
    import matplotlib.path as mpltPath

    xp = array_namespace_from_object(coords[lat_key])

    lat_dims = coords[lat_key].dims
    lon_dims = coords[lon_key].dims
    # Assert that latitude and longitude have the same dimensions
    #   (irregular data, e.g. x,y or obs)
    # or the dimensions are themselves (regular data) but we will probably
    # just use the rasterize function for the regular case
    assert (lat_dims == lon_dims) or (lat_dims == (lat_key,) and lon_dims == (lon_key,))
    if lat_dims == (lat_key,) and lon_dims == (lon_key,):
        lon_full, lat_full = xp.meshgrid(
            coords[lon_key].data,
            coords[lat_key].data,
        )
    else:
        lon_full, lat_full = (
            coords[lon_key].data,
            coords[lat_key].data,
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
    outdata_shape = tuple(len(coords[dim]) for dim in spatial_dims)
    outdata = xp.full(outdata_shape, xp.nan)
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


def _geopandas_to_shape_list(geodataframe):
    """Iterate over rows of a geodataframe."""
    return [row[1]["geometry"] for row in geodataframe.iterrows()]


def _array_mask_iterator(mask_arrays):
    """Iterate over mask arrays."""
    for mask_array in mask_arrays:
        yield mask_array > 0


def _shape_mask_iterator(shapes, target, regular=True, **kwargs):
    """Iterate over shape mask methods."""
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)
    if regular:
        mask_function = rasterize
    else:
        mask_function = mask_contains_points
    for shape in shapes:
        shape_da = mask_function([shape], target.coords, **kwargs)
        yield shape_da


def shapes_to_masks(shapes: gpd.GeoDataFrame | list[gpd.GeoDataFrame], target, regular=True, **kwargs):
    """Create a list of masked dataarrays, if possible use the shape_mask_iterator.

    Parameters
    ----------
    shapes :
        A geodataframe or list of geodataframes containing the polygons for masks
    target :
        A dataarray to to create a mask for, only the geospatial coordinates are used

    regular :
        If True, data is on a regular grid so use rasterize method, if False use mask_contains_points
    all_touched :
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.

    kwargs:
        kwargs accepted by the masking methods, rasterize or mask_contains_points

    Returns
    -------
    list[xr.DataArray]
        A list of masks where points inside each geometry are 1, and those outside are xp.nan

    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)
    if regular:
        mask_function = rasterize
    else:
        mask_function = mask_contains_points

    return [mask_function([shape], target.coords, **kwargs) for shape in shapes]


def shapes_to_mask(shapes, target, regular=True, **kwargs):
    """Create a single masked dataarray based on all features in shapes.

    If possible use the shape_mask_iterator.

    Parameters
    ----------
    shapes :
        A geodataframe or list of geodataframes containing the polygons for masks
    target :
        A dataarray to to create a mask for, only the geospatial coordinates are used

    regular :
        If True, data is on a regular grid so use rasterize method,
        if False use mask_contains_points
    all_touched :
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.

    kwargs:
        kwargs accepted by the masking methods, rasterize or mask_contains_points

    Returns
    -------
    xr.DataArray
        A mask where points inside any geometry are 1, and those outside are xp.nan

    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)
    if regular:
        mask_function = rasterize
    else:
        mask_function = mask_contains_points

    return mask_function(shapes, target.coords, **kwargs)


def get_mask_dim_index(
    mask_dim: str | None | T.Dict[str, T.Any],
    geodataframe: gpd.geodataframe.GeoDataFrame,
    default_index_name: str = "index",
):
    if isinstance(mask_dim, str):
        if mask_dim in geodataframe:
            mask_dim_index = pd.Index(geodataframe[mask_dim])
        else:
            mask_dim_index = geodataframe.index.rename(mask_dim)
    elif isinstance(mask_dim, dict):
        assert len(mask_dim) == 1, "If provided as a dictionary, mask_dim should have only one key value pair"
        _mask_dim, _mask_dim_values = mask_dim.items()
        mask_dim_index = pd.Index(_mask_dim_values, name=_mask_dim)
    elif mask_dim is None:
        # Use the index of the data frame
        mask_dim_index = geodataframe.index
        if mask_dim_index.name is None:
            mask_dim_index = mask_dim_index.rename(default_index_name)
    else:
        raise ValueError("Unrecognised mask_dim format")

    return mask_dim_index


@transform_inputs_decorator()
def mask(
    dataarray: xr.Dataset | xr.DataArray,
    geodataframe: gpd.geodataframe.GeoDataFrame,
    mask_dim: str | None = None,
    lat_key: str | None = None,
    lon_key: str | None = None,
    chunk: bool = True,
    union_geometries: bool = False,
    **mask_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Apply multiple shape masks to some gridded data.

    Each feature in shape is treated as an individual mask to apply to
    data. The data provided is returned with an additional dimension equal in
    length to the number of features in the shape object, this can result in very
    large files which will slow down your script. It may be better to loop
    over individual features, or directly apply the mask with the shapes.reduce.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    mask_dim :
        dimension that will be created to accomodate the masked arrays, default is the index
        of the geodataframe
    all_touched :
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.
    lat_key :
        key for latitude variable, default behaviour is to detect variable keys.
    lon_key :
        key for longitude variable, default behaviour is to detect variable keys.
    chunk :  bool
        Boolean to indicate whether to use chunking, default = `True`.
        This is advised as spatial.masks can create large results. If you are working with small
        arrays, or you have implemented you own chunking rules you may wish to disable it.
    union_geometries : bool
        Boolean to indicate whether to union all geometries before masking.
        Default is `False`, which will apply each geometry in the geodataframe as a separate mask.
    mask_kwargs :
        Any kwargs to pass into the mask method

    Returns
    -------
    xr.Dataset | xr.DataArray
        A masked data array with dimensions [feautre_id] + [data.dims].
        Each slice of layer corresponds to a feature in layer.

    """
    spatial_info = get_spatial_info(dataarray, lat_key=lat_key, lon_key=lon_key)
    # Get spatial info required by mask functions:
    mask_kwargs = {**mask_kwargs, **{key: spatial_info[key] for key in ["lat_key", "lon_key", "regular"]}}
    mask_dim_index = get_mask_dim_index(mask_dim, geodataframe)

    if union_geometries:
        loop_masks = [shapes_to_mask(geodataframe, dataarray, **mask_kwargs)]
    else:
        loop_masks = _shape_mask_iterator(geodataframe, dataarray, **mask_kwargs)

    masked_arrays = []
    for this_mask in loop_masks:
        this_masked_array = dataarray.where(this_mask)
        if chunk:
            this_masked_array = this_masked_array.chunk()
        masked_arrays.append(this_masked_array.copy())

    if union_geometries:
        out = masked_arrays[0]
    else:
        # TODO: remove ignore type if xarray concat typing is updated
        out = xr.concat(masked_arrays, dim=mask_dim_index.name)  # type: ignore
        if chunk:
            out = out.chunk({mask_dim_index.name: 1})

        out = out.assign_coords({mask_dim_index.name: mask_dim_index})

    out.attrs.update(geodataframe.attrs)

    return out


@transform_inputs_decorator()
def reduce(
    dataarray: xr.Dataset | xr.DataArray,
    geodataframe: gpd.GeoDataFrame | None = None,
    mask_arrays: xr.DataArray | list[xr.DataArray] | None = None,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Apply a shape object to an xarray.DataArray object using the specified 'how' method.

    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    mask_arrays :
        precomputed mask array[s], if provided this will be used instead of creating a new mask.
        They must be on the same spatial grid as the dataarray.
    how :
        method used to apply mask. Default='mean', which calls xp.nanmean
    weights :
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key/lon_key :
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims :
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim :
        dimension that will be created after the reduction of the spatial dimensions, default is the index
        of the dataframe
    all_touched :
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.
    mask_kwargs :
        Any kwargs to pass into the mask method
    mask_arrays :
        precomputed mask array[s], if provided this will be used instead of creating a new mask.
        They must be on the same spatial grid as the dataarray.
    return_as :
        what format to return the data object, `pandas` or `xarray`. Work In Progress
    compact :
        If True, return a compact pandas.DataFrame with the reduced data as a new column.
        If False, return a fully expanded pandas.DataFrame.
        Only valid if return_as is `pandas`
    how_label :
        label to append to variable name in returned object, default is not to append
    kwargs :
        kwargs recognised by the how function

    Returns
    -------
    xr.Dataset | xr.DataArray
        A data array with dimensions `features` + `data.dims not in 'lat','lon'`.
        Each slice of layer corresponds to a feature in layer.

    """
    assert not (
        geodataframe is not None and mask_arrays is not None
    ), "Either a geodataframe or mask arrays must be provided, not both"
    if mask_arrays is not None:
        _mask_arrays: list[xr.DataArray] | None = ensure_list(mask_arrays)
    else:
        _mask_arrays = None

    if isinstance(dataarray, xr.Dataset):
        return_as: str = kwargs.pop("return_as", "xarray")
        if return_as in ["xarray"]:
            out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
            for var in dataarray.data_vars:
                out_da = _reduce_dataarray_as_xarray(
                    dataarray[var], geodataframe=geodataframe, mask_arrays=_mask_arrays, **kwargs
                )
                out_ds[out_da.name] = out_da
            return out_ds
        elif "pandas" in return_as:
            logger.warning(
                "Returning reduced data in pandas format is considered "
                "experimental and may change in future"
                "versions of earthkit"
            )
            if geodataframe is not None:
                out = geodataframe
                for var in dataarray.data_vars:
                    out = _reduce_dataarray_as_pandas(dataarray[var], geodataframe=out, **kwargs)
            else:
                out = None
                for var in dataarray.data_vars:
                    _out = _reduce_dataarray_as_pandas(dataarray[var], mask_arrays=_mask_arrays, **kwargs)
                    if out is None:
                        out = _out
                    else:
                        out = pd.merge(out, _out)
            return out
        else:
            raise TypeError("Return as type not recognised or incompatible with inputs")
    else:
        return _reduce_dataarray_as_xarray(
            dataarray, geodataframe=geodataframe, mask_arrays=_mask_arrays, **kwargs
        )


def _reduce_dataarray_as_xarray(
    dataarray: xr.DataArray,
    geodataframe: gpd.GeoDataFrame | None = None,
    mask_arrays: list[xr.DataArray] | None = None,
    how: T.Callable | str = "mean",
    weights: None | str | ndarray = None,
    lat_key: str | None = None,
    lon_key: str | None = None,
    extra_reduce_dims: list | str = [],
    mask_dim: str | None = None,
    how_label: str | None = None,
    squeeze: bool = True,
    all_touched: bool = False,
    mask_kwargs: dict[str, T.Any] = dict(),
    return_geometry_as_coord: bool = False,
    **reduce_kwargs,
) -> xr.DataArray:
    """Reduce an xarray.DataArray object over its geospatial dimensions using the specified 'how' method.

    If a geodataframe is provided the DataArray is reduced over each feature in the geodataframe.
    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    mask_arrays :
        precomputed mask array[s], if provided this will be used instead of creating a new mask.
        They must be on the same spatial grid as the dataarray.
    how :
        method used to apply mask. Default='mean', which calls xp.nanmean
    weights :
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key :
        key for latitude variable, default behaviour is to detect variable keys.
    lon_key :
        key for longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims :
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim :
        dimension that will be created after the reduction of the spatial dimensions, default = `"index"`
    return_as :
        what format to return the data object, `"pandas"` or `"xarray"`. Work In Progress
    how_label :
        label to append to variable name in returned object, default is `how`
    mask_kwargs :
        Any kwargs to pass into the mask method
    reduce_kwargs :
        kwargs recognised by the how function
    return_geometry_as_coord :
        include the geometries as a coordinate in the returned xarray object. WARNING: geometries are not
        serialisable objects, therefore this xarray will not be saveable as netCDF.
    all_touched :
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.
    squeeze :
        If True, squeeze the output xarray object, default is True

    Returns
    -------
    xr.DataArray
        A data array with dimensions [features] + [data.dims not in ['lat','lon']].
        Each slice of layer corresponds to a feature in layer

    """
    xp = array_namespace_from_object(dataarray)
    extra_out_attrs = {}
    how_str: None | str = None
    if weights is None:
        # convert how string to a method to apply
        if isinstance(how, str):
            how_str = deepcopy(how)
            how = reduce_how = get_how_xp(how, xp=xp)
        else:
            reduce_how = how
        assert callable(how), f"how must be a callable: {how}"
        if how_str is None:
            # get label from how method
            how_str = how.__name__
    else:
        # Create any standard weights, e.g. latitude.
        # TODO: handle kwargs better, currently only lat_key is accepted
        if isinstance(weights, str):
            _weights = standard_weights(dataarray, weights, lat_key=lat_key)
        else:
            _weights = weights
        # We ensure the callable is a string
        if callable(how):
            how = weighted_how = how.__name__
        else:
            weighted_how = how
        if how_str is None:
            how_str = how

    how_str = how_str or how_label
    new_long_name_components = [
        comp for comp in [how_str, dataarray.attrs.get("long_name", dataarray.name)] if comp is not None
    ]
    new_long_name = " ".join(new_long_name_components)
    extra_out_attrs.update({"long_name": new_long_name})
    new_short_name_components = [f"{comp}" for comp in [dataarray.name, how_label] if comp is not None]
    new_short_name = "_".join(new_short_name_components)

    if isinstance(extra_reduce_dims, str):
        extra_reduce_dims = [extra_reduce_dims]

    spatial_info = get_spatial_info(dataarray, lat_key=lat_key, lon_key=lon_key)
    # Get spatial info required by mask functions:
    mask_kwargs = {**mask_kwargs, **{key: spatial_info[key] for key in ["lat_key", "lon_key", "regular"]}}
    # All touched only valid for rasterize method
    if spatial_info["regular"]:
        mask_kwargs.setdefault("all_touched", all_touched)
    else:
        if all_touched:
            logger.warning("all_touched only valid for regular data, ignoring")
    spatial_dims = spatial_info.get("spatial_dims")

    reduce_dims = spatial_dims + extra_reduce_dims
    extra_out_attrs.update({"reduce_dims": reduce_dims})
    reduce_kwargs.update({"dim": reduce_dims})
    # If using a pre-computed mask arrays,
    # then iterator is just dataarray*mask_array
    if mask_arrays is not None:
        masked_data_list = _array_mask_iterator(mask_arrays)
    else:
        # If no geodataframe, then no mask, so create a dummy mask:
        if geodataframe is None:
            masked_data_list = [xr.ones_like(dataarray)]
        else:
            masked_data_list = _shape_mask_iterator(geodataframe, dataarray, **mask_kwargs)

    reduced_list = []
    for masked_data in masked_data_list:
        this = dataarray.where(masked_data, other=xp.nan)

        # If weighted, use xarray weighted arrays which
        # correctly handle missing values etc.
        if weights is not None:
            this_weighted = this.weighted(_weights)
            reduced_list.append(this_weighted.__getattribute__(weighted_how)(**reduce_kwargs))
        else:
            reduced = this.reduce(reduce_how, **reduce_kwargs).compute()
            reduced = reduced.assign_attrs(dataarray.attrs)
            reduced_list.append(reduced)

    if squeeze:
        reduced_list = [red_data.squeeze() for red_data in reduced_list]

    # If no geodataframe, there is just one reduced array
    if geodataframe is not None:
        mask_dim_index = get_mask_dim_index(mask_dim, geodataframe)
        out_xr = xr.concat(reduced_list, dim=mask_dim_index)
    elif mask_dim is None and len(reduced_list) == 1:
        out_xr = reduced_list[0]
    else:
        _concat_dim_name = mask_dim or "index"
        out_xr = xr.concat(reduced_list, dim=_concat_dim_name)

    out_xr = out_xr.rename(new_short_name)
    if geodataframe is not None:
        if return_geometry_as_coord:
            out_xr = out_xr.assign_coords(
                **{"geometry": (mask_dim_index.name, [_g for _g in geodataframe["geometry"]])}
            )
        out_xr = out_xr.assign_attrs({**geodataframe.attrs, **extra_out_attrs})

    return out_xr


def _reduce_dataarray_as_pandas(
    dataarray: xr.DataArray, geodataframe: gpd.GeoDataFrame | None = None, compact: bool = False, **kwargs
) -> pd.DataFrame:
    """Reduce an xarray.DataArray object over its geospatial dimensions using the specified 'how' method.

    If a geodataframe is provided the DataArray is reduced over each feature in the geodataframe.
    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe :
        Geopandas Dataframe containing the polygons for aggregations
    compact :
        If True, return a compact pandas.DataFrame with the reduced data as a new column.
        If False, return a fully expanded pandas.DataFrame.
    kwargs :
        kwargs accepted by the :function:_reduce_dataarray_as_xarray function

    Returns
    -------
    pd.DataFrame
        A pandas.DataFrame similar to the geopandas dataframe, with the reduced data
        added as a new column.

    """
    out_xr = _reduce_dataarray_as_xarray(dataarray, geodataframe=geodataframe, **kwargs)

    reduce_attrs = {f"{dataarray.name}": dataarray.attrs, f"{out_xr.name}": out_xr.attrs}

    if geodataframe is None:
        mask_dim = kwargs.get("mask_dim", "index")
        if mask_dim not in out_xr.dims:
            out_xr = xr.concat([out_xr], dim=mask_dim)
        # If no geodataframe, then just convert xarray to dataframe
        out = out_xr.to_dataframe()
        # Add attributes to the dataframe
        out.attrs.update({"reduce_attrs": reduce_attrs})
        return out

    # Otherwise, splice the geodataframe and reduced xarray
    reduce_attrs = {
        **geodataframe.attrs.get("reduce_attrs", {}),
        **reduce_attrs,
    }

    # TODO: somehow remove repeat call of get_mask_dim_index (see _reduce_dataarray_as_xarray)
    mask_dim_index = get_mask_dim_index(kwargs.get("mask_dim"), geodataframe)
    mask_dim_name = mask_dim_index.name
    out = geodataframe.set_index(mask_dim_index)
    if mask_dim_name not in out_xr.dims:
        out_xr = xr.concat([out_xr], dim=mask_dim_name)
    if not compact:  # Return as a fully expanded pandas.DataFrame
        # Convert to DataFrame
        out = out.join(out_xr.to_dataframe())
    else:
        # add the reduced data into a new column as a numpy array,
        # store the dim information in the attributes
        _out_dims = [str(dim) for dim in dataarray.coords if dim in out_xr.dims]
        out_dims = {dim: dataarray[dim].data for dim in _out_dims}
        reduce_attrs[f"{out_xr.name}"].update({"dims": out_dims})
        reduced_list = [
            out_xr.sel(**{mask_dim_name: mask_dim_value}).data
            for mask_dim_value in out_xr[mask_dim_name].data
        ]
        out = out.assign(**{f"{out_xr.name}": reduced_list})

    out.attrs.update({"reduce_attrs": reduce_attrs})

    return out
