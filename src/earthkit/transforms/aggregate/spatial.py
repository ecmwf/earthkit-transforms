import logging
import typing as T
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from earthkit.transforms.tools import ensure_list, get_how, get_spatial_info, standard_weights
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def _transform_from_latlon(lat, lon):
    """Compute an Affine transformation matrix from latitude and longitude arrays.

    This function assumes that latitude and longitude values are regularly spaced
    and contiguous.

    Parameters
    ----------
    lat : array-like
        A 1D array or list of latitude values, sorted in descending order.
    lon : array-like
        A 1D array or list of longitude values, sorted in ascending order.

    Returns
    -------
    Affine
        An affine transformation matrix for the given lat/lon grid.

    Raises
    ------
    ValueError
        If lat or lon has fewer than two elements.
        If lat or lon is not a 1D array.
    """
    from affine import Affine

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Latitude and longitude must be 1D arrays or lists.")
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("Latitude and longitude arrays must contain at least two values.")

    # Compute translation (shifting origin to the corner of the first pixel)
    trans = Affine.translation(lon[0] - (lon[1] - lon[0]) / 2, lat[0] - (lat[1] - lat[0]) / 2)
    # Compute scaling (distance per pixel)
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])

    return trans * scale


def rasterize(
    shape_list: T.List[T.Tuple[T.Any, T.Any]],  # List of (geometry, value) tuples
    coords: xr.core.coordinates.DataArrayCoordinates,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    dtype: T.Type = int,
    **kwargs,
) -> xr.DataArray:
    """Rasterize a list of geometries onto the given xarray coordinates.

    This only works for regular and contiguous latitude and longitude grids.

    Parameters
    ----------
    shape_list : List[Tuple[geometry, value]]
        A list of geometries and associated values to rasterize.
    coords : xarray.core.coordinates.DataArrayCoordinates
        The coordinates of the DataArray to be masked. Must contain `lat_key` and `lon_key`.
    lat_key : str, optional
        The name of the latitude variable in `coords`, default is `"latitude"`.
    lon_key : str, optional
        The name of the longitude variable in `coords`, default is `"longitude"`.
    dtype : type, optional
        The data type of the returned mask, default is `int`.
    **kwargs
        Additional keyword arguments passed to `rasterio.features.rasterize`.

    Returns
    -------
    xr.DataArray
        A rasterized mask where points outside `shape_list` are set to `fill`.

    Raises
    ------
    ValueError
        If `lat_key` or `lon_key` are not present in `coords`, or if they are not 1D arrays.
    """
    from rasterio import features

    if lat_key not in coords or lon_key not in coords:
        raise ValueError(f"Coordinates must contain '{lat_key}' and '{lon_key}'.")

    lat, lon = coords[lat_key], coords[lon_key]

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            f"Latitude and longitude must be 1D arrays. Found lat.ndim={lat.ndim}, lon.ndim={lon.ndim}."
        )

    # Compute affine transformation
    transform = _transform_from_latlon(lat, lon)
    out_shape = (len(lat), len(lon))

    # Rasterize geometries
    raster = features.rasterize(shape_list, out_shape=out_shape, transform=transform, dtype=dtype, **kwargs)

    # Create xarray DataArray with spatial coordinates
    spatial_coords = {lat_key: lat, lon_key: lon}
    return xr.DataArray(raster, coords=spatial_coords, dims=(lat_key, lon_key))


def mask_contains_points(
    shape_list: T.List[T.List[Polygon]],
    coords: xr.core.coordinates.DataArrayCoordinates,
    lat_key: str = "lat",
    lon_key: str = "lon",
    fill: T.Any = np.nan,
) -> xr.DataArray:
    """Generate a mask where spatial points inside shape_list are marked.

    This function uses `matplotlib.Path`, which is significantly faster than `shapely`
    for checking point containment. It supports both irregular and regular latitude-longitude grids.

    Parameters
    ----------
    shape_list : List[List[shapely.geometry.Polygon]]
        A list of lists containing shapely Polygon geometries.
    coords : xarray.core.coordinates.DataArrayCoordinates
        The coordinates of the DataArray to be masked.
    lat_key : str, optional
        The name of the latitude variable in `coords`, default is `"lat"`.
    lon_key : str, optional
        The name of the longitude variable in `coords`, default is `"lon"`.
    fill : Any, optional
        The value to fill for points not covered by any shape, default is `np.nan`.

    Returns
    -------
    xr.DataArray
        A mask where points inside any shape are `True`, others are set to `fill`.

    Raises
    ------
    ValueError
        If `lat_key` or `lon_key` are missing or improperly formatted.
    """
    import matplotlib.path as mpltPath

    if lat_key not in coords or lon_key not in coords:
        raise ValueError(f"Coordinates must contain '{lat_key}' and '{lon_key}'.")

    lat_dims, lon_dims = coords[lat_key].dims, coords[lon_key].dims

    # Validate that the grid is regular or irregular
    if lat_dims != lon_dims and not (lat_dims == (lat_key,) and lon_dims == (lon_key,)):
        raise ValueError(f"Inconsistent coordinate dimensions: lat={lat_dims}, lon={lon_dims}.")

    # Generate a full coordinate grid if working with a regular grid
    if lat_dims == (lat_key,) and lon_dims == (lon_key,):
        lon_full, lat_full = np.meshgrid(coords[lon_key].values, coords[lat_key].values)
    else:
        lon_full, lat_full = coords[lon_key].values, coords[lat_key].values

    # Convert lat/lon pairs to (x, y) points
    points = np.column_stack((lon_full.ravel(), lat_full.ravel()))

    # Create output mask initialized with fill value
    spatial_dims = list(set(lat_dims + lon_dims))
    outdata_shape = [len(coords[dim]) for dim in spatial_dims]
    outdata = np.full(outdata_shape, fill, dtype=bool)

    # Check containment for each shape and update the mask
    for shape_group in shape_list:
        for shape in shape_group:
            path = mpltPath.Path(np.column_stack(shape.exterior.coords.xy))
            mask = path.contains_points(points).reshape(outdata.shape)
            outdata[mask] = True  # Mark inside-shape points

    # Create output xarray DataArray
    out_coords = {dim: coords[dim] for dim in spatial_dims}
    return xr.DataArray(outdata, coords=out_coords, dims=spatial_dims)


def _geopandas_to_shape_list(geodataframe):
    """Iterate over rows of a geodataframe."""
    return [row[1]["geometry"] for row in geodataframe.iterrows()]


def _array_mask_iterator(mask_arrays):
    """Iiterate over mask arrays."""
    for mask_array in mask_arrays:
        yield mask_array > 0


def _shape_mask_iterator(
    shapes: T.Union[gpd.GeoDataFrame, list], target: xr.DataArray, regular: bool = True, **kwargs
) -> T.Generator[xr.DataArray, None, None]:
    """Iterate over shape mask methods and yield masks as DataArrays.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame | list
        A GeoDataFrame or a list of shapes containing the polygon geometries.
    target : xr.DataArray
        The target DataArray whose spatial coordinates will be used for masking.
    regular : bool, optional
        If `True`, assumes data is on a regular grid and uses `rasterize`.
        If `False`, uses `mask_contains_points` (default: `True`).
    **kwargs :
        Additional arguments passed to the masking function.

    Yields
    ------
    xr.DataArray
        A DataArray mask for each shape in `shapes`.
    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)

    mask_function = rasterize if regular else mask_contains_points

    yield from (mask_function([shape], target.coords, **kwargs) for shape in shapes)


def shapes_to_masks(
    shapes: T.Union[gpd.GeoDataFrame, T.List[gpd.GeoDataFrame]],
    target: xr.DataArray,
    regular: bool = True,
    **kwargs,
) -> T.List[xr.DataArray]:
    """Generate a list of masks from given shapes for the target data's geospatial coordinates.

    This function creates masks where points inside each geometry are set to `1` and
    points outside are set to `np.nan`. If the data is on a regular grid, it uses `rasterize()`;
    otherwise, it falls back to `mask_contains_points()`.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame | List[gpd.GeoDataFrame]
        A GeoDataFrame or list of GeoDataFrames containing polygon geometries.
    target : xr.DataArray
        The data array for which the mask will be created (only its spatial coordinates are used).
    regular : bool, optional
        - If `True` (default), assumes data is on a regular grid and uses `rasterize()`.
        - If `False`, uses `mask_contains_points()`, which is better for irregular grids.
    all_touched : bool, optional
        If `True`, all pixels touched by geometries are considered inside the shape.
        Default is `False`. Only valid when `regular=True`.
    **kwargs
        Additional arguments passed to either `rasterize()` (for regular grids) or
        `mask_contains_points()` (for irregular grids).

    Returns
    -------
    List[xr.DataArray]
        A list of masks, where points inside each shape are `1` and points outside are `np.nan`.

    Notes
    -----
    - If `shapes` is a single GeoDataFrame, it is converted into a list of shapes.
    - If `shapes` is empty or contains no valid geometries, an empty list is returned.
    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)

    if not shapes:
        return []

    mask_function = rasterize if regular else mask_contains_points

    return [mask_function([shape], target.coords, **kwargs) for shape in shapes]


def shapes_to_mask(
    shapes: T.Union[gpd.GeoDataFrame, T.List[gpd.GeoDataFrame]],
    target: xr.DataArray,
    regular: bool = True,
    **kwargs,
) -> xr.DataArray:
    """Create a single mask DataArray based on all features in `shapes`.

    This function applies the given shapes as a mask over the target's spatial coordinates.
    If the data is on a regular grid, it uses `rasterize()`, otherwise it falls back
    to `mask_contains_points()`.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame | List[gpd.GeoDataFrame]
        A GeoDataFrame or list of GeoDataFrames containing polygon geometries.
    target : xr.DataArray
        The data array for which the mask will be created (only its spatial coordinates are used).
    regular : bool, optional
        - If `True` (default), assumes data is on a regular grid and uses `rasterize()`.
        - If `False`, uses `mask_contains_points()`, which is better for irregular grids.
    all_touched : bool, optional
        If `True`, all pixels touched by geometries are considered inside the shape.
        Default is `False`. Only valid when `regular=True`.
    **kwargs
        Additional arguments passed to either `rasterize()` (for regular grids) or
        `mask_contains_points()` (for irregular grids).

    Returns
    -------
    xr.DataArray
        A mask where points inside any geometry are `1`, and those outside are `np.nan`.

    Notes
    -----
    - If `shapes` is empty or contains no valid geometries, returns an empty mask filled with `np.nan`.
    - If `shapes` is a single GeoDataFrame, it is converted into a list of shapes.
    """
    if isinstance(shapes, gpd.GeoDataFrame):
        shapes = _geopandas_to_shape_list(shapes)

    if not shapes:  # Handle empty input
        empty_mask = np.full(target.shape, np.nan, dtype=float)
        return xr.DataArray(empty_mask, coords=target.coords, dims=target.dims)

    mask_function = rasterize if regular else mask_contains_points

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


def masks(
    dataarray: xr.Dataset | xr.DataArray,
    geodataframe: gpd.geodataframe.GeoDataFrame,
    *args,
    **kwargs,
):
    logger.warning(
        "earthkit.transforms.aggregate.spatial.masks is deprecated, "
        "please use earthkit.transforms.aggregate.spatial.mask instead."
    )
    return mask(dataarray, geodataframe, *args, **kwargs)


def mask(
    dataarray: T.Union[xr.Dataset, xr.DataArray],
    geodataframe: gpd.GeoDataFrame,
    mask_dim: T.Optional[str] = None,
    lat_key: T.Optional[str] = None,
    lon_key: T.Optional[str] = None,
    chunk: bool = True,
    union_geometries: bool = False,
    **mask_kwargs,
) -> T.Union[xr.Dataset, xr.DataArray]:
    """Apply multiple shape masks to gridded data.

    Each feature in `geodataframe` is treated as an individual mask applied to `dataarray`.
    The returned object has an additional dimension corresponding to the number of features,
    unless `union_geometries=True`, in which case a single mask is applied.

    Parameters
    ----------
    dataarray : xr.Dataset | xr.DataArray
        The input dataset or data array, which must have geospatial coordinates.
    geodataframe : gpd.GeoDataFrame
        A GeoDataFrame containing polygon geometries used for masking.
    mask_dim : str, optional
        The name of the new dimension that will store the masked arrays.
        Defaults to the index of `geodataframe`.
    lat_key : str, optional
        The name of the latitude coordinate in `dataarray`. Auto-detected if `None`.
    lon_key : str, optional
        The name of the longitude coordinate in `dataarray`. Auto-detected if `None`.
    chunk : bool, optional
        If `True` (default), applies chunking to prevent large memory usage.
    union_geometries : bool, optional
        If `True`, all geometries in `geodataframe` are merged into a single mask.
        If `False` (default), each geometry is applied separately.
    **mask_kwargs :
        Additional arguments passed to the masking functions (`shapes_to_mask` or `_shape_mask_iterator`).

    Returns
    -------
    xr.Dataset | xr.DataArray
        A masked data array with dimensions `[feature_id] + dataarray.dims`, unless
        `union_geometries=True`, in which case only `dataarray.dims` are returned.

    Notes
    -----
    - If `geodataframe` is empty, returns an unmodified `dataarray` or raises an error.
    - If `union_geometries=True`, all geometries are combined before masking.
    """
    if geodataframe.empty:
        raise ValueError("The input GeoDataFrame is empty. No masking can be performed.")

    # Extract spatial metadata
    spatial_info = get_spatial_info(dataarray, lat_key=lat_key, lon_key=lon_key)
    mask_kwargs = {**mask_kwargs, **{key: spatial_info[key] for key in ["lat_key", "lon_key", "regular"]}}

    # Get masking function and dimension index
    mask_dim_index = get_mask_dim_index(mask_dim, geodataframe)

    if union_geometries:
        loop_masks: T.Iterable = [shapes_to_mask(geodataframe, dataarray, **mask_kwargs)]
    else:
        loop_masks: T.Iterable = _shape_mask_iterator(geodataframe, dataarray, **mask_kwargs)

    masked_arrays = []
    for this_mask in loop_masks:
        this_masked_array = dataarray.where(this_mask)
        if chunk:
            this_masked_array = this_masked_array.chunk()
        masked_arrays.append(this_masked_array.copy())

    if union_geometries:
        out = masked_arrays[0]  # Single mask applied
    else:
        out = xr.concat(masked_arrays, dim=mask_dim_index.name)
        if chunk:
            out = out.chunk({mask_dim_index.name: 1})
        out = out.assign_coords({mask_dim_index.name: mask_dim_index})

    out.attrs.update(geodataframe.attrs)

    return out


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
    mask_dim (optional):
        dimension that will be created to accomodate the masked arrays, default is the index
        of the geodataframe
    all_touched (optional):
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.
    lat_key/lon_key (optional):
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    chunk : (optional) bool
        Boolean to indicate whether to use chunking, default = `True`.
        This is advised as spatial.masks can create large results. If you are working with small
        arrays, or you have implemented you own chunking rules you may wish to disable it.
    union_geometries : (optional) bool
        Boolean to indicate whether to union all geometries before masking.
        Default is `False`, which will apply each geometry in the geodataframe as a separate mask.
    mask_kwargs (optional):
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
        out = xr.concat(masked_arrays, dim=mask_dim_index.name)
        if chunk:
            out = out.chunk({mask_dim_index.name: 1})

        out = out.assign_coords({mask_dim_index.name: mask_dim_index})

    out.attrs.update(geodataframe.attrs)

    return out


def reduce(
    dataarray: xr.Dataset | xr.DataArray,
    geodataframe: gpd.GeoDataFrame | None = None,
    mask_arrays: xr.DataArray | list[xr.DataArray] | None = None,
    *args,
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
    how (optional):
        method used to apply mask. Default='mean', which calls np.nanmean
    weights (optional):
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key/lon_key (optional):
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims (optional):
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim (optional):
        dimension that will be created after the reduction of the spatial dimensions, default is the index
        of the dataframe
    all_touched (optional):
        If True, all pixels touched by geometries will be considered in,
        if False, only pixels whose center is within. Default is False.
        Only valid for regular data.
    mask_kwargs (optional):
        Any kwargs to pass into the mask method
    mask_arrays (optional):
        precomputed mask array[s], if provided this will be used instead of creating a new mask.
        They must be on the same spatial grid as the dataarray.
    return_as (optional):
        what format to return the data object, `pandas` or `xarray`. Work In Progress
    how_label (optional):
        label to append to variable name in returned object, default is not to append
    kwargs (optional):
        kwargs recognised by the how function

    Returns
    -------
    xr.Dataset | xr.DataArray
        A data array with dimensions `features` + `data.dims not in 'lat','lon'`.
        Each slice of layer corresponds to a feature in layer.

    """
    if mask_arrays is not None:
        mask_arrays = ensure_list(mask_arrays)
    if isinstance(dataarray, xr.Dataset):
        return_as: str = kwargs.get("return_as", "xarray")
        if return_as in ["xarray"]:
            out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
            for var in dataarray.data_vars:
                out_da = _reduce_dataarray(
                    dataarray[var], geodataframe=geodataframe, mask_arrays=mask_arrays, *args, **kwargs
                )
                out_ds[out_da.name] = out_da
            return out_ds
        elif "pandas" in return_as:
            logger.warning(
                "Returning reduced data in pandas format is considered experimental and may change in future"
                "versions of earthkit"
            )
            if geodataframe is not None:
                out = geodataframe
                for var in dataarray.data_vars:
                    out = _reduce_dataarray(dataarray[var], geodataframe=out, *args, **kwargs)
            else:
                out = None
                for var in dataarray.data_vars:
                    _out = _reduce_dataarray(dataarray[var], mask_arrays=mask_arrays, *args, **kwargs)
                    if out is None:
                        out = _out
                    else:
                        out = pd.merge(out, _out)
            return out
        else:
            raise TypeError("Return as type not recognised or incompatible with inputs")
    else:
        return _reduce_dataarray(
            dataarray, geodataframe=geodataframe, mask_arrays=mask_arrays, *args, **kwargs
        )


def _reduce_dataarray(
    dataarray: xr.DataArray,
    geodataframe: gpd.GeoDataFrame | None = None,
    mask_arrays: list[xr.DataArray] | None = None,
    how: T.Callable | str = "mean",
    weights: None | str | np.ndarray = None,
    lat_key: str | None = None,
    lon_key: str | None = None,
    extra_reduce_dims: list | str = [],
    mask_dim: str | None = None,
    return_as: str = "xarray",
    how_label: str | None = None,
    squeeze: bool = True,
    all_touched: bool = False,
    mask_kwargs: dict[str, T.Any] = dict(),
    return_geometry_as_coord: bool = False,
    **reduce_kwargs,
) -> xr.DataArray | pd.DataFrame:
    """Reduce an xarray.DataArray object over its geospatial dimensions using the specified 'how' method.

    If a geodataframe is provided the DataArray is reduced over each feature in the geodataframe.
    Geospatial coordinates are reduced to a dimension representing the list of features in the shape object.

    Parameters
    ----------
    dataarray :
        Xarray data object (must have geospatial coordinates).
    geodataframe (optional):
        Geopandas Dataframe containing the polygons for aggregations
    how (optional):
        method used to apply mask. Default='mean', which calls np.nanmean
    weights (optional):
        Provide weights for aggregation, also accepts recognised keys for weights, e.g.
        'latitude'
    lat_key/lon_key (optional):
        key for latitude/longitude variable, default behaviour is to detect variable keys.
    extra_reduce_dims (optional):
        any additional dimensions to aggregate over when reducing over spatial dimensions
    mask_dim (optional):
        dimension that will be created after the reduction of the spatial dimensions, default = `"index"`
    return_as (optional):
        what format to return the data object, `"pandas"` or `"xarray"`. Work In Progress
    how_label (optional):
        label to append to variable name in returned object, default is `how`
    mask_kwargs (optional):
        Any kwargs to pass into the mask method
    reduce_kwargs (optional):
        kwargs recognised by the how function
    return_geometry_as_coord (optional):
        include the geometries as a coordinate in the returned xarray object. WARNING: geometries are not
        serialisable objects, therefore this xarray will not be saveable as netCDF.

    Returns
    -------
    xr.Dataset | xr.DataArray | pd.DataFrame
        A data array with dimensions [features] + [data.dims not in ['lat','lon']].
        Each slice of layer corresponds to a feature in layer

    """
    extra_out_attrs = {}
    how_str: None | str = None
    if weights is None:
        # convert how string to a method to apply
        if isinstance(how, str):
            how_str = deepcopy(how)
            how = get_how(how)
        assert isinstance(how, T.Callable), f"how must be a callable: {how}"
        if how_str is None:
            # get label from how method
            how_str = how.__name__
    else:
        # Create any standard weights, e.g. latitude.
        # TODO: handle kwargs better, currently only lat_key is accepted
        if isinstance(weights, str):
            weights = standard_weights(dataarray, weights, lat_key=lat_key)
        # We ensure the callable is a string
        if callable(how):
            how = how.__name__
        if how_str is None:
            how_str = how

    how_str = how_str or how_label
    new_long_name_components = [
        comp for comp in [how_str, dataarray.attrs.get("long_name", dataarray.name)] if comp is not None
    ]
    new_long_name = " ".join(new_long_name_components)
    new_short_name_components = [comp for comp in [dataarray.name, how_label] if comp is not None]
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
    # If using a pre-computed mask arrays, then iterator is just dataarray*mask_array
    if mask_arrays is not None:
        masked_data_list = _array_mask_iterator(mask_arrays)
    else:
        # If no geodataframe, then no mask, so create a dummy mask:
        if geodataframe is None:
            masked_data_list = [dataarray]
        else:
            masked_data_list = _shape_mask_iterator(geodataframe, dataarray, **mask_kwargs)

    reduced_list = []
    for masked_data in masked_data_list:
        this = dataarray.where(masked_data, other=np.nan)

        # If weighted, use xarray weighted arrays which correctly handle missing values etc.
        if weights is not None:
            this = this.weighted(weights)
            reduced_list.append(this.__getattribute__(how)(**reduce_kwargs))
        else:
            reduced = this.reduce(how, **reduce_kwargs).compute()
            reduced = reduced.assign_attrs(dataarray.attrs)
            reduced_list.append(reduced)

    if squeeze:
        reduced_list = [red_data.squeeze() for red_data in reduced_list]

    # If no geodataframe, there is just one reduced array
    if geodataframe is not None:
        mask_dim_index = get_mask_dim_index(mask_dim, geodataframe)
        out_xr = xr.concat(reduced_list, dim=mask_dim_index)
    elif len(reduced_list) == 1:
        out_xr = reduced_list[0]
    else:
        _concat_dim_name = mask_dim or "index"
        out_xr = xr.concat(reduced_list, dim=_concat_dim_name)

    out_xr = out_xr.rename(new_short_name)

    if "pandas" in return_as:
        reduce_attrs = {
            f"{dataarray.name}": dataarray.attrs,
            f"{new_short_name}": {
                "long_name": new_long_name,
                "units": dataarray.attrs.get("units", "No units found"),
                **extra_out_attrs,
            },
        }

        if geodataframe is None:
            # If no geodataframe, then just convert xarray to dataframe
            out = out_xr.to_dataframe()
        else:
            # Otherwise, splice the geodataframe and reduced xarray
            reduce_attrs = {
                **geodataframe.attrs.get("reduce_attrs", {}),
                **reduce_attrs,
            }
            out = geodataframe.set_index(mask_dim_index)
            if return_as in ["pandas"]:  # Return as a fully expanded pandas.DataFrame
                # Convert to DataFrame
                out = out.join(out_xr.to_dataframe())
            elif return_as in ["pandas_compact"]:
                # add the reduced data into a new column as a numpy array,
                # store the dim information in the attributes

                out_dims = {
                    dim: dataarray.coords.get(dim).values if dim in dataarray.coords else None
                    for dim in reduced_list[0].dims
                }
                reduce_attrs[f"{new_short_name}"].update({"dims": out_dims})
                reduced_list = [red.values for red in reduced_list]
                out = out.assign(**{new_short_name: reduced_list})
        out.attrs.update({"reduce_attrs": reduce_attrs})
    else:
        if geodataframe is not None:
            if return_geometry_as_coord:
                out_xr = out_xr.assign_coords(
                    **{
                        "geometry": (mask_dim, [geom for geom in geodataframe["geometry"]]),
                    }
                )
            out_xr = out_xr.assign_attrs({**geodataframe.attrs, **extra_out_attrs})
        out = out_xr

    return out
