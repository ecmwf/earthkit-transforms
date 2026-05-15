import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

# from earthkit.data.core.temporary import temp_directory
from earthkit import data as ekd
from earthkit.transforms import spatial
from earthkit.transforms._tools import earthkit_remote_test_data_file
from earthkit.transforms.spatial import _aggregate as _spatial

try:
    import rasterio  # noqa: F401

    rasterio_available = True
except ImportError:
    rasterio_available = False

# Use caching for speedy repeats
ekd.settings.set("cache-policy", "user")


SAMPLE_ARRAY = xr.DataArray(
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ],
    dims=["latitude", "longitude"],
    coords={
        "latitude": [0, 60, 90],  # Chosen for latitude weight tests
        "longitude": [0, 30, 60, 90],
    },
)


class dummy_class:
    def __init__(self):
        self.to_pandas = pd.DataFrame
        self.to_geopandas = pd.DataFrame


def get_grid_data():
    remote_era5_file = earthkit_remote_test_data_file("era5_temperature_europe_20150101.grib")
    return ekd.from_source("url", remote_era5_file)


def get_shape_data():
    if rasterio_available:
        remote_nuts_url = earthkit_remote_test_data_file("NUTS_RG_60M_2021_4326_LEVL_0.geojson")
        return ekd.from_source("url", remote_nuts_url)
    return dummy_class()


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
def test_spatial_mask():
    single_masked_data = spatial.mask(get_grid_data(), get_shape_data(), union_geometries=True)
    assert isinstance(single_masked_data, xr.Dataset)


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
@pytest.mark.parametrize(
    "era5_data, nuts_data, expected_result_type",
    (
        [get_grid_data(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data().to_xarray()["2t"], get_shape_data(), xr.DataArray],
    ),
)
def test_spatial_masks_with_ek_objects(era5_data, nuts_data, expected_result_type):
    masked_data = spatial.mask(era5_data, nuts_data)
    assert isinstance(masked_data, expected_result_type)
    assert "index" in masked_data.dims
    nuts_pandas = ekd.from_object(nuts_data).to_pandas()
    assert len(masked_data["index"]) == len(nuts_pandas)


@pytest.mark.parametrize(
    "era5_data, expected_result_type",
    (
        [get_grid_data(), xr.Dataset],
        [get_grid_data().to_xarray(), xr.Dataset],
        [get_grid_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_spatial_reduce_no_geometry(era5_data, expected_result_type):
    reduced_data = spatial.reduce(era5_data)

    assert isinstance(reduced_data, expected_result_type)
    assert list(reduced_data.dims) == ["forecast_reference_time"]


def test_spatial_reduce_no_geometry_result():
    reduced_data = spatial.reduce(SAMPLE_ARRAY, how="mean")
    assert reduced_data.values == 2.0
    reduced_data = spatial.reduce(SAMPLE_ARRAY, how="mean", weights="latitude")
    assert np.isclose(reduced_data.values, 1 + (1.0 / 3))


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
@pytest.mark.parametrize(
    "era5_data, nuts_data, expected_result_type",
    (
        [get_grid_data(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data().to_xarray()["2t"], get_shape_data(), xr.DataArray],
    ),
)
def test_spatial_reduce_with_geometry(era5_data, nuts_data, expected_result_type):
    reduced_data = spatial.reduce(era5_data, nuts_data)
    assert isinstance(reduced_data, expected_result_type)
    assert all([dim in ["forecast_reference_time", "index"] for dim in reduced_data.dims])
    nuts_pandas = ekd.from_object(nuts_data).to_pandas()
    assert len(reduced_data["index"]) == len(nuts_pandas)


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
def test_spatial_reduce_with_precomputed_mask():
    era5_data_xr = get_grid_data().to_xarray()["2t"]
    ones = (era5_data_xr.isel(forecast_reference_time=0) * 0 + 1).astype(int).rename("mask")
    mask = spatial.mask(ones, get_shape_data(), all_touched=False)
    mask_arrays = [mask.sel(index=index) for index in mask.index]
    reduced_data_test = spatial.reduce(era5_data_xr, geodataframe=get_shape_data())

    # reduce with a single mask
    reduced_data = spatial.reduce(era5_data_xr, mask_arrays=mask_arrays[0])
    assert isinstance(reduced_data, xr.DataArray)
    assert all([dim in ["forecast_reference_time", "index"] for dim in reduced_data.dims])
    assert reduced_data.equals(reduced_data_test.isel(index=0))

    # reduce with list of masks
    reduced_data = spatial.reduce(era5_data_xr, mask_arrays=mask_arrays)
    assert isinstance(reduced_data, xr.DataArray)
    assert all([dim in ["forecast_reference_time", "index"] for dim in reduced_data.dims])
    assert len(reduced_data["index"]) == len(mask_arrays)
    assert reduced_data.equals(reduced_data_test)


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
@pytest.mark.parametrize(
    "era5_data, nuts_data, expected_result_type",
    ([get_grid_data().to_xarray(), get_shape_data().to_pandas(), xr.Dataset],),
)
def test_spatial_reduce_with_geometry_and_latitude_weights(era5_data, nuts_data, expected_result_type):
    reduced_data = spatial.reduce(era5_data, nuts_data, weights="latitude")
    assert isinstance(reduced_data, expected_result_type)
    assert all([dim in ["forecast_reference_time", "index"] for dim in reduced_data.dims])
    assert len(reduced_data["index"]) == len(nuts_data)

    # Ensure weights works with an abstract name for latitude
    era5_data = era5_data.rename({"latitude": "elephant"})
    reduced_data = spatial.reduce(era5_data, nuts_data, weights="latitude", lat_key="elephant")
    assert isinstance(reduced_data, expected_result_type)
    assert all([dim in ["forecast_reference_time", "index"] for dim in reduced_data.dims])
    assert len(reduced_data["index"]) == len(nuts_data)


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
def test_mask_kwargs():
    era5_data = get_grid_data()
    era5_xr = era5_data.to_xarray()

    nuts_data = get_shape_data()
    nuts_gpd = nuts_data.to_geopandas()
    nuts_DK = nuts_gpd[nuts_gpd["CNTR_CODE"] == "DK"]

    masked_data = spatial.mask(era5_xr, nuts_DK, all_touched=True)
    assert len(np.where(~np.isnan(masked_data["2t"].values.flat))[0]) == 3432

    masked_data = spatial.mask(era5_xr, nuts_DK, all_touched=False)
    assert len(np.where(~np.isnan(masked_data["2t"].values.flat))[0]) == 2448

    reduced_data = spatial.reduce(era5_xr, nuts_DK, all_touched=True)
    reduced_data_nested = spatial.reduce(era5_xr, nuts_DK, mask_kwargs=dict(all_touched=True))
    xr.testing.assert_equal(reduced_data, reduced_data_nested)
    np.testing.assert_allclose(reduced_data["2t"].mean(), 279.4813)

    reduced_data_2 = spatial.reduce(era5_xr, nuts_DK, all_touched=False)
    reduced_data_nested_2 = spatial.reduce(era5_xr, nuts_DK, mask_kwargs=dict(all_touched=False))
    xr.testing.assert_equal(reduced_data_2, reduced_data_nested_2)
    np.testing.assert_allclose(reduced_data_2["2t"].mean(), 279.54733)


def create_test_dataarray():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    data = np.random.rand(10, 20)
    return xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test_var",
    )


def create_test_geodataframe():
    polygons = [Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])]
    return gpd.GeoDataFrame(geometry=polygons, index=[1])


def test_reduce_mean():
    dataarray = create_test_dataarray()
    result = _spatial._reduce_dataarray_as_xarray(dataarray, how="mean")
    assert isinstance(result, xr.DataArray)
    assert "lat" not in result.dims
    assert "lon" not in result.dims


def test_reduce_with_geodataframe():
    dataarray = create_test_dataarray()
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_xarray(dataarray, geodataframe=geodataframe, how="mean")
    assert isinstance(result, xr.DataArray)
    assert "index" in result.dims  # Default mask_dim is "index"


def test_reduce_with_weights():
    dataarray = create_test_dataarray()
    result = _spatial._reduce_dataarray_as_xarray(dataarray, how="mean", weights="latitude")
    assert isinstance(result, xr.DataArray)


def test_reduce_invalid_how():
    dataarray = create_test_dataarray()
    with pytest.raises(ValueError):
        _spatial._reduce_dataarray_as_xarray(dataarray, how="invalid_method")


def test_reduce_with_mask():
    dataarray = create_test_dataarray()
    mask = xr.DataArray(np.random.randint(0, 2, size=dataarray.shape), coords=dataarray.coords, dims=dataarray.dims)
    result = _spatial._reduce_dataarray_as_xarray(dataarray, mask_arrays=[mask], how="sum")
    assert isinstance(result, xr.DataArray)


def test_return_geometry_as_coord():
    dataarray = create_test_dataarray()
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_xarray(dataarray, geodataframe=geodataframe, return_geometry_as_coord=True)
    assert "geometry" in result.coords
    assert len(result.coords["geometry"].values) == len(geodataframe)


def test_reduce_as_pandas():
    dataarray = create_test_dataarray()
    result = _spatial._reduce_dataarray_as_pandas(dataarray, how="mean")
    assert isinstance(result, pd.DataFrame)


def test_reduce_as_pandas_with_geodataframe():
    dataarray = create_test_dataarray()
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_pandas(dataarray, geodataframe=geodataframe, how="mean")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_reduce_as_pandas_compact():
    dataarray = create_test_dataarray()
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_pandas(dataarray, geodataframe=geodataframe, compact=True, how="mean")
    assert isinstance(result, pd.DataFrame)
    assert f"{dataarray.name}" in result.columns


def test_reduce_as_pandas_without_geodataframe():
    dataarray = create_test_dataarray()
    result = _spatial._reduce_dataarray_as_pandas(dataarray, how="sum")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


# ---------------------------------------------------------------------------
# Local (synthetic-data) tests — no network required
# ---------------------------------------------------------------------------


def test_spatial_reduce_dataset_local():
    """spatial.reduce on a Dataset should return a Dataset."""
    ds = xr.Dataset({"var": SAMPLE_ARRAY})
    result = spatial.reduce(ds)
    assert isinstance(result, xr.Dataset)
    assert "var" in result


@pytest.mark.parametrize("how", ("min", "max", "sum", "std"))
def test_spatial_reduce_how_options_local(how):
    """spatial.reduce should accept common how strings on SAMPLE_ARRAY."""
    result = spatial.reduce(SAMPLE_ARRAY, how=how)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ()


@pytest.mark.skipif(
    not rasterio_available,
    reason="rasterio is not available",
)
def test_spatial_reduce_with_shapely_geodataframe_local():
    """spatial.reduce with a simple shapely GeoDataFrame returns 'index' dim."""
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_xarray(create_test_dataarray(), geodataframe=geodataframe, how="mean")
    assert isinstance(result, xr.DataArray)
    assert "index" in result.dims


# ---------------------------------------------------------------------------
# mask_contains_points tests — no network required
# ---------------------------------------------------------------------------

# A 5×5 regular grid centred on the origin, 1-degree spacing
_MCP_LATS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_MCP_LONS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_MCP_INDEX = np.arange(5)
_MCP_DA = xr.DataArray(
    np.arange(5),
    name="test",
    dims=["points"],
    coords={
        "points": _MCP_INDEX,
        "latitude": ("points", _MCP_LATS),
        "longitude": ("points", _MCP_LONS),
    },
)
_MCP_COORDS = _MCP_DA.coords


def test_mask_contains_points_simple_polygon():
    """Points strictly inside a simple polygon are True; outside points are NaN."""
    # Box covering only the centre point (0, 0)
    shapes = [
        Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]),
    ]
    result = _spatial.mask_contains_points(shapes, _MCP_COORDS)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"points"}
    # Only the centre grid point (lat=0, lon=0) should be True
    assert bool(result.sel(points=2) == True)  # noqa: E712
    # All other points should be NaN
    other = result.where(~(result.points == 2), drop=True)
    # Check all values of other are nan
    assert np.isnan(other).all()


def test_mask_contains_points_simple_polygon_values():
    """Concise value check: centre True, corners NaN."""
    shape = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
    result = _spatial.mask_contains_points([shape], _MCP_COORDS)

    assert result.sel(points=2).item() == True  # noqa: E712
    assert np.isnan(result.sel(points=0).item())
    assert np.isnan(result.sel(points=4).item())


def test_mask_contains_points_polygon_with_hole():
    """Points inside a hole (interior ring) must NOT be marked as inside."""
    # Outer ring: covers (-1.5, -1.5) to (1.5, 1.5)
    # Hole: covers (-0.4, -0.4) to (0.4, 0.4) — excludes the exact centre point
    outer = [(-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5), (1.5, -1.5)]
    hole = [(-0.4, -0.4), (-0.4, 0.4), (0.4, 0.4), (0.4, -0.4)]
    shape = Polygon(outer, [hole])

    result = _spatial.mask_contains_points([shape], _MCP_COORDS)

    # The exact centre (0, 0) lies inside the hole — must be NaN
    assert np.isnan(result.sel(points=2).item()), "Centre point is inside the hole and should be NaN"
    # A point inside the outer ring but outside the hole should be True
    assert result.sel(points=3).item() == True  # noqa: E712


def test_mask_contains_points_multipolygon():
    """Points inside any sub-polygon of a MultiPolygon are marked True."""
    # Two separate boxes: top-right and bottom-left quadrants
    box1 = Polygon([(0.6, 0.6), (0.6, 1.5), (1.5, 1.5), (1.5, 0.6)])  # covers (1, 1)
    box2 = Polygon([(-1.5, -1.5), (-1.5, -0.6), (-0.6, -0.6), (-0.6, -1.5)])  # covers (-1, -1)
    shape = MultiPolygon([box1, box2])

    result = _spatial.mask_contains_points([shape], _MCP_COORDS)

    assert result.sel(points=3).item() == True  # noqa: E712
    assert result.sel(points=1).item() == True  # noqa: E712
    assert np.isnan(result.sel(points=2).item())
    assert np.isnan(result.sel(points=4).item())


def test_mask_contains_points_no_points_inside():
    """When no grid points fall inside the shape the result is all NaN."""
    # Shape entirely outside the grid extent
    shape = Polygon([(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)])
    result = _spatial.mask_contains_points([shape], _MCP_COORDS)
    assert np.all(np.isnan(result.values))


# ---------------------------------------------------------------------------
# shapes_to_mask / shapes_to_masks tests — no network required
# (irregular path: regular=False; regular path requires rasterio)
# ---------------------------------------------------------------------------

# Two non-overlapping boxes on the _MCP point grid:
#   box_a covers point index=3  (lat= 1, lon= 1)
#   box_b covers point index=1  (lat=-1, lon=-1)
_BOX_A = Polygon([(0.6, 0.6), (0.6, 1.4), (1.4, 1.4), (1.4, 0.6)])
_BOX_B = Polygon([(-1.4, -1.4), (-1.4, -0.6), (-0.6, -0.6), (-0.6, -1.4)])


def test_shapes_to_mask_returns_dataarray():
    """shapes_to_mask returns a single xr.DataArray."""
    result = _spatial.shapes_to_mask([_BOX_A], _MCP_DA, regular=False)
    assert isinstance(result, xr.DataArray)


def test_shapes_to_mask_single_shape():
    """Single shape: points inside → True, points outside → NaN."""
    result = _spatial.shapes_to_mask([_BOX_A], _MCP_DA, regular=False)
    assert result.sel(points=3).item() == True  # noqa: E712
    assert np.isnan(result.sel(points=1).item())
    assert np.isnan(result.sel(points=2).item())


def test_shapes_to_mask_union_of_two_shapes():
    """Two shapes passed together produce a union mask covering both."""
    result = _spatial.shapes_to_mask([_BOX_A, _BOX_B], _MCP_DA, regular=False)
    # Both covered points are True
    assert result.sel(points=3).item() == True  # noqa: E712
    assert result.sel(points=1).item() == True  # noqa: E712
    # Points outside both shapes are NaN
    assert np.isnan(result.sel(points=0).item())
    assert np.isnan(result.sel(points=2).item())
    assert np.isnan(result.sel(points=4).item())


def test_shapes_to_mask_geodataframe():
    """shapes_to_mask accepts a GeoDataFrame and unions all its geometries."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A, _BOX_B])
    result = _spatial.shapes_to_mask(gdf, _MCP_DA, regular=False)
    assert isinstance(result, xr.DataArray)
    assert result.sel(points=3).item() == True  # noqa: E712
    assert result.sel(points=1).item() == True  # noqa: E712
    assert np.isnan(result.sel(points=2).item())


def test_shapes_to_mask_no_points_inside():
    """Shapes entirely outside the grid → all NaN."""
    far_shape = Polygon([(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)])
    result = _spatial.shapes_to_mask([far_shape], _MCP_DA, regular=False)
    assert np.all(np.isnan(result.values))


def test_shapes_to_masks_returns_list():
    """shapes_to_masks returns a list with one mask per input shape."""
    result = _spatial.shapes_to_masks([_BOX_A, _BOX_B], _MCP_DA, regular=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(m, xr.DataArray) for m in result)


def test_shapes_to_masks_independent_masks():
    """Each mask in shapes_to_masks covers only its own shape."""
    masks = _spatial.shapes_to_masks([_BOX_A, _BOX_B], _MCP_DA, regular=False)
    mask_a, mask_b = masks
    # mask_a covers index=3 only
    assert mask_a.sel(points=3).item() == True  # noqa: E712
    assert np.isnan(mask_a.sel(points=1).item())
    # mask_b covers index=1 only
    assert mask_b.sel(points=1).item() == True  # noqa: E712
    assert np.isnan(mask_b.sel(points=3).item())


def test_shapes_to_masks_geodataframe():
    """shapes_to_masks accepts a GeoDataFrame, returning one mask per row."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A, _BOX_B])
    result = _spatial.shapes_to_masks(gdf, _MCP_DA, regular=False)
    assert len(result) == len(gdf)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_shapes_to_mask_regular_grid():
    """shapes_to_mask with regular=True (rasterize path) returns a 2-D mask."""
    # Use a box that covers the whole create_test_dataarray extent
    shape = Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
    target = create_test_dataarray()
    result = _spatial.shapes_to_mask([shape], target, regular=True, lat_key="lat", lon_key="lon")
    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"lat", "lon"}
    # All points inside → no NaN values (rasterize fills inside with 1)
    assert not np.any(np.isnan(result.values))


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_shapes_to_masks_regular_grid():
    """shapes_to_masks with regular=True returns a list of 2-D masks."""
    shape = Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
    target = create_test_dataarray()
    result = _spatial.shapes_to_masks([shape], target, regular=True, lat_key="lat", lon_key="lon")
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], xr.DataArray)
    assert set(result[0].dims) == {"lat", "lon"}


# ---------------------------------------------------------------------------
# spatial.mask tests — local (irregular / point-cloud path)
# and rasterio-gated regular path
# ---------------------------------------------------------------------------

# Re-use the point-cloud fixture and boxes from the mask_contains_points section.
# _MCP_DA  : DataArray with dim "points", coords latitude/longitude on "points"
# _BOX_A   : covers points index=3  (lat=1,  lon=1)
# _BOX_B   : covers points index=1  (lat=-1, lon=-1)

_MASK_GDF_TWO = gpd.GeoDataFrame(geometry=[_BOX_A, _BOX_B])


def test_spatial_mask_returns_dataarray_for_dataarray_input():
    """spatial.mask on a DataArray returns a DataArray."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude")
    assert isinstance(result, xr.DataArray)


def test_spatial_mask_returns_dataset_for_dataset_input():
    """spatial.mask on a Dataset returns a Dataset."""
    ds = xr.Dataset({"var": _MCP_DA})
    result = spatial.mask(ds, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude")
    assert isinstance(result, xr.Dataset)


def test_spatial_mask_dim_length_equals_number_of_shapes():
    """The new mask dimension has one entry per shape in the GeoDataFrame."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude")
    assert "index" in result.dims
    assert len(result["index"]) == len(_MASK_GDF_TWO)


def test_spatial_mask_default_dim_name_is_index():
    """When mask_dim is not supplied, the new dimension is named 'index'."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude")
    assert "index" in result.dims


def test_spatial_mask_custom_mask_dim_string():
    """mask_dim as a string renames the new dimension."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude", mask_dim="region")
    assert "region" in result.dims
    assert "index" not in result.dims


def test_spatial_mask_custom_mask_dim_from_column():
    """mask_dim pointing at a GeoDataFrame column populates the coordinate."""
    gdf = gpd.GeoDataFrame({"geometry": [_BOX_A, _BOX_B], "name": ["A", "B"]})
    result = spatial.mask(_MCP_DA, gdf, lat_key="latitude", lon_key="longitude", mask_dim="name")
    assert "name" in result.dims
    assert list(result["name"].values) == ["A", "B"]


def test_spatial_mask_values_inside_preserved():
    """Values at points inside a shape are preserved (not NaN)."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A])
    result = spatial.mask(_MCP_DA, gdf, lat_key="latitude", lon_key="longitude", chunk=False)
    # _MCP_DA value at index=3 is 3; BOX_A covers that point
    inside_val = result.isel(index=0).sel(points=3).item()
    assert inside_val == 3


def test_spatial_mask_values_outside_are_nan():
    """Values at points outside a shape are NaN."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A])
    result = spatial.mask(_MCP_DA, gdf, lat_key="latitude", lon_key="longitude", chunk=False)
    # Points 0, 1, 2, 4 are outside BOX_A
    for pt in [0, 1, 2, 4]:
        assert np.isnan(result.isel(index=0).sel(points=pt).item())


def test_spatial_mask_each_slice_independent():
    """Each slice along the mask dim corresponds only to its own shape."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude", chunk=False)
    # Slice 0 → BOX_A: only point 3 should be non-NaN
    slice_a = result.isel(index=0)
    assert not np.isnan(slice_a.sel(points=3).item())
    assert np.isnan(slice_a.sel(points=1).item())
    # Slice 1 → BOX_B: only point 1 should be non-NaN
    slice_b = result.isel(index=1)
    assert not np.isnan(slice_b.sel(points=1).item())
    assert np.isnan(slice_b.sel(points=3).item())


def test_spatial_mask_union_geometries_no_extra_dim():
    """union_geometries=True returns a single mask without a per-shape dimension."""
    result = spatial.mask(_MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude", union_geometries=True)
    # No new index dimension — result dims should match the input dims
    assert "index" not in result.dims
    assert "points" in result.dims


def test_spatial_mask_union_geometries_covers_all_shapes():
    """union_geometries=True mask covers all shapes simultaneously."""
    result = spatial.mask(
        _MCP_DA, _MASK_GDF_TWO, lat_key="latitude", lon_key="longitude", union_geometries=True, chunk=False
    )
    # Both BOX_A (point 3) and BOX_B (point 1) should be non-NaN
    assert not np.isnan(result.sel(points=3).item())
    assert not np.isnan(result.sel(points=1).item())
    # Points outside both boxes remain NaN
    assert np.isnan(result.sel(points=2).item())


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_regular_grid_dim_length():
    """spatial.mask on a regular grid (rasterize path) adds the correct dim length."""
    target = create_test_dataarray()
    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(-180, -90), (-180, 0), (0, 0), (0, -90)]),
            Polygon([(0, 0), (0, 90), (180, 90), (180, 0)]),
        ]
    )
    result = spatial.mask(target, gdf, lat_key="lat", lon_key="lon")
    assert isinstance(result, xr.DataArray)
    assert "index" in result.dims
    assert len(result["index"]) == 2


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_regular_grid_union():
    """union_geometries=True on a regular grid returns a 2-D result (no extra dim)."""
    target = create_test_dataarray()
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])])
    result = spatial.mask(target, gdf, lat_key="lat", lon_key="lon", union_geometries=True)
    assert "index" not in result.dims
    assert set(result.dims) == {"lat", "lon"}


# ---------------------------------------------------------------------------
# spatial.mask — value tests for 2-D regular data (rasterize path)
# ---------------------------------------------------------------------------

# 5×5 regular grid; values are unique so we can pin exact cells.
# lat/lon both in [-2,-1,0,1,2] with spacing 1.
# Pixel (row i, col j) has its center at lat = -2+i, lon = -2+j.
#   → (lat=1, lon=1) = array[3,3] = value 18   ← covered by _BOX_A
#   → (lat=-1, lon=-1) = array[1,1] = value 6  ← covered by _BOX_B
_REG_LATS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_REG_LONS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_REG_DA = xr.DataArray(
    np.arange(25).reshape(5, 5).astype(float),
    dims=["latitude", "longitude"],
    coords={"latitude": _REG_LATS, "longitude": _REG_LONS},
    name="test_reg",
)
_REG_GDF_TWO = gpd.GeoDataFrame(geometry=[_BOX_A, _BOX_B])


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_reg_values_inside_preserved():
    """Regular grid: value at the point inside the shape is preserved."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A])
    result = spatial.mask(_REG_DA, gdf, lat_key="latitude", lon_key="longitude", chunk=False)
    # BOX_A covers (lat=1, lon=1), array value = 3*5+3 = 18
    inside_val = result.isel(index=0).sel(latitude=1.0, longitude=1.0).item()
    assert inside_val == 18.0


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_reg_values_outside_are_nan():
    """Regular grid: values at points outside the shape are NaN."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A])
    result = spatial.mask(_REG_DA, gdf, lat_key="latitude", lon_key="longitude", chunk=False)
    slice_0 = result.isel(index=0)
    # All cells except (lat=1, lon=1) should be NaN
    for lat in _REG_LATS:
        for lon in _REG_LONS:
            val = slice_0.sel(latitude=lat, longitude=lon).item()
            if lat == 1.0 and lon == 1.0:
                assert not np.isnan(val)
            else:
                assert np.isnan(val), f"Expected NaN at (lat={lat}, lon={lon}), got {val}"


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_reg_each_slice_independent():
    """Regular grid: each slice covers only its own shape."""
    result = spatial.mask(_REG_DA, _REG_GDF_TWO, lat_key="latitude", lon_key="longitude", chunk=False)
    # Slice 0 → BOX_A: only (lat=1, lon=1) non-NaN, value=18
    slice_a = result.isel(index=0)
    assert slice_a.sel(latitude=1.0, longitude=1.0).item() == 18.0
    assert np.isnan(slice_a.sel(latitude=-1.0, longitude=-1.0).item())
    # Slice 1 → BOX_B: only (lat=-1, lon=-1) non-NaN, value=6
    slice_b = result.isel(index=1)
    assert slice_b.sel(latitude=-1.0, longitude=-1.0).item() == 6.0
    assert np.isnan(slice_b.sel(latitude=1.0, longitude=1.0).item())


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_mask_reg_union_covers_both_shapes():
    """Regular grid union mask: both target cells non-NaN, remaining cells NaN."""
    result = spatial.mask(
        _REG_DA,
        _REG_GDF_TWO,
        lat_key="latitude",
        lon_key="longitude",
        union_geometries=True,
        chunk=False,
    )
    assert result.sel(latitude=1.0, longitude=1.0).item() == 18.0
    assert result.sel(latitude=-1.0, longitude=-1.0).item() == 6.0
    # Every other cell must be NaN
    for lat in _REG_LATS:
        for lon in _REG_LONS:
            val = result.sel(latitude=lat, longitude=lon).item()
            if (lat == 1.0 and lon == 1.0) or (lat == -1.0 and lon == -1.0):
                assert not np.isnan(val)
            else:
                assert np.isnan(val), f"Expected NaN at (lat={lat}, lon={lon}), got {val}"


# ---------------------------------------------------------------------------
# spatial.reduce — value tests with GeoDataFrame input
# ---------------------------------------------------------------------------

# _QUAD_BOX: covers the 2×2 top-right cells of _REG_DA
#   (lat=1,lon=1)=18  (lat=1,lon=2)=19  (lat=2,lon=1)=23  (lat=2,lon=2)=24
#   mean=21.0  sum=84.0  min=18.0  max=24.0
_QUAD_BOX = Polygon([(0.6, 0.6), (0.6, 2.4), (2.4, 2.4), (2.4, 0.6)])

# _CENTRE_BOX: covers _MCP_DA diagonal points 1, 2, 3 (values 1, 2, 3)
#   mean=2.0  sum=6.0  min=1.0  max=3.0
_CENTRE_BOX = Polygon([(-1.4, -1.4), (-1.4, 1.4), (1.4, 1.4), (1.4, -1.4)])


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_reduce_reg_mean():
    """Regular grid: mean over a 2×2 region gives the exact arithmetic mean."""
    gdf = gpd.GeoDataFrame(geometry=[_QUAD_BOX])
    result = spatial.reduce(_REG_DA, gdf, how="mean")
    # (18+19+23+24) / 4 = 21.0
    np.testing.assert_allclose(result.isel(index=0).item(), 21.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_reduce_reg_sum():
    """Regular grid: sum over a 2×2 region."""
    gdf = gpd.GeoDataFrame(geometry=[_QUAD_BOX])
    result = spatial.reduce(_REG_DA, gdf, how="sum")
    # 18+19+23+24 = 84
    np.testing.assert_allclose(result.isel(index=0).item(), 84.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_reduce_reg_min_max():
    """Regular grid: min and max over a 2×2 region."""
    gdf = gpd.GeoDataFrame(geometry=[_QUAD_BOX])
    result_min = spatial.reduce(_REG_DA, gdf, how="min")
    result_max = spatial.reduce(_REG_DA, gdf, how="max")
    np.testing.assert_allclose(result_min.isel(index=0).item(), 18.0)
    np.testing.assert_allclose(result_max.isel(index=0).item(), 24.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_reduce_reg_two_shapes():
    """Regular grid: two shapes reduce independently to their correct means."""
    gdf = gpd.GeoDataFrame(geometry=[_QUAD_BOX, _BOX_B])
    result = spatial.reduce(_REG_DA, gdf, how="mean")
    # QUAD_BOX mean=21.0; BOX_B covers only (lat=-1,lon=-1)=6 → mean=6.0
    np.testing.assert_allclose(result.isel(index=0).item(), 21.0)
    np.testing.assert_allclose(result.isel(index=1).item(), 6.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
def test_spatial_reduce_reg_mask_dim_coord():
    """Regular grid: mask_dim column values propagate to the output coordinate."""
    gdf = gpd.GeoDataFrame({"geometry": [_QUAD_BOX, _BOX_B], "label": ["quad", "b"]})
    result = spatial.reduce(_REG_DA, gdf, how="mean", mask_dim="label")
    np.testing.assert_allclose(result.sel(label="quad").item(), 21.0)
    np.testing.assert_allclose(result.sel(label="b").item(), 6.0)


def test_spatial_reduce_irregular_mean():
    """Irregular (point-cloud) grid: mean over 3 diagonal points."""
    gdf = gpd.GeoDataFrame(geometry=[_CENTRE_BOX])
    result = spatial.reduce(_MCP_DA, gdf, how="mean", lat_key="latitude", lon_key="longitude")
    # points 1,2,3 → values 1,2,3 → mean=2.0
    np.testing.assert_allclose(result.isel(index=0).item(), 2.0)


def test_spatial_reduce_irregular_sum():
    """Irregular (point-cloud) grid: sum over 3 diagonal points."""
    gdf = gpd.GeoDataFrame(geometry=[_CENTRE_BOX])
    result = spatial.reduce(_MCP_DA, gdf, how="sum", lat_key="latitude", lon_key="longitude")
    # 1+2+3 = 6
    np.testing.assert_allclose(result.isel(index=0).item(), 6.0)


def test_spatial_reduce_irregular_min_max():
    """Irregular (point-cloud) grid: min and max over 3 diagonal points."""
    gdf = gpd.GeoDataFrame(geometry=[_CENTRE_BOX])
    result_min = spatial.reduce(_MCP_DA, gdf, how="min", lat_key="latitude", lon_key="longitude")
    result_max = spatial.reduce(_MCP_DA, gdf, how="max", lat_key="latitude", lon_key="longitude")
    np.testing.assert_allclose(result_min.isel(index=0).item(), 1.0)
    np.testing.assert_allclose(result_max.isel(index=0).item(), 3.0)


def test_spatial_reduce_irregular_two_shapes():
    """Irregular (point-cloud) grid: two shapes reduce independently."""
    gdf = gpd.GeoDataFrame(geometry=[_BOX_A, _BOX_B])
    result = spatial.reduce(_MCP_DA, gdf, how="mean", lat_key="latitude", lon_key="longitude")
    # BOX_A → point 3 (value 3); BOX_B → point 1 (value 1)
    np.testing.assert_allclose(result.isel(index=0).item(), 3.0)
    np.testing.assert_allclose(result.isel(index=1).item(), 1.0)


def test_spatial_reduce_irregular_mask_dim_coord():
    """Irregular (point-cloud) grid: mask_dim column values propagate to the output coordinate."""
    gdf = gpd.GeoDataFrame({"geometry": [_BOX_A, _BOX_B], "label": ["a", "b"]})
    result = spatial.reduce(_MCP_DA, gdf, how="mean", lat_key="latitude", lon_key="longitude", mask_dim="label")
    np.testing.assert_allclose(result.sel(label="a").item(), 3.0)
    np.testing.assert_allclose(result.sel(label="b").item(), 1.0)


# ---------------------------------------------------------------------------
# spatial.reduce — precomputed mask value tests
# Covers two invalid-cell conventions: 0 = invalid, NaN = invalid
# ---------------------------------------------------------------------------


def _reg_mask(valid_cells, convention):
    """Build a 5×5 mask for _REG_DA.

    valid_cells: list of (row, col) tuples that are valid (value=1).
    convention: "zero" → invalid cells are 0; "nan" → invalid cells are NaN.
    """
    fill = 0.0 if convention == "zero" else np.nan
    data = np.full((5, 5), fill)
    for r, c in valid_cells:
        data[r, c] = 1.0
    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": _REG_LATS, "longitude": _REG_LONS},
    )


def _mcp_mask(valid_points, convention):
    """Build a 5-element mask for _MCP_DA.

    valid_points: list of point indices (0–4) that are valid (value=1).
    convention: "zero" → invalid cells are 0; "nan" → invalid cells are NaN.
    """
    fill = 0.0 if convention == "zero" else np.nan
    data = np.full(5, fill)
    for p in valid_points:
        data[p] = 1.0
    return xr.DataArray(data, dims=["points"], coords={"points": _MCP_INDEX})


# _QUAD_BOX covers rows 3,4 × cols 3,4 → values 18,19,23,24 → mean=21, sum=84
# _BOX_A covers row 3, col 3 → value 18
# _BOX_B covers row 1, col 1 → value 6
_QUAD_BOX_CELLS = [(3, 3), (3, 4), (4, 3), (4, 4)]
_BOX_A_CELLS = [(3, 3)]
_BOX_B_CELLS = [(1, 1)]

# _CENTRE_BOX covers points 1,2,3 → values 1,2,3 → mean=2, sum=6
# _BOX_A covers point 3 → value 3
# _BOX_B covers point 1 → value 1
_CENTRE_POINTS = [1, 2, 3]
_BOX_A_POINTS = [3]
_BOX_B_POINTS = [1]


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_reg_single_mean(convention):
    """Regular grid, single precomputed mask: mean of 4-cell region = 21.0."""
    mask = _reg_mask(_QUAD_BOX_CELLS, convention)
    result = spatial.reduce(_REG_DA, mask_arrays=mask, how="mean")
    np.testing.assert_allclose(result.item(), 21.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_reg_single_sum(convention):
    """Regular grid, single precomputed mask: sum of 4-cell region = 84.0."""
    mask = _reg_mask(_QUAD_BOX_CELLS, convention)
    result = spatial.reduce(_REG_DA, mask_arrays=mask, how="sum")
    np.testing.assert_allclose(result.item(), 84.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_reg_single_min_max(convention):
    """Regular grid, single precomputed mask: min=18, max=24 over 4-cell region."""
    mask = _reg_mask(_QUAD_BOX_CELLS, convention)
    np.testing.assert_allclose(spatial.reduce(_REG_DA, mask_arrays=mask, how="min").item(), 18.0)
    np.testing.assert_allclose(spatial.reduce(_REG_DA, mask_arrays=mask, how="max").item(), 24.0)


@pytest.mark.skipif(not rasterio_available, reason="rasterio is not available")
@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_reg_list_two_masks(convention):
    """Regular grid, list of two precomputed masks: independent values per mask."""
    mask_a = _reg_mask(_BOX_A_CELLS, convention)  # covers cell 18
    mask_b = _reg_mask(_BOX_B_CELLS, convention)  # covers cell 6
    result = spatial.reduce(_REG_DA, mask_arrays=[mask_a, mask_b], how="mean")
    # list of two masks → "index" dim of length 2
    np.testing.assert_allclose(result.isel(index=0).item(), 18.0)
    np.testing.assert_allclose(result.isel(index=1).item(), 6.0)


@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_irr_single_mean(convention):
    """Irregular grid, single precomputed mask: mean over 3 points = 2.0."""
    mask = _mcp_mask(_CENTRE_POINTS, convention)
    result = spatial.reduce(_MCP_DA, mask_arrays=mask, how="mean", lat_key="latitude", lon_key="longitude")
    np.testing.assert_allclose(result.item(), 2.0)


@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_irr_single_sum(convention):
    """Irregular grid, single precomputed mask: sum over 3 points = 6.0."""
    mask = _mcp_mask(_CENTRE_POINTS, convention)
    result = spatial.reduce(_MCP_DA, mask_arrays=mask, how="sum", lat_key="latitude", lon_key="longitude")
    np.testing.assert_allclose(result.item(), 6.0)


@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_irr_single_min_max(convention):
    """Irregular grid, single precomputed mask: min=1, max=3 over 3 points."""
    mask = _mcp_mask(_CENTRE_POINTS, convention)
    np.testing.assert_allclose(
        spatial.reduce(_MCP_DA, mask_arrays=mask, how="min", lat_key="latitude", lon_key="longitude").item(), 1.0
    )
    np.testing.assert_allclose(
        spatial.reduce(_MCP_DA, mask_arrays=mask, how="max", lat_key="latitude", lon_key="longitude").item(), 3.0
    )


@pytest.mark.parametrize("convention", ["zero", "nan"])
def test_spatial_reduce_precomputed_mask_irr_list_two_masks(convention):
    """Irregular grid, list of two precomputed masks: independent values per mask."""
    mask_a = _mcp_mask(_BOX_A_POINTS, convention)  # covers point 3, value 3
    mask_b = _mcp_mask(_BOX_B_POINTS, convention)  # covers point 1, value 1
    result = spatial.reduce(_MCP_DA, mask_arrays=[mask_a, mask_b], how="mean", lat_key="latitude", lon_key="longitude")
    np.testing.assert_allclose(result.isel(index=0).item(), 3.0)
    np.testing.assert_allclose(result.isel(index=1).item(), 1.0)


# ---------------------------------------------------------------------------
# area kwarg tests — no network required
# ---------------------------------------------------------------------------

_AREA_FULL = {"north": 90, "south": 0, "east": 90, "west": 0}


def test_area_to_geodataframe_creates_geodataframe():
    """_area_to_geodataframe returns a GeoDataFrame with a single polygon."""
    gdf = _spatial._area_to_geodataframe(_AREA_FULL)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    bounds = gdf.geometry[0].bounds  # (minx, miny, maxx, maxy)
    assert bounds == (0, 0, 90, 90)


def test_area_to_geodataframe_missing_keys():
    """_area_to_geodataframe raises ValueError if keys are missing."""
    with pytest.raises(ValueError, match="missing required keys"):
        _spatial._area_to_geodataframe({"north": 90, "south": 0})


def test_reduce_with_area_kwarg():
    """spatial.reduce with area= returns correct result."""
    # SAMPLE_ARRAY: lat=[0,60,90], lon=[0,30,60,90], values = rows of 1,2,3
    # area covers lat=0 and lat=60 (mean of 1 and 2 = 1.5)
    # Note: shapely contains_xy excludes points on the boundary of the polygon
    area = {"north": 91, "south": -1, "east": 91, "west": -1}
    result = spatial.reduce(SAMPLE_ARRAY, area=area, how="mean")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.item(), 1.5)


def test_reduce_with_area_subset():
    """spatial.reduce with area= covering a subset returns correct mean."""
    # Only lat=60 (row with value 2), all longitudes
    # shapely.contains_xy is strict (boundary excluded), so use 59-61 range
    area = {"north": 61, "south": 59, "east": 91, "west": -1}
    result = spatial.reduce(SAMPLE_ARRAY, area=area, how="mean")
    np.testing.assert_allclose(result.item(), 2.0)


def test_mask_with_area_kwarg():
    """spatial.mask with area= returns masked data."""
    area = {"north": 91, "south": 59, "east": 91, "west": -1}
    result = spatial.mask(SAMPLE_ARRAY, area=area, chunk=False)
    assert isinstance(result, xr.DataArray)
    # lat=0 row should be all NaN (outside area)
    assert np.all(np.isnan(result.sel(latitude=0).values))
    # lat=60 row should be preserved
    assert np.all(result.sel(latitude=60).values == 2)


def test_area_and_geodataframe_raises_reduce():
    """spatial.reduce raises ValueError if both area and geodataframe are provided."""
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    with pytest.raises(ValueError, match="Only one of"):
        spatial.reduce(SAMPLE_ARRAY, geodataframe=gdf, area=_AREA_FULL, how="mean")


def test_area_and_geodataframe_raises_mask():
    """spatial.mask raises ValueError if both area and geodataframe are provided."""
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    with pytest.raises(ValueError, match="Only one of"):
        spatial.mask(SAMPLE_ARRAY, geodataframe=gdf, area=_AREA_FULL)


def test_area_and_positional_geodataframe_raises_reduce():
    """spatial.reduce raises ValueError if area and positional geodataframe are both given."""
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    with pytest.raises(ValueError, match="Only one of"):
        spatial.reduce(SAMPLE_ARRAY, gdf, area=_AREA_FULL, how="mean")


def test_area_and_positional_geodataframe_raises_mask():
    """spatial.mask raises ValueError if area and positional geodataframe are both given."""
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    with pytest.raises(ValueError, match="Only one of"):
        spatial.mask(SAMPLE_ARRAY, gdf, area=_AREA_FULL)


def test_mask_no_geodataframe_no_area_raises():
    """spatial.mask raises ValueError if neither geodataframe nor area is provided."""
    with pytest.raises(ValueError, match="Either"):
        spatial.mask(SAMPLE_ARRAY)


def test_area_none_passthrough_reduce():
    """spatial.reduce with area=None (default) behaves normally."""
    result = spatial.reduce(SAMPLE_ARRAY, how="mean")
    np.testing.assert_allclose(result.item(), 2.0)


def test_reduce_dataset_with_area():
    """spatial.reduce on a Dataset with area= returns a Dataset."""
    ds = xr.Dataset({"var": SAMPLE_ARRAY})
    area = {"north": 91, "south": -1, "east": 91, "west": -1}
    result = spatial.reduce(ds, area=area, how="mean")
    assert isinstance(result, xr.Dataset)
    assert "var" in result
