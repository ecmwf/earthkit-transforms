import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# from earthkit.data.core.temporary import temp_directory
from earthkit import data as ek_data
from earthkit.data.testing import earthkit_remote_test_data_file
from earthkit.transforms import spatial
from earthkit.transforms.spatial import _aggregate as _spatial
from shapely.geometry import Polygon

try:
    import rasterio  # noqa: F401

    rasterio_available = True
except ImportError:
    rasterio_available = False

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


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
    return ek_data.from_source("url", remote_era5_file)


def get_shape_data():
    if rasterio_available:
        remote_nuts_url = earthkit_remote_test_data_file("NUTS_RG_60M_2021_4326_LEVL_0.geojson")
        return ek_data.from_source("url", remote_nuts_url)
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
    assert len(masked_data["index"]) == len(nuts_data)


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
    assert len(reduced_data["index"]) == len(nuts_data)


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
    mask = xr.DataArray(
        np.random.randint(0, 2, size=dataarray.shape), coords=dataarray.coords, dims=dataarray.dims
    )
    result = _spatial._reduce_dataarray_as_xarray(dataarray, mask_arrays=[mask], how="sum")
    assert isinstance(result, xr.DataArray)


def test_return_geometry_as_coord():
    dataarray = create_test_dataarray()
    geodataframe = create_test_geodataframe()
    result = _spatial._reduce_dataarray_as_xarray(
        dataarray, geodataframe=geodataframe, return_geometry_as_coord=True
    )
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
    result = _spatial._reduce_dataarray_as_pandas(
        dataarray, geodataframe=geodataframe, compact=True, how="mean"
    )
    assert isinstance(result, pd.DataFrame)
    assert f"{dataarray.name}" in result.columns


def test_reduce_as_pandas_without_geodataframe():
    dataarray = create_test_dataarray()
    result = _spatial._reduce_dataarray_as_pandas(dataarray, how="sum")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
