import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr

from earthkit import data as ek_data
from earthkit.aggregate import spatial
from earthkit.data.testing import earthkit_remote_test_data_file

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


def get_grid_data():
    remote_era5_file = earthkit_remote_test_data_file("test-data", "era5_temperature_europe_20150101.grib")
    return ek_data.from_source("url", remote_era5_file)


def get_shape_data():
    remote_nuts_url = earthkit_remote_test_data_file("test-data", "NUTS_RG_60M_2021_4326_LEVL_0.geojson")
    return ek_data.from_source("url", remote_nuts_url)


def test_spatial_mask():
    single_masked_data = spatial.mask(get_grid_data(), get_shape_data())
    assert isinstance(single_masked_data, xr.Dataset)


@pytest.mark.parametrize(
    "era5_data, nuts_data, expected_result_type",
    (
        [get_grid_data(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data().to_xarray().t2m, get_shape_data(), xr.DataArray],
    ),
)
def test_spatial_masks_with_ek_objects(era5_data, nuts_data, expected_result_type):
    masked_data = spatial.masks(era5_data, nuts_data)
    assert isinstance(masked_data, expected_result_type)
    assert "index" in masked_data.dims
    assert len(masked_data["index"]) == len(nuts_data)


@pytest.mark.parametrize(
    "era5_data, expected_result_type",
    (
        [get_grid_data(), xr.Dataset],
        [get_grid_data().to_xarray(), xr.Dataset],
        [get_grid_data().to_xarray().t2m, xr.DataArray],
    ),
)
def test_spatial_reduce_no_geometry(era5_data, expected_result_type):
    reduced_data = spatial.reduce(era5_data)

    assert isinstance(reduced_data, expected_result_type)
    assert list(reduced_data.dims) == ["time"]


@pytest.mark.parametrize(
    "era5_data, nuts_data, expected_result_type",
    (
        [get_grid_data(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data(), xr.Dataset],
        [get_grid_data().to_xarray(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data(), get_shape_data().to_pandas(), xr.Dataset],
        [get_grid_data().to_xarray().t2m, get_shape_data(), xr.DataArray],
    ),
)
def test_spatial_reduce_with_geometry(era5_data, nuts_data, expected_result_type):
    reduced_data = spatial.reduce(era5_data, nuts_data)
    assert isinstance(reduced_data, expected_result_type)
    assert all([dim in ["time", "index"] for dim in reduced_data.dims])
    assert len(reduced_data["index"]) == len(nuts_data)
