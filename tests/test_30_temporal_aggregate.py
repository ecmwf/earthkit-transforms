import numpy as np
import pandas as pd
import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr
from earthkit.data.utils.testing import earthkit_remote_test_data_file

from earthkit import data as ek_data
from earthkit.transforms import temporal

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


def get_data(srcfile: str = "era5_temperature_europe_2015.grib"):
    remote_era5_file = earthkit_remote_test_data_file(srcfile)
    return ek_data.from_source("url", remote_era5_file)


@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_temporal_reduce(in_data, expected_return_type):
    reduced_data = temporal.reduce(in_data, how="mean")
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" not in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data
    reduced_data = temporal.reduce(in_data, how="mean", how_label="mean")
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" not in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t_mean" == reduced_data.name
    else:
        assert "2t_mean" in reduced_data


def test_standardise_time_basic():
    data = get_data().to_xarray()
    original_time = data.forecast_reference_time
    data_standardised = temporal.standardise_time(data)
    np.testing.assert_array_equal(original_time.values, data_standardised.forecast_reference_time.values)


def test_standardise_time_monthly():
    data = get_data().to_xarray()
    data_standardised = temporal.standardise_time(data, target_format="%Y-%m-15")
    assert all(
        pd.to_datetime(time_value).day == 15
        for time_value in data_standardised.forecast_reference_time.values
    )


@pytest.mark.parametrize(
    "how",
    ("mean", "median", "min", "max", "std", "sum"),
)
def test_temporal_reduce_hows(how):
    in_data = get_data().to_xarray()
    expected_return_type = xr.Dataset
    reduced_data = temporal.reduce(in_data, how=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" not in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data

    reduced_data = temporal.reduce(in_data, how=how, how_label=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" not in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert f"2t_{how}" == reduced_data.name
    else:
        assert f"2t_{how}" in reduced_data


@pytest.mark.parametrize(
    "method",
    ("mean", "median", "min", "max", "std", "sum"),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_temporal_methods(method, in_data, expected_return_type):
    reduced_data = temporal.__getattribute__(method)(in_data)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" not in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        # [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_temporal_daily_reduce_intypes(in_data, expected_return_type, how="mean"):
    reduced_data = temporal.daily_reduce(in_data, how=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


@pytest.mark.parametrize(
    "how",
    ("mean", "median", "min", "max", "std", "sum"),
)
def test_temporal_daily_reduce_hows(how, in_data=get_data().to_xarray(), expected_return_type=xr.Dataset):
    reduced_data = temporal.daily_reduce(in_data, how=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        # [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_temporal_monthly_reduce_intypes(in_data, expected_return_type, how="mean"):
    reduced_data = temporal.monthly_reduce(in_data, how=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


@pytest.mark.parametrize(
    "how",
    ("mean", "median", "min", "max", "std", "sum"),
)
def test_temporal_monthly_reduce_hows(how, in_data=get_data().to_xarray(), expected_return_type=xr.Dataset):
    reduced_data = temporal.monthly_reduce(in_data, how=how)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


@pytest.mark.parametrize(
    "method",
    (
        "daily_mean",
        "daily_median",
        "daily_min",
        "daily_max",
        "daily_std",
        "daily_sum",
        "monthly_mean",
        "monthly_median",
        "monthly_min",
        "monthly_max",
        "monthly_std",
        "monthly_sum",
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        # [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_temporal_daily_monthly_methods(method, in_data, expected_return_type):
    reduced_data = temporal.__getattribute__(method)(in_data)
    assert isinstance(reduced_data, expected_return_type)
    assert "forecast_reference_time" in list(reduced_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == reduced_data.name
    else:
        assert "2t" in reduced_data


def test_temporal_daily_reduce_extra_reduce_dims():
    time = pd.date_range("2024-01-01", periods=4, freq="12h")
    dataarray = xr.DataArray(
        [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]],
        dims=("time", "x"),
        coords={"time": time, "x": [0, 1]},
        name="test",
    )

    result = temporal.daily_reduce(dataarray, how="mean", time_dim="time", extra_reduce_dims=["x"])

    assert result.dims == ("time",)
    assert np.allclose(result.values, [4.0, 12.0])


def test_temporal_monthly_reduce_extra_reduce_dims():
    time = pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"])
    dataarray = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dims=("time", "x"),
        coords={"time": time, "x": [0, 1]},
        name="test",
    )

    result = temporal.monthly_reduce(dataarray, how="mean", time_dim="time", extra_reduce_dims=["x"])

    assert result.dims == ("time",)
    assert np.allclose(result.values, [2.5, 6.5])
