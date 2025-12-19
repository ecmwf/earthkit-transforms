import numpy as np
import pandas as pd
import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr
from earthkit import data as ek_data
from earthkit.data.testing import earthkit_remote_test_data_file
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


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
def test_accumulation_to_rate_base(time_dim_mode):
    test_file = "era5-sfc-precip-3deg-202401.grib"
    # data = get_data(test_file).to_xarray()
    # Check with DataArray
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode=time_dim_mode)["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data)
    assert "tp_rate" == rate_data.name
    assert original_units + "s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs

    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, valid_time=slice(0, 5))
    expected_sample = (data.isel(latitude=5, longitude=5, valid_time=slice(0, 5)).values) / (
        3600.0
    )  # seconds in an hour
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast_time", "valid_time"),
)
@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("seconds", "s^-1", 3600.0),
        ("minutes", "min^-1", 60.0),
        ("hours", "hours^-1", 1.0),
        ("3 hours", "(3 hours)^-1", 1.0 / 3.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_step_rate_units(time_dim_mode, rate_units, expected_units, sample_sf):
    test_file = "era5-sfc-precip-3deg-202401.grib"
    # accumulation_type = "start_of_step"  # default value
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode=time_dim_mode)["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data, rate_units=rate_units)
    assert "tp_rate" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(" rate")
    assert "standard_name" not in rate_data.attrs

    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, valid_time=slice(0, 5))
    expected_sample = (data.isel(latitude=5, longitude=5, valid_time=slice(0, 5)).values) / (sample_sf)
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast_time", "valid_time"),
)
def test_accumulation_to_rate_start_of_forecast(time_dim_mode):
    test_file = "seas5-precip-3deg-202401.grib"
    accumulation_type = "start_of_forecast"
    # data = get_data("seas5_precipitation_europe_2025.grib").to_xarray()
    # Check with DataArray
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode=time_dim_mode)["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data, accumulation_type=accumulation_type)
    assert "tp_rate" == rate_data.name
    assert original_units + "s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data.valid_time[0].values == data.valid_time[0].values
    isel_kwargs = {k: 0 for k in data.dims if k not in ("valid_time", "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(latitude=10, longitude=17, valid_time=slice(1, 5), **isel_kwargs)
    expected_sample = (
        data.isel(latitude=10, longitude=17, valid_time=slice(1, 5), **isel_kwargs).values
        - data.isel(latitude=10, longitude=17, valid_time=slice(0, 4), **isel_kwargs).values
    ) / (3600.0 * 24)  # 24 hours in seconds
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type="start_of_forecast", from_first_step=False
    )
    assert "tp_rate" == rate_data.name
    assert original_units + "s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data.valid_time[0].values == data.valid_time[1].values


@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("minutes", "min^-1", 60.0 * 24),
        ("hours", "hours^-1", 24.0),
        ("3 hours", "(3 hours)^-1", 8.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_forecast_rate_units(rate_units, expected_units, sample_sf):
    test_file = "seas5-precip-3deg-202401.grib"
    accumulation_type = "start_of_forecast"
    # data = get_data("seas5_precipitation_europe_2025.grib").to_xarray()
    # Check with DataArray
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode="valid_time")["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type=accumulation_type, rate_units=rate_units
    )
    assert "tp_rate" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(" rate")
    assert "standard_name" not in rate_data.attrs

    isel_kwargs = {k: 0 for k in data.dims if k not in ("valid_time", "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(latitude=10, longitude=17, valid_time=slice(1, 5), **isel_kwargs)
    expected_sample = (
        data.isel(latitude=10, longitude=17, valid_time=slice(1, 5), **isel_kwargs).values
        - data.isel(latitude=10, longitude=17, valid_time=slice(0, 4), **isel_kwargs).values
    ) / sample_sf
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


def test_accumulation_to_rate_start_of_day():
    test_file = "era5-land-precip-3deg-202401.grib"
    accumulation_type = "start_of_day"
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode="valid_time")["tp"]
    original_units = data.attrs["units"]
    rate_data = temporal.accumulation_to_rate(data, accumulation_type=accumulation_type)
    assert "tp_rate" == rate_data.name
    assert original_units + "s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data.valid_time[0].values == data.valid_time[0].values

    isel_kwargs = {k: 0 for k in data.dims if k not in ("valid_time", "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, valid_time=slice(26, 31), **isel_kwargs)
    expected_sample = (
        data.isel(latitude=5, longitude=5, valid_time=slice(26, 31), **isel_kwargs).values
        - data.isel(latitude=5, longitude=5, valid_time=slice(25, 30), **isel_kwargs).values
    ) / (3600.0 * 24)  # 24 hours in seconds
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

    # Check a value for the first timestep of a day, in this example at 01:00
    assert data.valid_time[25].values.astype("datetime64[h]").item().hour == 1
    # the rate can be computed
    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, valid_time=25, **isel_kwargs)
    expected_sample = (data.isel(latitude=5, longitude=5, valid_time=25, **isel_kwargs).values) / (
        3600.0 * 24
    )
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("minutes", "min^-1", 60.0 * 24),
        ("hours", "hours^-1", 24.0),
        ("3 hours", "(3 hours)^-1", 8.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_day_rate_units(rate_units, expected_units, sample_sf):
    test_file = "era5-land-precip-3deg-202401.grib"
    accumulation_type = "start_of_day"
    data = ek_data.from_source(
        "file",
        "/Users/edwardcomyn-platt/Work/Git_Repositories/EARTHKIT/earthkit-transforms/docs/notebooks/test_data/"
        + test_file,
    ).to_xarray(time_dim_mode="valid_time")["tp"]
    original_units = data.attrs["units"]
    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type=accumulation_type, rate_units=rate_units
    )
    assert "tp_rate" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(" rate")
    assert "standard_name" not in rate_data.attrs
    assert rate_data.valid_time[0].values == data.valid_time[1].values
    isel_kwargs = {k: 0 for k in data.dims if k not in ("valid_time", "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, valid_time=slice(26, 31), **isel_kwargs)
    expected_sample = (
        data.isel(latitude=5, longitude=5, valid_time=slice(26, 31), **isel_kwargs).values
        - data.isel(latitude=5, longitude=5, valid_time=slice(25, 30), **isel_kwargs).values
    ) / sample_sf
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)
