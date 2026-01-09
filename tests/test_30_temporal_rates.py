import numpy as np
import pytest

# from earthkit.data.core.temporary import temp_directory
from earthkit import data as ek_data
from earthkit.data.testing import earthkit_remote_test_data_file
from earthkit.transforms import temporal

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")

TO_XARRAY_KWARGS = {
    "valid_time": {"time_dim_mode": "valid_time"},
    "forecast": {
        "time_dim_mode": "forecast",
        "allow_holes": True,
        "ensure_dims": ["forecast_reference_time"],
    },
}
ACCUM_TIME_DIM = {"forecast": "step"}
RATE_SUFFIX = {
    "step_length": "per_step",
}
ERA5_LAND_TEST_POINT: dict[str, int] = {"latitude": 10, "longitude": 18}
SEAS5_TEST_POINT: dict[str, int] = {"latitude": 10, "longitude": 17}


def get_data(srcfile: str = "era5_temperature_europe_2015.grib"):
    remote_era5_file = earthkit_remote_test_data_file(srcfile)
    return ek_data.from_source("url", remote_era5_file)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
def test_accumulation_to_rate_base(time_dim_mode):
    test_file = "era5-sfc-precip-3deg-202401.grib"
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data)
    assert "tp_rate" == rate_data.name
    assert original_units + " s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs

    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, **{accum_time_dim: slice(0, 5)})
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (data.isel(latitude=5, longitude=5, **{accum_time_dim: slice(0, 5)}).values) / (
        3600.0
    )  # seconds in an hour
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("seconds", " s^-1", 3600.0),
        ("minutes", " min^-1", 60.0),
        ("hours", " hours^-1", 1.0),
        ("3 hours", " (3 hours)^-1", 1.0 / 3.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_step_rate_units(rate_units, expected_units, sample_sf, time_dim_mode):
    test_file = "era5-sfc-precip-3deg-202401.grib"
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data, rate_units=rate_units)
    assert f"tp_{RATE_SUFFIX.get(rate_units, 'rate')}" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(f" {RATE_SUFFIX.get(rate_units, 'rate').replace('_', ' ')}")
    assert "standard_name" not in rate_data.attrs

    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    numeric_test_sample = rate_data.isel(latitude=5, longitude=5, **{accum_time_dim: slice(0, 5)})
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (data.isel(latitude=5, longitude=5, **{accum_time_dim: slice(0, 5)}).values) / (
        sample_sf
    )
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)
    assert not np.all(np.isnan(numeric_test_sample.values)), "Array contains only NaN values"


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
def test_accumulation_to_rate_start_of_forecast(time_dim_mode):
    test_file = "seas5-precip-3deg-202401.grib"
    accumulation_type = "start_of_forecast"
    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(data, accumulation_type=accumulation_type)
    assert "tp_rate" == rate_data.name
    assert original_units + " s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data[accum_time_dim][0].values == data[accum_time_dim][0].values
    isel_kwargs = {k: 0 for k in data.dims if k not in (accum_time_dim, "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(
        **SEAS5_TEST_POINT, **{accum_time_dim: slice(1, 5)}, **isel_kwargs
    )
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (
        data.isel(**SEAS5_TEST_POINT, **{accum_time_dim: slice(1, 5)}, **isel_kwargs).values
        - data.isel(**SEAS5_TEST_POINT, **{accum_time_dim: slice(0, 4)}, **isel_kwargs).values
    ) / (3600.0 * 24)  # 24 hours in seconds
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

    # Chack the from_first_step=False option, should drop first timestep
    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type="start_of_forecast", from_first_step=False
    )
    assert "tp_rate" == rate_data.name
    assert original_units + " s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data[accum_time_dim][0].values == data[accum_time_dim][1].values
    assert not np.all(np.isnan(numeric_test_sample.values)), "Array contains only NaN values"


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("minutes", " min^-1", 60.0 * 24),
        ("hours", " hours^-1", 24.0),
        ("3 hours", " (3 hours)^-1", 8.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_forecast_rate_units(
    rate_units, expected_units, sample_sf, time_dim_mode
):
    test_file = "seas5-precip-3deg-202401.grib"
    accumulation_type = "start_of_forecast"
    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]
    original_units = data.attrs["units"]

    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type=accumulation_type, rate_units=rate_units
    )
    assert f"tp_{RATE_SUFFIX.get(rate_units, 'rate')}" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(f" {RATE_SUFFIX.get(rate_units, 'rate').replace('_', ' ')}")
    assert "standard_name" not in rate_data.attrs

    isel_kwargs = {k: 0 for k in data.dims if k not in (accum_time_dim, "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(
        **SEAS5_TEST_POINT, **{accum_time_dim: slice(1, 5)}, **isel_kwargs
    )
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (
        data.isel(**SEAS5_TEST_POINT, **{accum_time_dim: slice(1, 5)}, **isel_kwargs).values
        - data.isel(**SEAS5_TEST_POINT, **{accum_time_dim: slice(0, 4)}, **isel_kwargs).values
    ) / sample_sf
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
def test_accumulation_to_rate_start_of_day(time_dim_mode):
    test_file = "era5-land-precip-3deg-202401.grib"
    accumulation_type = "start_of_day"
    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]

    original_units = data.attrs["units"]
    rate_data = temporal.accumulation_to_rate(data, accumulation_type=accumulation_type)
    assert "tp_rate" == rate_data.name
    assert original_units + " s^-1" == rate_data.attrs["units"]
    assert "standard_name" not in rate_data.attrs
    assert rate_data[accum_time_dim][0].values == data[accum_time_dim][0].values

    isel_kwargs = {k: 1 for k in data.dims if k not in (accum_time_dim, "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(
        **ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(2, 22)}, **isel_kwargs
    )
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (
        data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(2, 22)}, **isel_kwargs).values
        - data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(1, 21)}, **isel_kwargs).values
    ) / 3600.0  # hours to seconds
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

    if time_dim_mode == "valid_time":
        # Check a value for the first timestep of a day, in this example at 01:00
        assert data.valid_time[25].values.astype("datetime64[h]").item().hour == 1
        # the rate can be computed
        numeric_test_sample = rate_data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: 25}, **isel_kwargs)
        assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
        assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
        expected_sample = (
            data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: 25}, **isel_kwargs).values
        ) / 3600.0
        np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

        # Check a period that is not the first day
        numeric_test_sample = rate_data.isel(
            **ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(26, 31)}, **isel_kwargs
        )
        assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
        assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
        expected_sample = (
            data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(26, 31)}, **isel_kwargs).values
            - data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(25, 30)}, **isel_kwargs).values
        ) / 3600.0  # hours in seconds
        np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
@pytest.mark.parametrize(
    "rate_units, expected_units, sample_sf",
    [
        ("seconds", " s^-1", 3600.0),
        ("minutes", " min^-1", 60.0),
        ("hours", " hours^-1", 1.0),
        ("3 hours", " (3 hours)^-1", 1.0 / 3.0),
        ("step_length", "", 1.0),
    ],
)
def test_accumulation_to_rate_start_of_day_rate_units(time_dim_mode, rate_units, expected_units, sample_sf):
    test_file = "era5-land-precip-3deg-202401.grib"
    accumulation_type = "start_of_day"
    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]

    original_units = data.attrs["units"]
    rate_data = temporal.accumulation_to_rate(
        data, accumulation_type=accumulation_type, rate_units=rate_units
    )
    assert f"tp_{RATE_SUFFIX.get(rate_units, 'rate')}" == rate_data.name
    assert original_units + expected_units == rate_data.attrs["units"]
    assert rate_data.attrs["long_name"].endswith(f" {RATE_SUFFIX.get(rate_units, 'rate').replace('_', ' ')}")
    assert "standard_name" not in rate_data.attrs
    assert rate_data[accum_time_dim][0].values == data[accum_time_dim][0].values
    isel_kwargs = {k: 1 for k in data.dims if k not in (accum_time_dim, "latitude", "longitude")}
    numeric_test_sample = rate_data.isel(
        **ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(6, 11)}, **isel_kwargs
    )
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (
        data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(6, 11)}, **isel_kwargs).values
        - data.isel(**ERA5_LAND_TEST_POINT, **{accum_time_dim: slice(5, 10)}, **isel_kwargs).values
    ) / sample_sf
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)


@pytest.mark.parametrize(
    "time_dim_mode",
    ("forecast", "valid_time"),
)
def test_deaccumulate(time_dim_mode):
    test_file = "seas5-precip-3deg-202401.grib"
    data = get_data(test_file).to_xarray(**TO_XARRAY_KWARGS[time_dim_mode])["tp"]
    accum_time_dim = ACCUM_TIME_DIM.get(time_dim_mode, time_dim_mode)
    original_units = data.attrs["units"]

    deaccum_data = temporal.deaccumulate(data)
    assert "tp_per_step" == deaccum_data.name
    assert original_units == deaccum_data.attrs["units"]
    assert deaccum_data.attrs["long_name"].endswith(" per step")
    assert "standard_name" not in deaccum_data.attrs

    isel_kwargs = {k: 0 for k in data.dims if k not in (accum_time_dim, "latitude", "longitude")}

    # Check first timestep is unchanged
    np.testing.assert_allclose(
        deaccum_data.isel(**{accum_time_dim: 0}).values,
        data.isel(**{accum_time_dim: 0}).values
    )

    numeric_test_sample = deaccum_data.isel(
        **{accum_time_dim: slice(1, 5)}, **SEAS5_TEST_POINT, **isel_kwargs
    )
    assert not np.all(np.isnan(numeric_test_sample.values)), "Sample array contains only NaN values"
    assert not np.all(numeric_test_sample.values == 0), "Sample array contains only zero values"
    expected_sample = (
        data.isel(**{accum_time_dim: slice(1, 5)}, **SEAS5_TEST_POINT, **isel_kwargs).values
        - data.isel(**{accum_time_dim: slice(0, 4)}, **SEAS5_TEST_POINT, **isel_kwargs).values
    )
    np.testing.assert_allclose(numeric_test_sample.values, expected_sample)

