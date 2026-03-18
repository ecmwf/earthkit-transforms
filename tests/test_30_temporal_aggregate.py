import numpy as np
import pandas as pd
import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr
from earthkit.data.testing import earthkit_remote_test_data_file

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
    assert all(pd.to_datetime(time_value).day == 15 for time_value in data_standardised.forecast_reference_time.values)


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


# ---------------------------------------------------------------------------
# Local (synthetic-data) tests — no network required
# ---------------------------------------------------------------------------


def _make_hourly_da(n_hours=48, start="2020-01-01"):
    """Two days of hourly data starting at `start`."""
    time = pd.date_range(start, periods=n_hours, freq="h")
    data = np.arange(float(n_hours))
    return xr.DataArray(data, dims=["time"], coords={"time": time}, name="var")


def _make_daily_da(n_days=60, start="2020-01-01"):
    """Two months of daily data (Jan+Feb 2020 by default)."""
    time = pd.date_range(start, periods=n_days, freq="D")
    data = np.arange(float(n_days))
    return xr.DataArray(data, dims=["time"], coords={"time": time}, name="var")


def _make_monthly_da_local(n_months=36, start="2020-01-01"):
    """Three years of monthly data."""
    time = pd.date_range(start, periods=n_months, freq="MS")
    data = np.arange(float(n_months))
    return xr.DataArray(data, dims=["time"], coords={"time": time}, name="var")


# --- temporal.reduce and temporal.method (local) ------------------------------------------------


def test_temporal_reduce_local_mean():
    da = _make_hourly_da()
    result = temporal.reduce(da, how="mean")
    assert isinstance(result, xr.DataArray)
    assert result.dims == ()
    np.testing.assert_allclose(result.values, np.mean(da.values))


def test_temporal_reduce_local_how_label():
    da = _make_hourly_da()
    result = temporal.reduce(da, how="mean", how_label="mean")
    assert result.name == "var_mean"


@pytest.mark.parametrize(
    "freq, expected_length",
    (
        ("D", 90),
        ("dayofyear", 90),
        ("5D", 18),
        ("weekofyear", 14),
        ("day", 90),
        ("MS", 3),
        ("ME", 3),
        ("3MS", 1),
        ("month", 3),
        ("YE", 1),
        ("year", 1),
    ),
)
def test_temporal_reduce_local_frequency_resample(freq, expected_length):
    """temporal.reduce with frequency= should trigger the resample path."""
    da = _make_hourly_da(n_hours=24 * 90)
    result = temporal.reduce(da, how="mean", frequency=freq)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time",)
    assert len(result) == expected_length


@pytest.mark.parametrize(
    "method",
    (
        temporal.mean,
        temporal.median,
        temporal.min,
        temporal.max,
        temporal.std,
        temporal.sum,
    ),
)
@pytest.mark.parametrize(
    "freq, expected_length",
    (
        ("D", 90),
        ("dayofyear", 90),
        ("5D", 18),
        ("weekofyear", 14),
        ("day", 90),
        ("MS", 3),
        ("ME", 3),
        ("3MS", 1),
        ("month", 3),
        ("YE", 1),
        ("YE", 1),
        ("year", 1),
    ),
)
def test_temporal_methods_local_frequency_resample(method, freq, expected_length):
    """temporal.reduce with frequency= should trigger the resample path."""
    da = _make_hourly_da(n_hours=24 * 90)
    result = method(da, how="mean", frequency=freq)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time",)
    assert len(result) == expected_length


# --- temporal.standardise_time (local) -------------------------------------


def test_standardise_time_local_preserves_values():
    da = _make_monthly_da_local()
    result = temporal.standardise_time(da)
    # Default format keeps full datetime; values should be unchanged
    np.testing.assert_array_equal(result.values, da.values)


def test_standardise_time_local_monthly_format():
    da = _make_monthly_da_local()
    result = temporal.standardise_time(da, target_format="%Y-%m-15")
    days = [pd.Timestamp(t).day for t in result.time.values]
    assert all(d == 15 for d in days)


# --- temporal.daily_reduce (local) -----------------------------------------


@pytest.mark.parametrize("how", ("mean", "min", "max", "std", "sum"))
def test_daily_reduce_local(how):
    da = _make_hourly_da()
    result = temporal.daily_reduce(da, how=how)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time",)
    assert len(result) == 2  # 48 hours = 2 days


def test_daily_reduce_local_returns_correct_mean():
    da = _make_hourly_da()
    result = temporal.daily_reduce(da, how="mean")
    # First day: hours 0-23 → mean = 11.5; second day: hours 24-47 → mean = 35.5
    np.testing.assert_allclose(result.values, [11.5, 35.5])


# --- temporal.monthly_reduce (local) ---------------------------------------


@pytest.mark.parametrize("how", ("mean", "min", "max", "std", "sum"))
def test_monthly_reduce_local(how):
    da = _make_daily_da()
    result = temporal.monthly_reduce(da, how=how)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time",)
    assert len(result) == 2  # Jan + Feb


def test_monthly_reduce_local_returns_correct_mean():
    da = _make_daily_da()
    result = temporal.monthly_reduce(da, how="mean")
    # Jan: days 0-30 (31 days) → mean = 15.0; Feb 2020: days 31-59 (29 days) → mean = 45.0
    assert len(result) == 2
    np.testing.assert_allclose(result.isel(time=0).values, np.mean(np.arange(31.0)))
    np.testing.assert_allclose(result.isel(time=1).values, np.mean(np.arange(31.0, 60.0)))


# --- temporal.rolling_reduce (local) ----------------------------------------


def test_rolling_reduce_local():
    da = _make_daily_da(n_days=10)
    result = temporal.rolling_reduce(da, window_length=3, how_reduce="mean")
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time",)
    assert len(result) == len(da)
