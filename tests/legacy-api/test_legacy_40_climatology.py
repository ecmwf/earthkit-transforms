import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr
from earthkit import data as ek_data
from earthkit.data.testing import earthkit_remote_test_data_file
from earthkit.transforms.aggregate import climatology

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


def get_data():
    remote_era5_file = earthkit_remote_test_data_file("era5_temperature_france_2015_2016_2017_3deg.grib")
    return ek_data.from_source("url", remote_era5_file)


@pytest.mark.parametrize(
    "method, how",
    (
        (climatology.monthly_mean, "mean"),
        (climatology.monthly_median, "median"),
        (climatology.monthly_min, "min"),
        (climatology.monthly_max, "max"),
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_climatology_monthly(in_data, expected_return_type, method, how):
    clim_m = method(in_data)
    assert isinstance(clim_m, expected_return_type)
    assert "month" in list(clim_m.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == clim_m.name
    else:
        assert "2t" in clim_m

    clim_m = method(in_data, how_label=how)
    assert isinstance(clim_m, expected_return_type)
    assert "month" in list(clim_m.dims)
    if expected_return_type == xr.DataArray:
        assert f"2t_{how}" == clim_m.name
    else:
        assert f"2t_{how}" in clim_m


@pytest.mark.parametrize(
    "method, how",
    (
        (climatology.daily_mean, "mean"),
        # (climatology.daily_median, "median"),
        # (climatology.daily_min, "min"),
        (climatology.daily_max, "max"),
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        # [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_climatology_daily(in_data, expected_return_type, method, how):
    clim_d = method(in_data)
    assert isinstance(clim_d, expected_return_type)
    assert "dayofyear" in list(clim_d.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == clim_d.name
    else:
        assert "2t" in clim_d

    clim_d = method(in_data, how_label=how)
    assert isinstance(clim_d, expected_return_type)
    assert "dayofyear" in list(clim_d.dims)
    if expected_return_type == xr.DataArray:
        assert f"2t_{how}" == clim_d.name
    else:
        assert f"2t_{how}" in clim_d


@pytest.mark.parametrize(
    "clim_method",
    (
        climatology.monthly_mean,
        climatology.monthly_median,
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        # [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_anomaly_monthly(in_data, expected_return_type, clim_method):
    clim_m = clim_method(in_data)
    anom_m = climatology.anomaly(in_data, clim_m, frequency="month")
    assert isinstance(anom_m, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "2t" == anom_m.name
    else:
        assert "2t" in anom_m
    anom_m = climatology.anomaly(in_data, clim_m, frequency="month", how_label="anomaly")
    assert isinstance(anom_m, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "2t_anomaly" == anom_m.name
    else:
        assert "2t_anomaly" in anom_m


@pytest.mark.parametrize(
    "clim_method",
    (
        climatology.daily_mean,
        climatology.daily_median,
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        # [get_data(), xr.Dataset],
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_anomaly_daily(in_data, expected_return_type, clim_method):
    clim_d = clim_method(in_data)
    anom_d = climatology.anomaly(in_data, clim_d, frequency="dayofyear")
    assert isinstance(anom_d, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "2t" == anom_d.name
    else:
        assert "2t" in anom_d
    anom_d = climatology.anomaly(in_data, clim_d, frequency="dayofyear", how_label="anomaly")
    assert isinstance(anom_d, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "2t_anomaly" == anom_d.name
    else:
        assert "2t_anomaly" in anom_d
