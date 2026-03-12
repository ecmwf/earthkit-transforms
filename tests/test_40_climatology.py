import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr
from earthkit.data.utils.testing import earthkit_remote_test_data_file

from earthkit import data as ek_data
from earthkit.transforms import climatology

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


def get_data():
    remote_era5_file = earthkit_remote_test_data_file("era5_temperature_france_2015_2016_2017_3deg.grib")
    return ek_data.from_source("url", remote_era5_file)


@pytest.mark.parametrize(
    "method",
    (
        climatology.mean,
        climatology.median,
        climatology.min,
        climatology.max,
        climatology.std,
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
def test_climatology_base(in_data, expected_return_type, method):
    clim = method(in_data)
    assert isinstance(clim, expected_return_type)
    assert "year" in list(clim.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == clim.name
    else:
        assert "2t" in clim

    # Check alternate frequencies
    for freq in ["month", "dayofyear"]:
        clim = method(in_data, frequency=freq)
        assert freq in list(clim.dims)


@pytest.mark.parametrize(
    "clim_method",
    (
        climatology.mean,
        climatology.median,
    ),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        [get_data().to_xarray(), xr.Dataset],
        [get_data().to_xarray()["2t"], xr.DataArray],
    ),
)
def test_anomaly_base(in_data, expected_return_type, clim_method):
    clim_m = clim_method(in_data)
    anom_m = climatology.anomaly(in_data, clim_m)

    assert isinstance(anom_m, expected_return_type)
    # Dimensions of the anomaly should be the same as the input data
    assert all(dim in list(anom_m.dims) for dim in in_data.dims)
    if expected_return_type == xr.DataArray:
        assert "2t" == anom_m.name
    else:
        assert "2t" in anom_m

    # Check alternate frequencies
    for freq in ["month", "dayofyear"]:
        clim_m = clim_method(in_data, frequency=freq)
        anom_m = climatology.anomaly(in_data, clim_m, frequency=freq)
        # Dimensions of the anomaly should be the same as the input data
        assert all(dim in list(anom_m.dims) for dim in in_data.dims)


# @pytest.mark.parametrize(
#     "time_string, expected_dim",
#     (
#         ("D", "dayofyear"),
#         ("M", "month"),
#         ("Y", "year"),
#     )
# )
# @pytest.mark.parametrize(
#     "method",
#     (
#         climatology.mean,
#         climatology.median,
#     ),
# )
# def test_climatology_pandas_time_strings(method, time_string, expected_dim):
#     in_data = get_data().to_xarray()
#     clim = method(in_data, frequency=time_string)
#     assert expected_dim in list(clim.dims)

#     # Check alternate frequencies
#     for dim in ["month", "dayofyear"]:
#         clim = method(in_data, frequency=dim)
#         assert dim in list(clim.dims)


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
