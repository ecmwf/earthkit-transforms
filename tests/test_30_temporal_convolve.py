import numpy as np
import pytest
import xarray as xr

from earthkit import data as ek_data
from earthkit.transforms import temporal
from earthkit.transforms._tools import earthkit_remote_test_data_file

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
def test_temporal_convolve(in_data, expected_return_type):
    window = np.asarray([1.0, 1.0, 2.0, 1.0, 1.0])
    convolved_data = temporal.convolve(in_data, window)
    assert isinstance(convolved_data, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "2t" == convolved_data.name
    else:
        assert "2t" in convolved_data
        # Test dataarray from here
        convolved_data = convolved_data["2t"]
    assert convolved_data.dims == ("forecast_reference_time", "latitude", "longitude")


@pytest.fixture
def time_series():
    return xr.DataArray(np.ones(20, dtype=float), dims=["time"], coords={"time": np.arange(20)})


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_temporal_convolve_remove_partial_periods_bounds(time_series, k):
    window = np.ones(k, dtype=float) / k  # normalised
    result = temporal.convolve(time_series, window, remove_partial_periods=True)
    assert result.sizes["time"] == time_series.sizes["time"] - k + 1
    np.testing.assert_array_almost_equal(result.values, 1.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_temporal_convolve_keeps_full_bounds_by_default(time_series, k):
    window = np.ones(k, dtype=float)
    result = temporal.convolve(time_series, window, time_dim="time")
    assert result.sizes["time"] == time_series.sizes["time"]
    assert result.time.values[0] == time_series.time.values[0]
    assert result.time.values[-1] == time_series.time.values[-1]
