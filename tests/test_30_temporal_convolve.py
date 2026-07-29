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
