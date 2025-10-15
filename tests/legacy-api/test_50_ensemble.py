# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
import xarray as xr
from earthkit.transforms.aggregate.ensemble import (
    max,
    mean,
    min,
    reduce,
)


# Sample data for tests
@pytest.fixture
def sample_data():
    data = xr.DataArray(
        np.arange(12).reshape(3, 2, 2),
        dims=("ensemble_member", "lat", "lon"),
        coords={"ensemble_member": [0, 1, 2]},
    )
    return data


def test_reduce_raises_if_no_ensemble_dim_found():
    data = xr.DataArray(np.arange(8).reshape(2, 2, 2), dims=("foo", "lat", "lon"))
    with pytest.raises(ValueError, match="Unable to find dimension key for axis 'realization' in dataarray"):
        reduce(data)


def test_mean_actual_reduction(sample_data):
    # This tests integration if you want to use real data (no mocking)
    result = mean(sample_data, dim="ensemble_member")
    expected = sample_data.mean(dim="ensemble_member")
    assert np.allclose(result, expected)


def test_min_max_actual(sample_data):
    result_min = min(sample_data, dim="ensemble_member")
    expected_min = sample_data.min(dim="ensemble_member")
    assert np.allclose(result_min, expected_min)

    result_max = max(sample_data, dim="ensemble_member")
    expected_max = sample_data.max(dim="ensemble_member")
    assert np.allclose(result_max, expected_max)
