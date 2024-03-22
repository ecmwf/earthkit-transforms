import numpy as np
import pytest
import xarray as xr

from earthkit.aggregate.general import _reduce_dataarray, _rolling_reduce_dataarray, reduce, rolling_reduce
from earthkit.aggregate.tools import nanaverage


# Helper function to create a dummy DataArray
def create_dataarray(weights=None):
    data = xr.DataArray([[1, np.nan], [1, 2]], dims=("x", "y"))
    if weights is not None:
        data = data.weighted(weights=weights)
    return data


@pytest.mark.parametrize(
    "how, expected_result_method",
    (
        ["mean", np.nanmean],
        ["numpy.nanmean", np.nanmean],
        ["nanaverage", nanaverage],
        ["average", nanaverage],
        ["stddev", np.nanstd],
        ["std", np.nanstd],
        ["stdev", np.nanstd],
        ["sum", np.nansum],
        ["max", np.nanmax],
        ["min", np.nanmin],
        ["median", np.nanmedian],
    ),
)
# Test case for reducing without weights using built-in method
def test_reduce_dataarray(how, expected_result_method):
    dataarray = create_dataarray()
    result = _reduce_dataarray(dataarray, how=how)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ()  # Ensure reduced to scalar
    assert result.values == expected_result_method(dataarray.values)

    result = reduce(dataarray, how)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ()  # Ensure reduced to scalar
    assert result.values == expected_result_method(dataarray.values)

    result = reduce(xr.Dataset({"test": dataarray}), how)
    assert isinstance(result, xr.Dataset)
    assert result[f"test_{how}"].shape == ()  # Ensure reduced to scalar
    assert result[f"test_{how}"].values == expected_result_method(dataarray.values)


@pytest.mark.parametrize(
    "how",
    (
        "numpy.mean",
        "numpy.average",
    ),
)
# Test case for reducing with methods that should not ignore nans
def test_reduce_dataarray_nan(how):
    dataarray = create_dataarray()
    result = _reduce_dataarray(dataarray, how=how)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ()  # Ensure reduced to scalar
    assert np.isnan(result.values)


@pytest.mark.parametrize(
    "how, expected_result_method",
    (
        ["mean", lambda x: x.mean()],
        ["nanmean", lambda x: x.mean()],
        ["average", lambda x: x.mean()],
        ["stddev", lambda x: x.std()],
        ["std", lambda x: x.std()],
        ["stdev", lambda x: x.std()],
    ),
)
# Test case for reducing with weights
def test_reduce_dataarray_with_weights(how, expected_result_method):
    dataarray = create_dataarray()
    weights = xr.DataArray([1, 2], dims="x", name="x")  # Dummy weights
    weighted_dataarray = create_dataarray(weights=weights)

    result = _reduce_dataarray(dataarray, how=how, weights=weights)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ()  # Ensure reduced to scalar
    assert result.values == expected_result_method(weighted_dataarray)

    result = reduce(dataarray, how, weights=weights)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ()  # Ensure reduced to scalar
    assert result.values == expected_result_method(weighted_dataarray)

    result = reduce(xr.Dataset({"test": dataarray}), how, weights=weights)
    assert isinstance(result, xr.Dataset)
    assert result[f"test_{how}"].shape == ()  # Ensure reduced to scalar
    assert result[f"test_{how}"].values == expected_result_method(weighted_dataarray)


# Helper function to create a dummy DataArray
def create_dataarray_rolling(weights=None):
    data = xr.DataArray(np.arange(25).reshape([5, 5]), dims=("x", "y"), name="test")
    if weights is not None:
        data = data.weighted(weights=weights)
    return data


@pytest.mark.parametrize(
    "how, expected_result",
    (
        ["mean", 2.5],
        ["numpy.nanmean", 2.5],
        ["nanaverage", 2.5],
        ["average", 2.5],
        ["stddev", 2.5],
        ["std", 2.5],
        ["stdev", 2.5],
        ["sum", 5],
        ["max", 5],
        ["min", 0],
        ["median", 2.5],
    ),
)
# Test case for reducing without weights using built-in method
def test_rolling_reduce_dataarray(how, expected_result):
    dataarray = create_dataarray_rolling()
    result = _rolling_reduce_dataarray(
        dataarray, how_reduce=how, x=2, center=True, chunk=False, how_dropna="any"
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == (4, 5)
    assert result.values[0, 0] == expected_result

    result = rolling_reduce(dataarray, how_reduce=how, x=2, center=True, chunk=False, how_dropna="any")
    assert isinstance(result, xr.DataArray)
    assert result.shape == (4, 5)
    assert result.values[0, 0] == expected_result

    result = rolling_reduce(
        xr.Dataset({"test": dataarray}), how_reduce=how, x=2, center=True, chunk=False, how_dropna="any"
    )
    assert isinstance(result, xr.Dataset)
    assert result[f"test_{how}"].shape == (4, 5)
    assert result[f"test_{how}"].values[0, 0] == expected_result
