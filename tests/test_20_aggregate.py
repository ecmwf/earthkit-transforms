import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earthkit.transforms import reduce, rolling_reduce
from earthkit.transforms._aggregate import (
    _reduce_dataarray,
    _rolling_reduce_dataarray,
    resample,
)
from earthkit.transforms._tools import nanaverage


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
    assert result["test"].shape == ()  # Ensure reduced to scalar
    assert result["test"].values == expected_result_method(dataarray.values)

    result = reduce(xr.Dataset({"test": dataarray}), how, how_label=how)
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
    assert result["test"].shape == ()  # Ensure reduced to scalar
    assert result["test"].values == expected_result_method(weighted_dataarray)

    result = reduce(xr.Dataset({"test": dataarray}), how, weights=weights, how_label=how)
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
    result = _rolling_reduce_dataarray(dataarray, how_reduce=how, x=2, center=True, chunk=False, how_dropna="any")
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
    assert result["test"].shape == (4, 5)
    assert result["test"].values[0, 0] == expected_result

    result = rolling_reduce(
        xr.Dataset({"test": dataarray}),
        how_reduce=how,
        x=2,
        center=True,
        chunk=False,
        how_dropna="any",
        how_label=how,
    )
    assert isinstance(result, xr.Dataset)
    assert result[f"test_{how}"].shape == (4, 5)
    assert result[f"test_{how}"].values[0, 0] == expected_result


def create_time_dataarray():
    time = pd.date_range("2024-01-01", periods=4, freq="D")
    return xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=("time",), coords={"time": time}, name="test")


def test_resample_dataarray_mean_2d():
    dataarray = create_time_dataarray()
    result = resample(dataarray, frequency="2D", how="mean")

    assert result.dims == ("time",)
    assert len(result.time) == 2
    assert np.allclose(result.values, [1.5, 3.5])


def test_resample_dataset_how_label():
    dataarray = create_time_dataarray()
    dataset = xr.Dataset({"test": dataarray})
    result = resample(dataset, frequency="2D", how="sum", how_label="sum")

    assert "test_sum" in result
    assert np.allclose(result["test_sum"].values, [3.0, 7.0])


def test_resample_extra_reduce_dims():
    time = pd.date_range("2024-01-01", periods=4, freq="D")
    dataarray = xr.DataArray(
        [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]],
        dims=("time", "x"),
        coords={"time": time, "x": [0, 1]},
        name="test",
    )
    result = resample(dataarray, frequency="2D", how="mean")
    assert result.dims == ("time", "x")

    result = resample(dataarray, frequency="2D", how="mean", extra_reduce_dims=["x"])
    assert result.dims == ("time",)
    assert np.allclose(result.values, [2.0, 6.5])


def test_resample_frequency_mapping_month():
    time = pd.date_range("2024-01-01", periods=40, freq="D")
    dataarray = xr.DataArray(np.arange(40.0), dims=("time",), coords={"time": time}, name="test")

    result = resample(dataarray, frequency="month", how="mean")
    expected = dataarray.resample(time="MS").mean(dim="time")

    xr.testing.assert_allclose(result, expected)
