from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
import pandas as pd
import pytest
import xarray as xr
from earthkit.transforms._tools import (
    get_dim_key,
    get_how,
    get_how_xp,
    get_spatial_info,
    groupby_kwargs_decorator,
    latitude_weights,
    nanaverage,
    time_dim_decorator,
)


# Define a dummy function to decorate
def dummy_func(dataarray, *args, time_dim=None, **kwargs):
    return dataarray, time_dim


# Define a dummy function to decorate
def dummy_func2(dataarray, *args, time_dim=None, **kwargs):
    return dataarray


# Test case for the decorator when time_shift is provided
def test_time_dim_decorator_time_shift_provided():
    # Prepare test data
    dataarray = xr.DataArray(
        [1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)}
    )

    # Call the decorated function with time_shift provided
    result, result_time_dim = time_dim_decorator(dummy_func)(dataarray, time_shift={"days": 1})

    assert result_time_dim == "time"

    # Check if the time dimension is correctly shifted
    expected_coords = pd.date_range("2000-01-02", periods=3)
    assert all(result.coords["time"].values == expected_coords)


# Test case for the decorator when both time_dim and time_shift are provided
def test_time_dim_decorator_time_dim_and_time_shift_provided():
    # Prepare test data
    dataarray = xr.DataArray(
        [1, 2, 3], dims=["dummy"], coords={"dummy": pd.date_range("2000-01-01", periods=3)}
    )

    # Call the decorated function with both time_dim and time_shift provided
    result, result_time_dim = time_dim_decorator(dummy_func)(
        dataarray, time_dim="dummy", time_shift={"days": 1}
    )

    # Check if the time dimension remains unchanged
    assert result_time_dim == "dummy"

    # Check if the time dimension is correctly shifted
    expected_coords = pd.date_range("2000-01-02", periods=3)
    assert all(result.coords["dummy"].values == expected_coords)


# Test case for the decorator when both time_dim and time_shift are provided
def test_time_dim_decorator_not_found_error():
    dataarray = xr.DataArray([1, 2, 3], dims=["dummy"], coords={"dummy": [1, 2, 3]})

    # Check error raised when cannot find time dimension
    with pytest.raises(KeyError):
        time_dim_decorator(dummy_func)(dataarray, time_shift={"days": 1})


# Test case for the decorator when time_shift is provided and remove_partial_periods=True
def test_time_dim_decorator_time_shift_provided_trim_shifted():
    # Prepare test data
    dataarray = xr.DataArray(
        [1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)}
    )

    # Call the decorated function with time_shift provided
    result = time_dim_decorator(dummy_func2)(dataarray, time_shift={"days": 1}, remove_partial_periods=True)
    # Check if the time dimension is correctly shifted, and the start and end dates are removed
    expected_coords = pd.date_range("2000-01-03", periods=1)
    assert all(result.coords["time"].values == expected_coords)


# Define a dummy function to decorate
def gb_dummy_func(*args, groupby_kwargs=None, **kwargs):
    return groupby_kwargs, kwargs


# Test case for the decorator when groupby_kwargs is None
def test_groupby_kwargs_decorator_none():
    # Call the decorated function with no groupby_kwargs
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)()

    # Check if groupby_kwargs is None and other kwargs are empty
    assert result_groupby_kwargs == {}
    assert result_kwargs == {}


# Test case for the decorator when groupby_kwargs is provided
def test_groupby_kwargs_decorator_provided():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1}
    other_kwargs = {"method": "linear", "fill_value": 0}

    # Call the decorated function with groupby_kwargs provided
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)(
        **groupby_kwargs, **other_kwargs
    )

    assert result_groupby_kwargs == groupby_kwargs
    assert result_kwargs == other_kwargs


# Test case for the decorator when some groupby_kwargs are provided as keyword arguments
def test_groupby_kwargs_decorator_partial_provided():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1}
    other_kwargs = {"method": "linear"}

    # Call the decorated function with some groupby_kwargs provided as keyword arguments
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)(
        **groupby_kwargs, **other_kwargs
    )

    assert result_groupby_kwargs == groupby_kwargs
    assert result_kwargs == other_kwargs


# Test case for the decorator when groupby_kwargs is provided
def test_groupby_kwargs_decorator_override():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1}
    other_kwargs = {"method": "linear", "fill_value": 0}

    override_groupby_kwargs = {"frequency": "hour"}

    # Call the decorated function with groupby_kwargs provided
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)(
        groupby_kwargs=override_groupby_kwargs, **groupby_kwargs, **other_kwargs
    )

    groupby_kwargs.update(override_groupby_kwargs)
    assert result_groupby_kwargs == groupby_kwargs
    assert result_kwargs == other_kwargs


@pytest.mark.parametrize(
    "how_str, how_expected",
    (
        ["mean", np.nanmean],
        ["numpy.mean", np.mean],
        ["numpy.nanmean", np.nanmean],
        ["nanaverage", nanaverage],
        ["average", nanaverage],
        ["numpy.average", np.average],
        ["stddev", np.nanstd],
        ["std", np.nanstd],
        ["stdev", np.nanstd],
        ["sum", np.nansum],
        ["max", np.nanmax],
        ["min", np.nanmin],
        ["median", np.nanmedian],
        ["q", np.nanquantile],
        ["quantile", np.nanquantile],
        ["percentile", np.nanpercentile],
        ["p", np.nanpercentile],
    ),
)
def test_get_how(how_str: str, how_expected: Any):
    assert how_expected == get_how(how_str)


def test_get_how_xp_np():
    test_case = (
        ["mean", np.nanmean],
        ["numpy.mean", np.mean],
        ["numpy.nanmean", np.nanmean],
        ["nanaverage", nanaverage],
        ["average", nanaverage],
        ["numpy.average", np.average],
        ["stddev", np.nanstd],
        ["std", np.nanstd],
        ["stdev", np.nanstd],
        ["sum", np.nansum],
        ["max", np.nanmax],
        ["min", np.nanmin],
        ["median", np.nanmedian],
        ["q", np.nanquantile],
        ["quantile", np.nanquantile],
        ["percentile", np.nanpercentile],
        ["p", np.nanpercentile],
    )
    for how_str, expected in test_case:
        assert expected == get_how_xp(how_str, xp=np)


@pytest.mark.parametrize(
    "data_object",
    (
        xr.DataArray(np.random.rand(10, 10), dims=["x", "y"]),
        xr.Dataset(
            {"var": (["x", "y"], np.random.rand(10, 10))}, coords={"x": np.arange(10), "y": np.arange(10)}
        ),
        np.random.rand(10, 10),
    ),
)
def test_get_how_xp_np_objects(data_object):
    test_cases = (
        ["mean", np.nanmean],
        ["numpy.mean", np.mean],
        ["numpy.nanmean", np.nanmean],
        ["nanaverage", nanaverage],
        ["average", nanaverage],
        ["numpy.average", np.average],
        ["stddev", np.nanstd],
        ["std", np.nanstd],
        ["stdev", np.nanstd],
        ["sum", np.nansum],
        ["max", np.nanmax],
        ["min", np.nanmin],
        ["median", np.nanmedian],
        ["q", np.nanquantile],
        ["quantile", np.nanquantile],
        ["percentile", np.nanpercentile],
        ["p", np.nanpercentile],
    )
    for how_str, expected in test_cases:
        assert expected == get_how_xp(how_str, data_object=data_object)


@pytest.mark.skipif(cp is None, reason="Cupy is not installed")
def test_get_how_xp_cp():
    test_cases = [
        ("mean", cp.nanmean),
        ("numpy.mean", cp.mean),
        ("numpy.nanmean", cp.nanmean),
        ("nanaverage", nanaverage),
        ("average", nanaverage),
        ("numpy.average", cp.average),
        ("stddev", cp.nanstd),
        ("std", cp.nanstd),
        ("stdev", cp.nanstd),
        ("sum", cp.nansum),
        ("max", cp.nanmax),
        ("min", cp.nanmin),
        ("median", cp.nanmedian),
        ("q", cp.nanquantile),
        ("quantile", cp.nanquantile),
        ("percentile", cp.nanpercentile),
        ("p", cp.nanpercentile),
    ]
    for how_str, expected in test_cases:
        assert get_how_xp(how_str, xp=cp) == expected


# Define a helper function to create dummy dataarray with given axis attribute
def create_dataarray_with_axis(axes, dims):
    dim_arrs = {}
    for axis, dim in zip(axes, dims):
        dim_arrs[dim] = xr.DataArray([1, 2], dims=dim)
        if axis is not None:
            dim_arrs[dim] = dim_arrs[dim].assign_attrs({"axis": axis})

    return xr.DataArray(np.ones([2] * len(dims)), dims=dims, coords=dim_arrs)


@pytest.mark.parametrize(
    "axis, dim_expected",
    (
        ["x", "lon"],
        ["y", "lat"],
        ["t", "time"],
        ["z", "z"],
        ["realization", "realization"],
    ),
)
# Test case for the function when axis matches an attribute in the dataset's dimensions
def test_get_dim_keys(axis, dim_expected):
    # Create a dataarray with an axis attribute
    dataarray = create_dataarray_with_axis(["x", "y", "t", None], ["lon", "lat", "time", "realization"])
    # Check if the correct dimension key is returned
    assert dim_expected == get_dim_key(dataarray, axis)
    # Create a dataarray with an axis attribute
    dataarray = create_dataarray_with_axis([None, None, None, None], ["lon", "lat", "time", "realization"])
    # Check if the correct dimension key is returned
    assert dim_expected == get_dim_key(dataarray, axis)


# Test case for the function when axis matches an attribute in the dataset's dimensions
def test_get_dim_keys_raises():
    axis = "z"
    # Create a dataarray with an axis attribute
    dataarray = create_dataarray_with_axis(["x", None], ["lon", "realization"])
    # Check if the correct dimension key is returned
    with pytest.raises(ValueError, match=f"Unable to find dimension key for axis '{axis}' in dataarray"):
        get_dim_key(dataarray, axis, raise_error=True)


# Test case for the function when axis matches an attribute in the dataset's dimensions
def test_get_spatial_info():
    # Create a dataarray with an axis attribute
    dataarray = create_dataarray_with_axis(["x", "y", "t"], ["lon", "lat", "time"])
    # Check if the correct dimension key is returned
    expected_result = {"lat_key": "lat", "lon_key": "lon", "regular": True, "spatial_dims": ["lat", "lon"]}
    assert expected_result == get_spatial_info(dataarray)


def test_latitude_weights():
    da = xr.DataArray(np.arange(5), dims=("y"), coords={"y": [-90, -60, 0, 60, 90]})

    weights = latitude_weights(da)
    assert np.allclose(weights, [0, 0.5, 1, 0.5, 0])

    weights = latitude_weights(da.rename({"y": "latitude"}))
    assert np.allclose(weights, [0, 0.5, 1, 0.5, 0])
