import pytest
import xarray as xr
import pandas as pd
from earthkit.aggregate.tools import time_dim_decorator, groupby_kwargs_decorator, GROUPBY_KWARGS

# Define a dummy function to decorate
def dummy_func(dataarray, *args, time_dim=None, **kwargs):
    return dataarray

# Test case for the decorator when time_dim is None
def test_time_dim_decorator_time_dim_none():
    # Prepare test data
    dataarray = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)})

    # Call the decorated function
    result = time_dim_decorator(dummy_func)(dataarray)

    # Check if the time dimension is correctly detected
    assert result.dims == ("time",)

# Test case for the decorator when time_dim is provided
def test_time_dim_decorator_time_dim_provided():
    # Prepare test data
    dataarray = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)})

    # Call the decorated function with time_dim provided
    result = time_dim_decorator(dummy_func)(dataarray, time_dim="time")

    # Check if the time dimension remains unchanged
    assert result.dims == ("time",)

# Test case for the decorator when time_shift is provided
def test_time_dim_decorator_time_shift_provided():
    # Prepare test data
    dataarray = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)})

    # Call the decorated function with time_shift provided
    result = time_dim_decorator(dummy_func)(dataarray, time_shift={"days": 1})

    # Check if the time dimension is correctly shifted
    expected_coords = pd.date_range("2000-01-02", periods=3)
    assert all(result.coords["time"].values == expected_coords)

# Test case for the decorator when both time_dim and time_shift are provided
def test_time_dim_decorator_time_dim_and_time_shift_provided():
    # Prepare test data
    dataarray = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": pd.date_range("2000-01-01", periods=3)})

    # Call the decorated function with both time_dim and time_shift provided
    result = time_dim_decorator(dummy_func)(dataarray, time_dim="time", time_shift={"days": 1})

    # Check if the time dimension remains unchanged
    assert result.dims == ("time",)

    # Check if the time dimension is correctly shifted
    expected_coords = pd.date_range("2000-01-02", periods=3)
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
    groupby_kwargs = {"frequency": "day", "bin_widths": 1, "squeeze": True}
    other_kwargs = {"method": "linear", "fill_value": 0}
    
    # Call the decorated function with groupby_kwargs provided
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)(groupby_kwargs=groupby_kwargs, **other_kwargs)
    
    # Check if groupby_kwargs and other kwargs are correctly merged
    expected_kwargs = {"method": "linear", "fill_value": 0}
    assert result_groupby_kwargs == groupby_kwargs
    assert result_kwargs == expected_kwargs

# Test case for the decorator when some groupby_kwargs are provided as keyword arguments
def test_groupby_kwargs_decorator_partial_provided():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1, "squeeze": True}
    other_kwargs = {"method": "linear"}
    
    # Call the decorated function with some groupby_kwargs provided as keyword arguments
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator(gb_dummy_func)(frequency="day", bin_widths=1, squeeze=True, **other_kwargs)
    
    # Check if groupby_kwargs and other kwargs are correctly merged
    expected_kwargs = {"method": "linear"}
    assert result_groupby_kwargs == groupby_kwargs
    assert result_kwargs == expected_kwargs

# Test case for the decorator when a non-groupby_kwargs key is provided
def test_groupby_kwargs_decorator_non_groupby_key():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1, "squeeze": True}
    other_kwargs = {"method": "linear"}
    
    # Call the decorated function with a non-groupby_kwargs key provided
    with pytest.raises(TypeError):
        groupby_kwargs_decorator(gb_dummy_func)(invalid_key="value", **groupby_kwargs, **other_kwargs)
