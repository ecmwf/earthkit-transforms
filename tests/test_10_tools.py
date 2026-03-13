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
    ensure_list,
    get_dim_key,
    get_how,
    get_how_xp,
    get_spatial_info,
    groupby_bins,
    groupby_kwargs_decorator,
    groupby_time,
    latitude_weights,
    nanaverage,
    normalize_dims,
    season_order_decorator,
    standard_weights,
    time_dim_decorator,
    timedelta_to_largest_unit,
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
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator()(gb_dummy_func)()

    # Check if groupby_kwargs is None and other kwargs are empty
    assert result_groupby_kwargs == {}
    assert result_kwargs == {}


# Test case for the decorator when groupby_kwargs is provided
def test_groupby_kwargs_decorator_provided():
    # Prepare groupby_kwargs and other kwargs
    groupby_kwargs = {"frequency": "day", "bin_widths": 1}
    other_kwargs = {"method": "linear", "fill_value": 0}

    # Call the decorated function with groupby_kwargs provided
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator()(gb_dummy_func)(
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
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator()(gb_dummy_func)(
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
    result_groupby_kwargs, result_kwargs = groupby_kwargs_decorator()(gb_dummy_func)(
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


# ---------------------------------------------------------------------------
# normalize_dims
# ---------------------------------------------------------------------------


def test_normalize_dims_none():
    assert normalize_dims(None) == []


def test_normalize_dims_string():
    assert normalize_dims("time") == ["time"]


def test_normalize_dims_list():
    assert normalize_dims(["lat", "lon"]) == ["lat", "lon"]


def test_normalize_dims_tuple():
    assert normalize_dims(("lat", "lon")) == ["lat", "lon"]


# ---------------------------------------------------------------------------
# ensure_list
# ---------------------------------------------------------------------------


def test_ensure_list_already_list():
    assert ensure_list([1, 2, 3]) == [1, 2, 3]


def test_ensure_list_scalar():
    assert ensure_list("foo") == ["foo"]


def test_ensure_list_to_list_method():
    # pandas Index has a .to_list() method
    idx = pd.Index(["a", "b", "c"])
    assert ensure_list(idx) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# timedelta_to_largest_unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "td, expected",
    [
        (pd.Timedelta("1 days"), "days"),
        (pd.Timedelta("2 days"), "(2 days)"),
        (pd.Timedelta("1 hour"), "hours"),
        (pd.Timedelta("3 hours"), "(3 hours)"),
        (pd.Timedelta("1 minute"), "minutes"),
        (pd.Timedelta("90 seconds"), "(1.5 minutes)"),
        (pd.Timedelta("1 second"), "seconds"),
    ],
)
def test_timedelta_to_largest_unit(td, expected):
    assert timedelta_to_largest_unit(td) == expected


# ---------------------------------------------------------------------------
# season_order_decorator
# ---------------------------------------------------------------------------


def _make_seasonal_da():
    """Create a DataArray with a 'season' dimension in wrong order."""
    return xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=["season"],
        coords={"season": ["SON", "JJA", "MAM", "DJF"]},
    )


def _dummy_season_func(dataarray, *args, **kwargs):
    return dataarray


def test_season_order_decorator_passthrough():
    """Without frequency='season', result is returned unchanged."""
    da = _make_seasonal_da()
    decorated = season_order_decorator(_dummy_season_func)
    result = decorated(da)
    # Season order should be unchanged because frequency != 'season'
    assert list(result.season.values) == ["SON", "JJA", "MAM", "DJF"]


def test_season_order_decorator_season_reindexed():
    """With frequency='season', result must be reindexed to canonical DJF/MAM/JJA/SON order."""
    da = _make_seasonal_da()
    decorated = season_order_decorator(_dummy_season_func)
    result = decorated(da, frequency="season")
    assert list(result.season.values) == ["DJF", "MAM", "JJA", "SON"]


# ---------------------------------------------------------------------------
# nanaverage with weights and axis
# ---------------------------------------------------------------------------


def test_nanaverage_with_weights_no_nan():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    result = nanaverage(data, weights=weights)
    assert np.isclose(result, np.mean(data))


def test_nanaverage_with_weights_and_nan():
    data = np.array([1.0, np.nan, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    result = nanaverage(data, weights=weights)
    assert np.isclose(result, np.nanmean(data))


def test_nanaverage_with_weights_and_axis():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, 3.0])  # weight axis-1
    result = nanaverage(data, weights=weights, axis=1)
    expected = np.average(data, weights=weights, axis=1)
    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# standard_weights error path
# ---------------------------------------------------------------------------


def test_standard_weights_unrecognised():
    da = xr.DataArray(
        np.ones((3, 4)),
        dims=["latitude", "longitude"],
        coords={"latitude": [-30, 0, 30], "longitude": [0, 90, 180, 270]},
    )
    with pytest.raises(NotImplementedError):
        standard_weights(da, weights="nonsense")


# ---------------------------------------------------------------------------
# groupby_time
# ---------------------------------------------------------------------------


def _make_monthly_da():
    time = pd.date_range("2020-01-01", periods=24, freq="MS")
    return xr.DataArray(np.arange(24.0), dims=["time"], coords={"time": time})


def test_groupby_time_by_month():
    da = _make_monthly_da()
    grouped = groupby_time(da, frequency="month")
    result = grouped.mean()
    assert "month" in result.dims
    assert len(result) == 12


def test_groupby_time_infer_freq():
    """groupby_time should infer the frequency from the data when not provided."""
    da = _make_monthly_da()
    # monthly data infers "month"
    grouped = groupby_time(da)
    result = grouped.mean()
    # Result will have a time-derived grouping dimension
    assert result.ndim == 1


def test_groupby_time_invalid_frequency():
    da = _make_monthly_da()
    with pytest.raises((ValueError, AttributeError)):
        groupby_time(da, frequency="nonsense_freq")


# ---------------------------------------------------------------------------
# groupby_bins
# ---------------------------------------------------------------------------


def test_groupby_bins_int_bin_widths():
    time = pd.date_range("2020-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12.0), dims=["time"], coords={"time": time})
    grouped = groupby_bins(da, frequency="month", bin_widths=3)
    result = grouped.mean()
    assert result.ndim == 1
    # 12 months / 3 per bin = 4 bins
    assert len(result) == 4


def test_groupby_bins_list_bin_widths():
    time = pd.date_range("2020-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12.0), dims=["time"], coords={"time": time})
    grouped = groupby_bins(da, frequency="month", bin_widths=[0, 6, 12])
    result = grouped.mean()
    assert result.ndim == 1
    assert len(result) == 2


# ---------------------------------------------------------------------------
# groupby_kwargs_decorator climatology validation
# ---------------------------------------------------------------------------


def test_groupby_kwargs_decorator_climatology_invalid_freq():
    """frequency='day' must raise ValueError when climatology=True."""

    @groupby_kwargs_decorator(climatology=True)
    def dummy(data, groupby_kwargs=None):
        return groupby_kwargs

    with pytest.raises(ValueError, match="frequency='day' is not accepted for climatology"):
        dummy(None, frequency="day")


def test_groupby_kwargs_decorator_climatology_valid_freq():
    """frequency='month' must NOT raise when climatology=True."""

    @groupby_kwargs_decorator(climatology=True)
    def dummy(data, groupby_kwargs=None):
        return groupby_kwargs

    result = dummy(None, frequency="month")
    assert result["frequency"] == "month"
