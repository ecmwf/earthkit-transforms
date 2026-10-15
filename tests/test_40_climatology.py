import tracemalloc

import dask
import dask.array
import numpy as np
import pandas as pd
import pytest

# from earthkit.data.core.temporary import temp_directory
import xarray as xr

from earthkit import data as ek_data
from earthkit.transforms import climatology
from earthkit.transforms._tools import earthkit_remote_test_data_file

# Use caching for speedy repeats
ek_data.settings.set("cache-policy", "user")


def get_data():
    remote_era5_file = earthkit_remote_test_data_file("ERA5-Reading-2m-temperature-1940-2025.nc")
    return ek_data.from_source("url", remote_era5_file)


@pytest.fixture(scope="session")
def era5_dataset():
    """Session-scoped computed xarray.Dataset for ERA5 test data."""
    return get_data().to_xarray().sel(valid_time=slice("2000", "2020")).compute()


@pytest.fixture(scope="session")
def era5_da_t2m(era5_dataset):
    """Session-scoped DataArray for the 't2m' variable from ERA5 dataset."""
    return era5_dataset["t2m"]


@pytest.fixture
def in_data(request):
    """Indirect fixture to select between dataset and dataarray variants."""
    return request.getfixturevalue(request.param)


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_climatology_base(in_data, expected_return_type, method):
    clim = method(in_data)
    assert isinstance(clim, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "t2m" == clim.name
    else:
        assert "t2m" in clim

    # Check alternate frequencies
    for freq in ["month", "dayofyear"]:
        clim = method(in_data, frequency=freq)
        assert freq in list(clim.dims)


@pytest.mark.parametrize(
    "freq, expected_dim",
    (
        ("D", "dayofyear"),
        ("dayofyear", "dayofyear"),
        # ("day", "day"),  # "day" is invalid for climatology calculations
        # ("5D", "dayofyear"),
        ("week", "week"),
        ("weekofyear", "weekofyear"),
        ("MS", "month"),
        ("ME", "month"),
        # ("3MS", "month"),
        ("month", "month"),
        ("YE", "year"),
        ("year", "year"),
    ),
)
@pytest.mark.parametrize(
    "method",
    (
        climatology.mean,
        climatology.median,
    ),
)
def test_climatology_frequency(era5_dataset, method, freq, expected_dim):
    clim = method(era5_dataset, frequency=freq)
    assert "t2m" in clim
    assert expected_dim in list(clim.dims)


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_anomaly_base(in_data, expected_return_type, clim_method):
    clim_m = clim_method(in_data)
    anom_m = climatology.anomaly(in_data, clim_m)

    assert isinstance(anom_m, expected_return_type)
    # Dimensions and shape of the anomaly should be the same as the input data
    assert all(dim in list(anom_m.dims) for dim in in_data.dims)
    assert all(anom_m.sizes[dim] == in_data.sizes[dim] for dim in in_data.dims)
    if expected_return_type == xr.DataArray:
        assert "t2m" == anom_m.name
    else:
        assert "t2m" in anom_m


@pytest.mark.parametrize(
    "freq, expected_dim_length",
    (
        ("D", 7671),
        ("dayofyear", 7671),
        # ("day", "Ambiguous, not supported"),
        ("week", 1097),
        ("weekofyear", 1097),
        ("MS", 252),
        ("ME", 252),
        # ("3MS", "month"),
        ("month", 252),
        ("YE", 21),
        ("year", 21),
    ),
)
@pytest.mark.parametrize(
    "clim_method",
    (
        climatology.mean,
        climatology.max,
    ),
)
def test_anomaly_frequency(era5_dataset, clim_method, expected_dim_length, freq):
    clim_m = clim_method(era5_dataset, frequency=freq)
    anom_m = climatology.anomaly(era5_dataset, clim_m, frequency=freq)
    # Dimensions of the anomaly should be the same as the input data
    assert all(dim in list(anom_m.dims) for dim in era5_dataset.dims)
    # Check valid_time is the correct length for the frequency used
    assert anom_m.sizes["valid_time"] == expected_dim_length


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_climatology_monthly(in_data, expected_return_type, method, how):
    clim_m = method(in_data)
    assert isinstance(clim_m, expected_return_type)
    assert "month" in list(clim_m.dims)
    if expected_return_type == xr.DataArray:
        assert "t2m" == clim_m.name
    else:
        assert "t2m" in clim_m

    clim_m = method(in_data, how_label=how)
    assert isinstance(clim_m, expected_return_type)
    assert "month" in list(clim_m.dims)
    if expected_return_type == xr.DataArray:
        assert f"t2m_{how}" == clim_m.name
    else:
        assert f"t2m_{how}" in clim_m


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_climatology_daily(in_data, expected_return_type, method, how):
    clim_d = method(in_data)
    assert isinstance(clim_d, expected_return_type)
    assert "dayofyear" in list(clim_d.dims)
    if expected_return_type == xr.DataArray:
        assert "t2m" == clim_d.name
    else:
        assert "t2m" in clim_d

    clim_d = method(in_data, how_label=how)
    assert isinstance(clim_d, expected_return_type)
    assert "dayofyear" in list(clim_d.dims)
    if expected_return_type == xr.DataArray:
        assert f"t2m_{how}" == clim_d.name
    else:
        assert f"t2m_{how}" in clim_d


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_anomaly_monthly(in_data, expected_return_type, clim_method):
    clim_m = clim_method(in_data)
    anom_m = climatology.anomaly(in_data, clim_m, frequency="month")
    assert isinstance(anom_m, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "t2m" == anom_m.name
    else:
        assert "t2m" in anom_m
    anom_m = climatology.anomaly(in_data, clim_m, frequency="month", how_label="anomaly")
    assert isinstance(anom_m, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "t2m_anomaly" == anom_m.name
    else:
        assert "t2m_anomaly" in anom_m


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
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_anomaly_daily(in_data, expected_return_type, clim_method):
    clim_d = clim_method(in_data)
    anom_d = climatology.anomaly(in_data, clim_d, frequency="dayofyear")
    assert isinstance(anom_d, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "t2m" == anom_d.name
    else:
        assert "t2m" in anom_d
    anom_d = climatology.anomaly(in_data, clim_d, frequency="dayofyear", how_label="anomaly")
    assert isinstance(anom_d, expected_return_type)
    if expected_return_type == xr.DataArray:
        assert "t2m_anomaly" == anom_d.name
    else:
        assert "t2m_anomaly" in anom_d


@pytest.mark.parametrize(
    "frequency",
    ("dayofyear", "month"),
)
@pytest.mark.parametrize(
    "in_data, expected_return_type",
    (
        pytest.param("era5_dataset", xr.Dataset, id="dataset"),
        pytest.param("era5_da_t2m", xr.DataArray, id="dataarray"),
    ),
    indirect=["in_data"],
)
def test_quantile_categories(in_data, expected_return_type, frequency):
    clim_q = climatology.quantiles(in_data, [0.1, 0.5, 0.9], frequency=frequency)
    categories = climatology.quantile_categories(in_data, clim_q)
    assert isinstance(categories, expected_return_type)
    # Dimensions and shape of the categories should be the same as the input data
    assert all(dim in list(categories.dims) for dim in in_data.dims)
    assert all(categories.sizes[dim] == in_data.sizes[dim] for dim in in_data.dims)
    if expected_return_type == xr.DataArray:
        assert "t2m" == categories.name
        # Three quantiles give four categories, all of which occur
        np.testing.assert_array_equal(np.unique(categories.values), [0, 1, 2, 3])
    else:
        assert "t2m" in categories


# ---------------------------------------------------------------------------
# Local (synthetic-data) tests — no network required
# ---------------------------------------------------------------------------


def _make_monthly_clim_da(n_years=3, start_year=2000):
    """n_years of monthly data, values = month index (0-based)."""
    time = pd.date_range(f"{start_year}-01-01", periods=12 * n_years, freq="MS")
    # Values cycle 0..11 over months so monthly mean is simply 0..11
    data = np.tile(np.arange(12.0), n_years) + 1.0  # 1..12 repeated
    return xr.DataArray(data, dims=["time"], coords={"time": time}, name="var")


def _make_daily_clim_da(n_years=4, start_year=2001):
    """n_years of daily data (non-leap years only for simplicity: 365*n_years days)."""
    time = pd.date_range(f"{start_year}-01-01", periods=365 * n_years, freq="D")
    rng = np.random.default_rng(42)
    data = rng.random(len(time))
    return xr.DataArray(data, dims=["time"], coords={"time": time}, name="var")


# --- climatology.reduce (local) --------------------------------------------


def test_climatology_reduce_local_mean():
    da = _make_monthly_clim_da()
    result = climatology.reduce(da, how="mean")
    assert isinstance(result, xr.DataArray)


def test_climatology_reduce_local_monthly_frequency():
    da = _make_monthly_clim_da()
    result = climatology.reduce(da, how="mean", frequency="month")
    assert isinstance(result, xr.DataArray)
    assert "month" in result.dims
    assert len(result) == 12


def test_climatology_reduce_local_climatology_range():
    """climatology_range should restrict the data used."""
    da = _make_monthly_clim_da(n_years=5, start_year=2000)
    result_all = climatology.mean(da, frequency="month")
    result_range = climatology.mean(da, frequency="month", climatology_range=("2000", "2001"))
    # Both have 12-month output but values may differ if data varies across years
    assert "month" in result_range.dims
    assert len(result_range) == 12
    # With uniform data across years the values are the same
    np.testing.assert_allclose(result_all.values, result_range.values)


# --- climatology.min regression (bug fix) ----------------------------------


def test_climatology_min_le_mean_local():
    """Regression: climatology.min must be <= climatology.mean for all months."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=36, freq="MS")
    data = xr.DataArray(rng.random(36) + 1.0, dims=["time"], coords={"time": time}, name="var")
    clim_min = climatology.min(data, frequency="month")
    clim_mean = climatology.mean(data, frequency="month")
    assert (clim_min.values <= clim_mean.values + 1e-10).all()


# --- climatology.monthly_* (local) -----------------------------------------


@pytest.mark.parametrize(
    "method",
    (climatology.monthly_mean, climatology.monthly_min, climatology.monthly_max, climatology.monthly_std),
)
def test_climatology_monthly_local(method):
    da = _make_monthly_clim_da()
    result = method(da)
    assert isinstance(result, xr.DataArray)
    assert "month" in result.dims
    assert len(result) == 12


def test_climatology_monthly_mean_values_local():
    """With data that repeats the same value per month, the monthly mean == that value."""
    da = _make_monthly_clim_da()
    result = climatology.monthly_mean(da)
    # Each month value is constant across years (1..12), so clim mean == the month value
    np.testing.assert_allclose(result.values, np.arange(1.0, 13.0), rtol=1e-10)


# --- climatology.daily_* (local) -------------------------------------------


def test_climatology_daily_local():
    da = _make_daily_clim_da()
    result = climatology.daily_mean(da)
    assert isinstance(result, xr.DataArray)
    assert "dayofyear" in result.dims
    assert len(result) == 365


# --- climatology.quantiles / percentiles (local) ---------------------------


def test_climatology_quantiles_local():
    da = _make_monthly_clim_da()
    result = climatology.quantiles(da, q=[0.25, 0.5, 0.75])
    assert isinstance(result, xr.DataArray)
    assert "quantile" in result.dims
    assert len(result.coords["quantile"]) == 3


def test_climatology_percentiles_local():
    da = _make_monthly_clim_da()
    result = climatology.percentiles(da, p=[25, 50, 75])
    assert isinstance(result, xr.DataArray)
    assert "percentile" in result.dims
    assert len(result.coords["percentile"]) == 3


# --- climatology.anomaly (local) -------------------------------------------


def test_climatology_anomaly_local():
    da = _make_monthly_clim_da()
    clim = climatology.monthly_mean(da)
    anom = climatology.anomaly(da, clim, frequency="month")
    assert isinstance(anom, xr.DataArray)
    # Anomaly dims must include the original time dimension
    assert "time" in anom.dims
    assert anom.shape == da.shape


def test_climatology_anomaly_local_near_zero_mean():
    """With a constant-by-month dataset, monthly anomaly should be ~0."""
    da = _make_monthly_clim_da()
    clim = climatology.monthly_mean(da)
    anom = climatology.anomaly(da, clim, frequency="month")
    np.testing.assert_allclose(anom.values, 0.0, atol=1e-10)


# --- climatology.auto_anomaly (local) --------------------------------------


def test_climatology_auto_anomaly_local():
    da = _make_monthly_clim_da()
    result = climatology.auto_anomaly(da, frequency="month")
    assert isinstance(result, xr.DataArray)
    assert "time" in result.dims


# --- anomaly: clim frequency ≠ anomaly frequency (local) -------------------


def test_anomaly_monthly_clim_no_freq_broadcasts_to_original_shape():
    """Monthly climatology with no anomaly frequency broadcasts back to the original time axis.

    Code path: clim_freq='month', frequency=None →
    groupby(month) - clim → broadcast_like(da)
    """
    da = _make_daily_clim_da()
    clim = climatology.monthly_mean(da)
    anom = climatology.anomaly(da, clim)
    assert isinstance(anom, xr.DataArray)
    assert anom.shape == da.shape
    assert "time" in anom.dims


def test_anomaly_scalar_clim_no_freq_subtracts_scalar():
    """Overall (scalar) climatology with no anomaly frequency subtracts the scalar from the input.

    Code path: clim_freq='year', frequency=None → da - clim_scalar
    """
    da = _make_daily_clim_da()
    clim = climatology.mean(da)  # no frequency → scalar
    anom = climatology.anomaly(da, clim)
    assert isinstance(anom, xr.DataArray)
    assert anom.shape == da.shape
    assert "time" in anom.dims
    # Anomaly mean must be near zero (mean of da minus mean of da)
    np.testing.assert_allclose(float(anom.mean()), 0.0, atol=1e-10)


def test_anomaly_relative_near_zero_for_constant_per_month():
    """Relative anomaly is ~0 when data equals the climatology exactly.

    With constant-per-month data, (data - clim) / clim * 100 == 0.
    """
    da = _make_monthly_clim_da()
    clim = climatology.monthly_mean(da)
    anom_rel = climatology.anomaly(da, clim, frequency="month", relative=True)
    assert isinstance(anom_rel, xr.DataArray)
    assert anom_rel.shape == da.shape
    np.testing.assert_allclose(anom_rel.values, 0.0, atol=1e-10)


# --- anomaly: lazily loaded data and climatologies (local) ----------------


@pytest.mark.parametrize("relative", (False, True))
@pytest.mark.parametrize("frequency", (None, "month"))
def test_anomaly_lazy_climatology(relative, frequency):
    """A lazily loaded climatology, chunked by group as climatology.quantiles returns it, gives the eager anomaly."""
    da = _make_daily_clim_da().chunk(time=365)
    clim = climatology.daily_mean(da).compute()
    result = climatology.anomaly(da, clim.chunk(dayofyear=1), relative=relative, frequency=frequency)
    if frequency is None:
        assert result.chunksizes["time"] == da.chunksizes["time"]
    xr.testing.assert_allclose(
        result.compute(),
        climatology.anomaly(da.compute(), clim, relative=relative, frequency=frequency),
        rtol=1e-12,
    )


def test_anomaly_lazy_data_eager_climatology():
    """Lazily loaded data gives a lazy anomaly, without holding a climatology the size of the data in memory."""
    da = (_make_daily_clim_da() + xr.DataArray(np.zeros(500), dims=["x"])).chunk(time=365)
    clim = climatology.monthly_mean(da).compute()
    tracemalloc.start()
    try:
        result = climatology.anomaly(da, clim)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.chunksizes["time"] == da.chunksizes["time"]
    assert peak < da.nbytes
    xr.testing.assert_identical(result.compute(), climatology.anomaly(da.compute(), clim))


def test_anomaly_lazy_climatology_of_part_of_the_grid():
    """Points the climatology does not cover have missing anomalies, as for eager data."""
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    da = xr.DataArray(np.ones((24, 6)), dims=["time", "x"], coords={"time": time, "x": np.arange(6)}, name="var").chunk(
        x=3
    )
    clim = xr.DataArray(
        np.zeros((12, 3)), dims=["month", "x"], coords={"month": np.arange(1, 13), "x": [0, 2, 4]}, name="var"
    )
    result = climatology.anomaly(da, clim.chunk(month=1))
    expected = climatology.anomaly(da.compute(), clim)
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(expected.isel(time=0).values, [1, np.nan, 1, np.nan, 1, np.nan])


def test_anomaly_lazy_data_with_a_group_dimension():
    """A dimension of the data named like the climatology's groups does not change the groups' chunks."""
    time = pd.date_range("2000-01-01", periods=3, freq="MS")
    da = xr.DataArray(np.ones((3, 2)), dims=["time", "month"], coords={"time": time}, name="var")
    clim = xr.DataArray(np.arange(3.0), dims=["month"], coords={"month": [1, 2, 3]}, name="var")
    result = climatology.anomaly(da.chunk(month=1), clim)
    xr.testing.assert_identical(result.compute(), climatology.anomaly(da, clim))


@pytest.mark.parametrize("lazy", (False, True))
def test_anomaly_time_coordinate_not_a_dimension(lazy):
    """Time steps along another dimension are compared with the climatology of their time coordinate."""
    time = pd.date_range("2000-01-31", periods=3, freq="D")  # 31 January, 1 and 2 February
    da = xr.DataArray([1.0, 1.0, 1.0], dims=["obs"], coords={"valid_time": ("obs", time)}, name="var")
    clim = xr.DataArray([1.0, -1.0], dims=["month"], coords={"month": [1, 2]}, name="var")
    if lazy:
        clim = clim.chunk(month=1)
    result = climatology.anomaly(da, clim, time_dim="valid_time")
    np.testing.assert_array_equal(result.values, [0, 2, 2])


def test_anomaly_missing_times():
    """Time steps at missing times have missing anomalies, even if the climatology has a missing group."""
    time = pd.DatetimeIndex(["2000-01-01", "NaT", "2000-02-01"])
    da = xr.DataArray([10.0, 20.0, 30.0], dims=["time"], coords={"time": time}, name="var")
    clim = xr.DataArray([1.0, 2.0, 5.0], dims=["month"], coords={"month": [1.0, 2.0, np.nan]}, name="var")
    np.testing.assert_array_equal(climatology.anomaly(da, clim).values, [9, np.nan, 28])


@pytest.mark.parametrize("method", ("anomaly", "quantile_categories"))
def test_eager_data_larger_than_a_chunk_lazy_climatology(method):
    """Eager data larger than the chunks chosen for it gets the same chunks for its values and coordinates."""
    time = pd.date_range("2000-01-01", periods=2000, freq="D")
    da = xr.DataArray(np.zeros((2000, 1100)), dims=["time", "x"], coords={"time": time}, name="var")
    clim = xr.DataArray(np.ones((12, 1100)), dims=["month", "x"], coords={"month": np.arange(1, 13)}, name="var")
    if method == "quantile_categories":
        clim = clim.expand_dims(quantile=[0.5])
    result = getattr(climatology, method)(da, clim.chunk(month=1))
    assert len(result.chunksizes["time"]) > 1
    xr.testing.assert_identical(result.compute(), getattr(climatology, method)(da, clim))


@pytest.mark.parametrize(
    "method, empty",
    (("anomaly", "x"), ("quantile_categories", "x"), ("quantile_categories", "quantile")),
)
def test_empty_dimension_lazy_climatology(method, empty):
    """Eager data with an empty dimension is compared with a lazily loaded climatology as with an eager one."""
    n_x = 0 if empty == "x" else 3
    time = pd.date_range("2000-01-01", periods=4, freq="MS")
    da = xr.DataArray(np.ones((4, n_x)), dims=["time", "x"], coords={"time": time}, name="var")
    clim = xr.DataArray(np.zeros((12, n_x)), dims=["month", "x"], coords={"month": np.arange(1, 13)}, name="var")
    if method == "quantile_categories":
        clim = clim.expand_dims(quantile=[] if empty == "quantile" else [0.5])
    result = getattr(climatology, method)(da, clim.chunk(month=1))
    xr.testing.assert_identical(result.compute(), getattr(climatology, method)(da, clim))


def test_anomaly_leaves_dask_configuration_unchanged(monkeypatch):
    """Anomalies of lazily loaded data do not set the dask configuration, which all threads share."""

    def set_configuration(*_args, **_kwargs):
        raise AssertionError("the dask configuration was set")

    da = (_make_monthly_clim_da() + xr.DataArray(np.zeros(100), dims=["x"])).chunk(time=12)
    clim = climatology.monthly_mean(da).compute()
    monkeypatch.setattr(dask.config, "set", set_configuration)
    for data, clim_month in ((da, clim), (da.compute(), clim.chunk(month=1)), (da, clim.chunk(month=1))):
        climatology.anomaly(data, clim_month)


def test_anomaly_missing_groups():
    """Time steps whose group is missing from the climatology have missing anomalies."""
    time = pd.date_range("2000-01-30", periods=4, freq="D")  # 30 and 31 January, 1 and 2 February
    da = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["time"], coords={"time": time}, name="var")
    clim = xr.DataArray([1.0], dims=["month"], coords={"month": [1]}, name="var")
    np.testing.assert_array_equal(climatology.anomaly(da, clim).values, [0, 1, np.nan, np.nan])


def test_anomaly_scalar_coordinates():
    """Scalar coordinates stay scalar rather than being repeated along the time dimension."""
    da = _make_monthly_clim_da().assign_coords(latitude=51.5)
    result = climatology.anomaly(da, climatology.monthly_mean(da))
    assert result["latitude"].dims == ()


def test_anomaly_bin_widths_with_grouped_climatology_raises():
    da = _make_monthly_clim_da()
    with pytest.raises(ValueError, match="bin_widths"):
        climatology.anomaly(da, climatology.monthly_mean(da), bin_widths=2)


# --- auto_anomaly: varied climatology_frequency / frequency (local) --------


def test_auto_anomaly_monthly_clim_freq_broadcasts_to_original_shape():
    """auto_anomaly with climatology_frequency='month' and no anomaly frequency.

    The monthly climatology is broadcast back to the original monthly time axis.
    """
    da = _make_monthly_clim_da()
    result = climatology.auto_anomaly(da, climatology_frequency="month")
    assert isinstance(result, xr.DataArray)
    assert result.shape == da.shape
    assert "time" in result.dims


def test_auto_anomaly_daily_clim_freq_broadcasts_to_original_shape():
    """auto_anomaly with climatology_frequency='dayofyear' and no anomaly frequency.

    The dayofyear climatology is broadcast back to the original daily time axis.
    """
    da = _make_daily_clim_da()
    result = climatology.auto_anomaly(da, climatology_frequency="dayofyear")
    assert isinstance(result, xr.DataArray)
    assert result.shape == da.shape
    assert "time" in result.dims


def test_auto_anomaly_climatology_range_restricts_clim_years():
    """climatology_range limits the years used to build the climatology.

    The anomaly is still returned over the full input time axis.
    """
    da = _make_monthly_clim_da(n_years=5, start_year=2000)
    result = climatology.auto_anomaly(da, frequency="month", climatology_range=("2000", "2002"))
    assert isinstance(result, xr.DataArray)
    assert "time" in result.dims
    # Full 5-year monthly series is preserved
    assert result.sizes["time"] == 60


def test_auto_anomaly_climatology_how_median():
    """climatology_how='median' produces the same result as 'mean' for constant-per-month data."""
    da = _make_monthly_clim_da()
    result_mean = climatology.auto_anomaly(da, frequency="month", climatology_how="mean")
    result_median = climatology.auto_anomaly(da, frequency="month", climatology_how="median")
    assert isinstance(result_median, xr.DataArray)
    np.testing.assert_allclose(result_mean.values, result_median.values, rtol=1e-10)


def test_auto_anomaly_dataset_input():
    """auto_anomaly accepts xr.Dataset and returns xr.Dataset."""
    da = _make_monthly_clim_da()
    ds = da.to_dataset(name="var")
    result = climatology.auto_anomaly(ds, frequency="month")
    assert isinstance(result, xr.Dataset)
    assert "var" in result
    assert "time" in result.dims


# --- climatology.quantile_categories (local) -------------------------------


def test_quantile_categories_local_values():
    """Values are categorised by the number of quantiles they are at or above.

    Quantiles without a time grouping are compared with every value, so no time dimension is needed.
    """
    da = xr.DataArray([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dims=["x"], name="var")
    clim = xr.DataArray([1.0, 2.0, 3.0], dims=["quantile"], coords={"quantile": [0.25, 0.5, 0.75]}, name="var")
    result = climatology.quantile_categories(da, clim)
    assert isinstance(result, xr.DataArray)
    # A value equal to a quantile falls in the category above it
    np.testing.assert_array_equal(result.values, [0, 1, 1, 2, 2, 3, 3])


def test_quantile_categories_local_monthly_climatology():
    """Each time step is compared with the quantiles of its own month."""
    da = _make_monthly_clim_da()  # values 1..12, by month
    months = np.arange(1, 13)
    clim = xr.DataArray(
        np.stack([months - 0.5, months + 0.5]),
        dims=["quantile", "month"],
        coords={"quantile": [0.1, 0.9], "month": months},
        name="var",
    )
    result = climatology.quantile_categories(da, clim)
    assert result.shape == da.shape
    # Every value lies between the quantiles of its month, but outside those of every other month
    np.testing.assert_array_equal(result.values, 1)


def test_quantile_categories_local_dayofyear_climatology():
    da = _make_daily_clim_da()
    clim = climatology.quantiles(da, [1 / 3, 2 / 3], frequency="dayofyear").compute()
    result = climatology.quantile_categories(da, clim)
    expected = [
        int((value >= clim.sel(dayofyear=dayofyear)).sum())
        for value, dayofyear in zip(da.values, da.time.dt.dayofyear.values)
    ]
    np.testing.assert_array_equal(result.values, expected)


def test_quantile_categories_local_percentiles():
    """Percentiles categorise the data as the equivalent quantiles do."""
    da = _make_daily_clim_da()
    from_quantiles = climatology.quantile_categories(da, climatology.quantiles(da, [0.1, 0.9], frequency="month"))
    from_percentiles = climatology.quantile_categories(da, climatology.percentiles(da, [10, 90], frequency="month"))
    xr.testing.assert_identical(from_quantiles, from_percentiles)


def test_quantile_categories_local_quantile_dim():
    da = _make_daily_clim_da()
    clim = climatology.quantiles(da, [0.1, 0.9], frequency="month")
    result = climatology.quantile_categories(da, clim.rename(quantile="threshold"), quantile_dim="threshold")
    xr.testing.assert_identical(result, climatology.quantile_categories(da, clim))


def test_quantile_categories_local_missing_values():
    """Categories are missing where the data or any of its quantiles are missing."""
    time = pd.date_range("2000-01-01", periods=3, freq="D")
    da = xr.DataArray([[np.nan, 1.0], [1.0, 1.0], [3.0, 3.0]], dims=["time", "x"], coords={"time": time}, name="var")
    clim = xr.DataArray(
        [[0.0, 0.0], [2.0, np.nan]], dims=["quantile", "x"], coords={"quantile": [0.1, 0.9]}, name="var"
    )
    result = climatology.quantile_categories(da, clim)
    np.testing.assert_array_equal(result.values, [[np.nan, np.nan], [1, np.nan], [2, np.nan]])


def test_quantile_categories_local_dataset_input():
    """quantile_categories accepts xr.Dataset and categorises each variable by its own climatology."""
    ds = _make_daily_clim_da().to_dataset(name="var")
    clim = climatology.quantiles(ds, [0.1, 0.9], frequency="month")
    result = climatology.quantile_categories(ds, clim)
    assert isinstance(result, xr.Dataset)
    xr.testing.assert_identical(result["var"], climatology.quantile_categories(ds["var"], clim["var"]))


def test_quantile_categories_local_attributes():
    """Categories keep the name and long_name of the data, but not its units or standard_name."""
    da = _make_monthly_clim_da().assign_attrs(units="K", standard_name="air_temperature", long_name="Temperature")
    clim = climatology.quantiles(da, [0.5], frequency="month")
    result = climatology.quantile_categories(da, clim)
    assert result.name == "var"
    assert result.attrs == {"long_name": "Temperature quantile category"}
    assert climatology.quantile_categories(da, clim, how_label="category").name == "var_category"


def test_quantile_categories_local_lazy():
    """Lazily loaded data is categorised lazily, keeping the time chunks of the data."""
    da = _make_daily_clim_da().chunk(time=365)
    clim = climatology.quantiles(da, [0.1, 0.9], frequency="month")
    result = climatology.quantile_categories(da, clim)
    assert result.chunksizes["time"] == da.chunksizes["time"]
    xr.testing.assert_identical(result.compute(), climatology.quantile_categories(da.compute(), clim.compute()))


def test_quantile_categories_local_lazy_data_eager_climatology():
    """Lazily loaded data is categorised lazily, without holding quantiles the size of the data in memory."""
    da = (_make_daily_clim_da() + xr.DataArray(np.zeros(500), dims=["x"])).chunk(time=365)
    clim = climatology.quantiles(da, [0.1, 0.9], frequency="month").compute()
    tracemalloc.start()
    try:
        result = climatology.quantile_categories(da, clim)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.chunksizes["time"] == da.chunksizes["time"]
    assert peak < da.nbytes
    xr.testing.assert_identical(result.compute(), climatology.quantile_categories(da.compute(), clim))


def test_quantile_categories_local_small_time_chunks():
    """Small time chunks are categorised without holding many groups of a lazily loaded climatology in memory."""
    time = pd.date_range("2000-01-01", periods=366, freq="D")
    da = xr.DataArray(dask.array.ones((366, 2000), chunks=(1, 2000)), dims=["time", "x"], coords={"time": time})
    clim = xr.DataArray(
        dask.array.zeros((2, 366, 2000), chunks=(1, 1, 2000)),
        dims=["quantile", "dayofyear", "x"],
        coords={"quantile": [0.1, 0.9], "dayofyear": np.arange(1, 367)},
    )
    result = climatology.quantile_categories(da, clim)
    tracemalloc.start()
    try:
        total = result.sum().compute(scheduler="single-threaded")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert total == 2 * 366 * 2000
    assert peak < 2 * clim.nbytes


def test_quantile_categories_local_missing_groups():
    """Time steps whose group is missing from the climatology are categorised as missing."""
    time = pd.date_range("2000-01-30", periods=4, freq="D")  # 30 and 31 January, 1 and 2 February
    da = xr.DataArray([0.0, 2.0, 0.0, 2.0], dims=["time"], coords={"time": time}, name="var")
    clim = xr.DataArray([[1.0]], dims=["quantile", "month"], coords={"quantile": [0.5], "month": [1]}, name="var")
    result = climatology.quantile_categories(da, clim)
    np.testing.assert_array_equal(result.values, [0, 1, np.nan, np.nan])


@pytest.mark.parametrize("lazy", (False, True))
def test_quantile_categories_local_time_coordinate_not_a_dimension(lazy):
    """Time steps along another dimension are matched to the groups of their time coordinate."""
    time = pd.date_range("2000-01-31", periods=3, freq="D")  # 31 January, 1 and 2 February
    da = xr.DataArray([0.0, 0.0, 0.0], dims=["obs"], coords={"valid_time": ("obs", time)}, name="var")
    clim = xr.DataArray(
        [[1.0, -1.0]], dims=["quantile", "month"], coords={"quantile": [0.5], "month": [1, 2]}, name="var"
    )
    if lazy:
        clim = clim.chunk(month=1)
    result = climatology.quantile_categories(da, clim, time_dim="valid_time")
    np.testing.assert_array_equal(result.values, [0, 1, 1])


def test_quantile_categories_local_missing_times():
    """Time steps at missing times are categorised as missing, even if the climatology has a missing group."""
    time = pd.DatetimeIndex(["2000-01-01", "NaT", "2000-02-01"])
    da = xr.DataArray([0.0, 0.0, 0.0], dims=["time"], coords={"time": time}, name="var")
    clim = xr.DataArray(
        [[1.0, 1.0, -1.0]],
        dims=["quantile", "month"],
        coords={"quantile": [0.5], "month": [1.0, 2.0, np.nan]},
        name="var",
    )
    np.testing.assert_array_equal(climatology.quantile_categories(da, clim).values, [0, np.nan, 0])


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("labelled", (False, True))
def test_quantile_categories_local_data_with_a_group_dimension(labelled, lazy):
    """A dimension of the data named like the climatology's groups is not matched to the groups."""
    coords = {"time": pd.date_range("2000-01-01", periods=3, freq="MS")}
    if labelled:
        coords["month"] = [0, 1]
    da = xr.DataArray(np.zeros((3, 2)), dims=["time", "month"], coords=coords, name="var")
    clim = xr.DataArray(
        [[1.0, -1.0, 1.0]], dims=["quantile", "month"], coords={"quantile": [0.5], "month": [1, 2, 3]}, name="var"
    )
    if lazy:
        da, clim = da.chunk(month=1), clim.chunk(month=1)
    result = climatology.quantile_categories(da, clim)
    np.testing.assert_array_equal(result.values, [[0, 0], [1, 1], [0, 0]])


def test_quantile_categories_local_time_shift():
    """Time steps are matched to the groups of their shifted times."""
    time = pd.date_range("2000-01-31T22:00", periods=3, freq="h")  # 22:00, 23:00 and 00:00
    da = xr.DataArray([0.0, 0.0, 0.0], dims=["time"], coords={"time": time}, name="var")
    clim = xr.DataArray(
        [[1.0, -1.0]], dims=["quantile", "month"], coords={"quantile": [0.5], "month": [1, 2]}, name="var"
    )
    np.testing.assert_array_equal(climatology.quantile_categories(da, clim).values, [0, 0, 1])
    np.testing.assert_array_equal(climatology.quantile_categories(da, clim, time_shift="1h").values, [0, 1, 1])


@pytest.mark.parametrize("lazy", (False, True))
def test_quantile_categories_local_scalar_coordinates(lazy):
    """Scalar coordinates, such as the quantile of an ensemble median, are kept rather than conflicting."""
    ensemble = xr.DataArray(
        np.arange(24.0).reshape(2, 12),
        dims=["member", "time"],
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS")},
        name="var",
    )
    median = ensemble.quantile(0.5, dim="member")  # 6..17, with a scalar quantile coordinate
    months = np.arange(1, 13)
    clim = xr.DataArray(
        np.stack([months + 4.5, months + 5.5]),
        dims=["quantile", "month"],
        coords={"quantile": [0.1, 0.9], "month": months},
        name="var",
    )
    if lazy:
        clim = clim.chunk()
    result = climatology.quantile_categories(median, clim)
    np.testing.assert_array_equal(result.values, 1)
    assert float(result["quantile"]) == 0.5
    # A time coordinate left on the climatology does not conflict with the time steps of the data
    clim_with_time = clim.assign_coords(time=pd.Timestamp("1990-01-01"))
    xr.testing.assert_identical(climatology.quantile_categories(median, clim_with_time), result)


def test_quantile_categories_local_missing_coordinates():
    """Points without quantiles are categorised as missing, rather than dropped."""
    da = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims=["x"], coords={"x": [0, 1, 2, 3]}, name="var")
    clim = xr.DataArray([[1.0, 1.0], [2.0, 2.0]], dims=["quantile", "x"], coords={"x": [1, 2]}, name="var")
    result = climatology.quantile_categories(da, clim)
    np.testing.assert_array_equal(result.x, da.x)
    np.testing.assert_array_equal(result.values, [np.nan, 1, 2, np.nan])


def test_quantile_categories_local_no_quantiles():
    """Without quantiles, every value is in the one category, and missing values stay missing."""
    da = xr.DataArray([np.nan, 1.0], dims=["x"], name="var")
    result = climatology.quantile_categories(da, xr.DataArray([], dims=["quantile"]))
    np.testing.assert_array_equal(result.values, [np.nan, 0])


def test_quantile_categories_local_quantile_dim_in_data_raises():
    da = xr.DataArray([0.0, 1.0], dims=["quantile"], name="var")
    with pytest.raises(ValueError, match="also in the dataarray"):
        climatology.quantile_categories(da, xr.DataArray([0.5, 1.5], dims=["quantile"]))


def test_quantile_categories_local_no_quantile_dim_raises():
    da = _make_monthly_clim_da()
    with pytest.raises(ValueError, match="quantile dimension"):
        climatology.quantile_categories(da, climatology.monthly_mean(da))


def test_quantile_categories_local_ungrouped_climatology_raises():
    """Quantiles grouped by year, the default of climatology.quantiles, cannot be matched to the data."""
    da = _make_monthly_clim_da()
    with pytest.raises(ValueError, match="year"):
        climatology.quantile_categories(da, climatology.quantiles(da, [0.5]))
