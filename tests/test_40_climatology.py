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
