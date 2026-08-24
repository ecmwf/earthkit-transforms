import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earthkit import transforms as ekt
from earthkit.transforms import fourier as _fourier
from earthkit.transforms.fourier import fft, ifft


def _signal_dataarray(n=128, dt=1.0, freq=0.1, dim="x"):
    coord = np.arange(n) * dt
    values = np.sin(2 * np.pi * freq * coord)
    return xr.DataArray(values, coords={dim: coord}, dims=(dim,), name="signal")


def test_fft_returns_complex_with_frequency_dim():
    da = _signal_dataarray()
    result = fft(da, dim="x")
    assert isinstance(result, xr.DataArray)
    assert "frequency" in result.dims
    assert "x" not in result.dims
    assert np.iscomplexobj(result.values)
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(128, d=1.0))


def test_fft_detects_dominant_frequency():
    da = _signal_dataarray(n=200, dt=0.5, freq=0.2)
    result = fft(da, dim="x")
    peak_freq = np.abs(result["frequency"].values[np.argmax(np.abs(result.values))])
    assert peak_freq == pytest.approx(0.2, abs=1e-9)


def test_fft_ifft_roundtrip():
    da = _signal_dataarray(n=64, dt=2.0, freq=0.05)
    spectrum = fft(da, dim="x")
    restored = ifft(spectrum, dim="frequency", output_dim="x", output_coord=da["x"].values)
    np.testing.assert_allclose(restored.real.values, da.values, atol=1e-9)
    np.testing.assert_allclose(restored["x"].values, da["x"].values)


def test_fft_ifft_roundtrip_uses_source_dim_by_default():
    da = _signal_dataarray(dim="x")
    spectrum = fft(da, dim="x")
    restored = ifft(spectrum, dim="frequency")
    assert "x" in restored.dims


def test_fft_dataset():
    da = _signal_dataarray()
    ds = xr.Dataset({"a": da, "b": da * 2})
    result = fft(ds, dim="x")
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"a", "b"}
    assert "frequency" in result.dims


def test_fft_missing_dim_raises():
    da = _signal_dataarray()
    with pytest.raises(ValueError):
        fft(da, dim="not_a_dim")


def _temporal_dataarray(n=48, freq_per_hour=1 / 24):
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    seconds = (times - times[0]).total_seconds().to_numpy()
    values = np.sin(2 * np.pi * freq_per_hour * (seconds / 3600))
    return xr.DataArray(values, coords={"time": times}, dims=("time",), name="signal")


def test_temporal_fft_detects_time_dim_and_hz_frequency():
    da = _temporal_dataarray(n=48)
    result = ekt.temporal.fft(da)
    assert "frequency" in result.dims
    assert "time" not in result.dims
    # Hourly sampling -> spacing of 3600 s -> frequencies in Hz
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(48, d=3600.0))


def test_temporal_fft_time_dim_override():
    da = _temporal_dataarray().rename({"time": "forecast_time"})
    result = ekt.temporal.fft(da, time_dim="forecast_time")
    assert "frequency" in result.dims


def test_temporal_fft_ifft_roundtrip():
    da = _temporal_dataarray(n=72)
    spectrum = ekt.temporal.fft(da)
    restored = ekt.temporal.ifft(spectrum, time_dim="time", time_coord=da["time"].values)
    np.testing.assert_allclose(restored.real.values, da.values, atol=1e-9)
    np.testing.assert_array_equal(restored["time"].values, da["time"].values)


@pytest.mark.parametrize("transform", ["fft", "rfft"])
def test_temporal_transform_passes_static_variables_through(transform):
    # A Dataset mixing a time-varying field with a static one (e.g. a land-sea mask)
    # must transform the former and pass the latter through untouched.
    signal = _temporal_dataarray(n=48)
    static = xr.DataArray([0.0, 1.0], dims=("location",), name="lsm")
    ds = xr.Dataset({"t2m": signal, "lsm": static})
    result = getattr(ekt.temporal, transform)(ds)
    assert "frequency" in result["t2m"].dims
    assert "time" not in result["t2m"].dims
    assert result["lsm"].dims == ("location",)
    np.testing.assert_array_equal(result["lsm"].values, static.values)


# ------------------------------------------------------------------------------------------
# Convenience `period` coordinate on temporal wrappers
# ------------------------------------------------------------------------------------------
def _assert_period_matches_frequency(result, units="s"):
    assert "period" in result.coords
    assert result["period"].dims == ("frequency",)
    freqs = result["frequency"].values
    period = result["period"]
    nonzero = freqs != 0.0
    if units is None:
        # Unlabelled periods stay floating-point, with NaN at the zero-frequency term.
        periods = period.values
        np.testing.assert_allclose(periods[nonzero], 1.0 / freqs[nonzero])
        assert np.all(np.isnan(periods[~nonzero]))
        assert "units" not in period.attrs
    else:
        # Time-valued periods are typed as durations, with NaT at the zero-frequency term.
        assert np.issubdtype(period.dtype, np.timedelta64)
        expected = pd.to_timedelta(1.0 / freqs[nonzero], unit=units).to_numpy()
        np.testing.assert_array_equal(period.values[nonzero], expected)
        assert np.all(np.isnat(period.values[~nonzero]))
        assert "units" not in period.attrs


def test_temporal_fft_adds_period_coordinate():
    da = _temporal_dataarray(n=48)
    result = ekt.temporal.fft(da)
    _assert_period_matches_frequency(result, units="s")


def test_temporal_rfft_adds_period_coordinate():
    da = _temporal_dataarray(n=48)
    result = ekt.temporal.rfft(da)
    _assert_period_matches_frequency(result, units="s")


def test_temporal_ihfft_adds_period_coordinate():
    da = _temporal_dataarray(n=48)
    result = ekt.temporal.ihfft(da)
    _assert_period_matches_frequency(result, units="s")


def test_temporal_fftfreq_adds_period_coordinate_without_units():
    freqs = ekt.temporal.fftfreq(10, sample_spacing=3600.0)
    _assert_period_matches_frequency(freqs, units=None)


def test_temporal_rfftfreq_adds_period_coordinate_without_units():
    freqs = ekt.temporal.rfftfreq(10, sample_spacing=3600.0)
    _assert_period_matches_frequency(freqs, units=None)


def test_temporal_fft_period_identifies_dominant_cycle():
    # 24-hour cycle sampled hourly -> dominant period of 24 h (86400 s).
    da = _temporal_dataarray(n=48, freq_per_hour=1 / 24)
    result = ekt.temporal.fft(da)
    power = np.abs(result)
    positive = power.where(result["frequency"] > 0, drop=True)
    dominant_period = positive["period"].isel(frequency=positive.argmax("frequency"))
    assert dominant_period.values / np.timedelta64(1, "s") == pytest.approx(24 * 3600.0, rel=1e-6)


# ------------------------------------------------------------------------------------------
# Array API standard coverage: rfft / irfft / hfft / ihfft
# ------------------------------------------------------------------------------------------
def test_rfft_output_length_and_roundtrip():
    da = _signal_dataarray(n=64, dt=1.0, freq=0.1)
    spectrum = _fourier.rfft(da, dim="x")
    assert spectrum.sizes["frequency"] == 64 // 2 + 1
    np.testing.assert_allclose(spectrum["frequency"].values, np.fft.rfftfreq(64, d=1.0))
    restored = _fourier.irfft(spectrum, dim="frequency", n=64, output_coord=da["x"].values)
    np.testing.assert_allclose(restored.values, da.values, atol=1e-9)
    assert np.isrealobj(restored.values)
    assert "x" in restored.dims


def test_hfft_ihfft_roundtrip():
    da = _signal_dataarray(n=50, dt=1.0, freq=0.05)
    spectrum = _fourier.ihfft(da, dim="x")
    assert spectrum.sizes["frequency"] == 50 // 2 + 1
    assert np.iscomplexobj(spectrum.values)
    restored = _fourier.hfft(spectrum, dim="frequency", n=50, output_coord=da["x"].values)
    np.testing.assert_allclose(restored.values, da.values, atol=1e-9)
    assert np.isrealobj(restored.values)


# ------------------------------------------------------------------------------------------
# Array API standard coverage: fftn / ifftn / rfftn / irfftn
# ------------------------------------------------------------------------------------------
def _grid_dataarray(nx=8, ny=6):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ny, nx))
    return xr.DataArray(
        data,
        coords={"y": np.arange(ny) * 1.0, "x": np.arange(nx) * 1.0},
        dims=("y", "x"),
        name="field",
    )


def test_fftn_ifftn_roundtrip():
    da = _grid_dataarray()
    spectrum = _fourier.fftn(da, dims=["y", "x"])
    assert set(spectrum.dims) == {"y_frequency", "x_frequency"}
    np.testing.assert_allclose(spectrum["x_frequency"].values, np.fft.fftfreq(8, d=1.0))
    restored = _fourier.ifftn(spectrum, dims=["y_frequency", "x_frequency"])
    assert set(restored.dims) == {"y", "x"}
    np.testing.assert_allclose(restored.real.transpose("y", "x").values, da.values, atol=1e-9)


def test_rfftn_irfftn_roundtrip():
    da = _grid_dataarray(nx=8, ny=6)
    spectrum = _fourier.rfftn(da, dims=["y", "x"])
    # Last transformed dim is halved
    assert spectrum.sizes["x_frequency"] == 8 // 2 + 1
    assert spectrum.sizes["y_frequency"] == 6
    restored = _fourier.irfftn(spectrum, dims=["y_frequency", "x_frequency"], s=[6, 8])
    assert set(restored.dims) == {"y", "x"}
    np.testing.assert_allclose(restored.transpose("y", "x").values, da.values, atol=1e-9)


# ------------------------------------------------------------------------------------------
# Array API standard coverage: fftfreq / rfftfreq / fftshift / ifftshift
# ------------------------------------------------------------------------------------------
def test_fftfreq_and_rfftfreq():
    freqs = _fourier.fftfreq(10, sample_spacing=0.5)
    assert isinstance(freqs, xr.DataArray)
    np.testing.assert_allclose(freqs.values, np.fft.fftfreq(10, d=0.5))
    rfreqs = _fourier.rfftfreq(10, sample_spacing=0.5)
    np.testing.assert_allclose(rfreqs.values, np.fft.rfftfreq(10, d=0.5))
    assert rfreqs.sizes["frequency"] == 10 // 2 + 1


def test_fftshift_ifftshift_roundtrip_and_values():
    da = _signal_dataarray(n=8, dt=1.0, freq=0.1)
    spectrum = _fourier.fft(da, dim="x")
    shifted = _fourier.fftshift(spectrum, dim="frequency")
    np.testing.assert_allclose(shifted["frequency"].values, np.fft.fftshift(spectrum["frequency"].values))
    np.testing.assert_allclose(shifted.values, np.fft.fftshift(spectrum.values))
    unshifted = _fourier.ifftshift(shifted, dim="frequency")
    np.testing.assert_allclose(unshifted["frequency"].values, spectrum["frequency"].values)
    np.testing.assert_allclose(unshifted.values, spectrum.values)


def test_fft_with_n_padding_changes_length():
    da = _signal_dataarray(n=50, dt=1.0, freq=0.1)
    spectrum = _fourier.fft(da, dim="x", n=64)
    assert spectrum.sizes["frequency"] == 64


# ------------------------------------------------------------------------------------------
# Sample-spacing inference edge cases
# ------------------------------------------------------------------------------------------
def test_fft_without_coordinate_uses_unit_spacing():
    da = xr.DataArray(np.sin(np.arange(16) * 0.5), dims=("x",), name="signal")
    result = _fourier.fft(da, dim="x")
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(16, d=1.0))


def test_fft_single_point_coordinate_uses_unit_spacing():
    da = xr.DataArray([1.0], coords={"x": [0.0]}, dims=("x",), name="signal")
    result = _fourier.fft(da, dim="x")
    assert result.sizes["frequency"] == 1
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(1, d=1.0))


@pytest.mark.parametrize("calendar", ["360_day", "noleap", "julian"])
def test_temporal_fft_cftime_calendar(calendar):
    # cftime coordinates (non-standard calendars) also yield a spacing in seconds
    pytest.importorskip("cftime")
    times = xr.date_range("2000-01-01", periods=48, freq="h", calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.sin(np.arange(48) * 0.3), coords={"time": times}, dims=("time",), name="signal")
    result = ekt.temporal.fft(da)
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(48, d=3600.0))


def test_fft_irregular_spacing_warns(caplog):
    # Coordinate with a non-uniform step (0, 1, 2, 4)
    da = xr.DataArray(np.arange(4.0), coords={"x": [0.0, 1.0, 2.0, 4.0]}, dims=("x",), name="signal")
    with caplog.at_level("WARNING"):
        _fourier.fft(da, dim="x")
    assert any("not regularly spaced" in record.message for record in caplog.records)


def test_fft_regular_spacing_does_not_warn(caplog):
    da = _signal_dataarray(n=32, dt=2.0, freq=0.1)
    with caplog.at_level("WARNING"):
        _fourier.fft(da, dim="x")
    assert not any("regularly spaced" in record.message for record in caplog.records)


def test_fft_explicit_sample_spacing_skips_spacing_check(caplog):
    # Irregular coordinate, but an explicit sample_spacing bypasses inference (no warning)
    da = xr.DataArray(np.arange(4.0), coords={"x": [0.0, 1.0, 2.0, 4.0]}, dims=("x",), name="signal")
    with caplog.at_level("WARNING"):
        _fourier.fft(da, dim="x", sample_spacing=1.0)
    assert not any("regularly spaced" in record.message for record in caplog.records)


# ------------------------------------------------------------------------------------------
# Default dimensions and error handling
# ------------------------------------------------------------------------------------------
def test_fftn_default_dims_transforms_all():
    da = _grid_dataarray(nx=8, ny=6)
    spectrum = _fourier.fftn(da)
    assert set(spectrum.dims) == {"y_frequency", "x_frequency"}


def test_fftshift_default_dims_shifts_all():
    da = _grid_dataarray(nx=8, ny=6)
    shifted = _fourier.fftshift(da)
    np.testing.assert_allclose(shifted.values, np.fft.fftshift(da.values))
    unshifted = _fourier.ifftshift(da)
    np.testing.assert_allclose(unshifted.values, np.fft.ifftshift(da.values))


def test_ifft_missing_dim_raises():
    da = _signal_dataarray()
    with pytest.raises(ValueError):
        _fourier.ifft(da, dim="not_a_dim")


def test_fftn_missing_dim_raises():
    da = _grid_dataarray()
    with pytest.raises(ValueError):
        _fourier.fftn(da, dims=["not_a_dim"])


def test_fftn_s_length_mismatch_raises():
    da = _grid_dataarray()
    with pytest.raises(ValueError):
        _fourier.fftn(da, dims=["y", "x"], s=[8])


def test_irfftn_default_output_size_roundtrip():
    da = _grid_dataarray(nx=8, ny=6)
    spectrum = _fourier.rfftn(da, dims=["y", "x"])
    # No `s` -> last dim defaults to 2 * (m - 1), recovering the even original length
    restored = _fourier.irfftn(spectrum, dims=["y_frequency", "x_frequency"])
    assert set(restored.dims) == {"y", "x"}
    np.testing.assert_allclose(restored.transpose("y", "x").values, da.values, atol=1e-9)


def test_ifftn_explicit_output_dims_and_coords():
    da = _grid_dataarray(nx=8, ny=6)
    spectrum = _fourier.fftn(da, dims=["y", "x"])
    restored = _fourier.ifftn(
        spectrum,
        dims=["y_frequency", "x_frequency"],
        output_dims=["yy", "xx"],
        output_coords={"yy": np.arange(6), "xx": np.arange(8)},
    )
    assert set(restored.dims) == {"yy", "xx"}
    np.testing.assert_array_equal(restored["xx"].values, np.arange(8))


def test_rfft_dataset():
    da = _signal_dataarray()
    ds = xr.Dataset({"a": da, "b": da * 2})
    result = _fourier.rfft(ds, dim="x")
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"a", "b"}
    assert result.sizes["frequency"] == 128 // 2 + 1


# ------------------------------------------------------------------------------------------
# Frequency and period units follow the provenance of the sample spacing
# ------------------------------------------------------------------------------------------
def test_temporal_fft_datetime_coordinate_is_labelled_hz():
    result = ekt.temporal.fft(_temporal_dataarray(n=48))
    assert result["frequency"].attrs["units"] == "Hz"
    # A frequency in Hz gives a period that is a timedelta duration, not a units-labelled float.
    assert np.issubdtype(result["period"].dtype, np.timedelta64)
    assert "units" not in result["period"].attrs


def test_temporal_fft_numeric_coordinate_is_unlabelled():
    # A numeric time coordinate has unknown units, so neither coordinate may claim any.
    da = xr.DataArray(np.sin(np.arange(48) * 0.3), coords={"time": np.arange(48.0)}, dims=("time",))
    result = ekt.temporal.fft(da)
    assert "units" not in result["frequency"].attrs
    assert "units" not in result["period"].attrs
    # The values are still the reciprocal of the coordinate's own units (hours, here).
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(48, d=1.0))


def test_temporal_fft_explicit_sample_spacing_units_are_used():
    da = _temporal_dataarray(n=48)
    result = ekt.temporal.fft(da, sample_spacing=1.0, sample_spacing_units="h")
    assert result["frequency"].attrs["units"] == "h-1"
    # Hours are a time unit, so the period is a timedelta duration of 48 hours.
    assert np.issubdtype(result["period"].dtype, np.timedelta64)
    assert result["period"].values[1] / np.timedelta64(1, "h") == pytest.approx(48.0)


def test_temporal_fft_explicit_sample_spacing_without_units_is_unlabelled():
    result = ekt.temporal.fft(_temporal_dataarray(n=48), sample_spacing=3600.0)
    assert "units" not in result["frequency"].attrs
    assert "units" not in result["period"].attrs


def test_temporal_fftfreq_units_are_opt_in():
    assert "units" not in ekt.temporal.fftfreq(10, sample_spacing=3600.0)["period"].attrs
    labelled = ekt.temporal.rfftfreq(10, sample_spacing=3600.0, sample_spacing_units="s")
    assert labelled["frequency"].attrs["units"] == "Hz"
    # Seconds are a time unit, so the labelled period is a timedelta duration.
    assert np.issubdtype(labelled["period"].dtype, np.timedelta64)
    assert "units" not in labelled["period"].attrs


# ------------------------------------------------------------------------------------------
# Datasets carrying variables without the transform dimension
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("func", [ekt.temporal.fft, ekt.temporal.rfft, ekt.temporal.ihfft])
def test_temporal_transforms_pass_through_variables_without_time(func):
    da = _temporal_dataarray(n=48)
    static = xr.DataArray([1.0, 2.0], coords={"z": [10.0, 20.0]}, dims=("z",))
    result = func(xr.Dataset({"t2m": da, "orography": static}))
    assert set(result.data_vars) == {"t2m", "orography"}
    assert result["orography"].dims == ("z",)
    np.testing.assert_array_equal(result["orography"].values, static.values)
    assert "frequency" in result["t2m"].dims


def test_fftn_passes_through_variables_without_transform_dims():
    ds = xr.Dataset({"field": _grid_dataarray(nx=8, ny=6), "scalar": xr.DataArray(3.0)})
    result = _fourier.fftn(ds, dims=["y", "x"])
    assert float(result["scalar"]) == 3.0
    assert set(result["field"].dims) == {"y_frequency", "x_frequency"}


# ------------------------------------------------------------------------------------------
# Exact round trips, including odd-length signals
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [48, 49, 50, 51])
def test_temporal_rfft_irfft_roundtrip_is_exact_for_odd_and_even(n):
    da = _temporal_dataarray(n=n)
    restored = ekt.temporal.irfft(ekt.temporal.rfft(da))
    assert restored.sizes["time"] == n
    np.testing.assert_allclose(restored.values, da.values, atol=1e-9)


@pytest.mark.parametrize("n", [48, 49])
def test_temporal_ihfft_hfft_roundtrip_is_exact_for_odd_and_even(n):
    da = _temporal_dataarray(n=n)
    restored = ekt.temporal.hfft(ekt.temporal.ihfft(da))
    assert restored.sizes["time"] == n
    np.testing.assert_allclose(restored.values, da.values, atol=1e-9)


@pytest.mark.parametrize("nx", [8, 9])
def test_rfftn_irfftn_roundtrip_is_exact_for_odd_and_even(nx):
    da = _grid_dataarray(nx=nx, ny=6)
    spectrum = _fourier.rfftn(da, dims=["y", "x"])
    restored = _fourier.irfftn(spectrum, dims=["y_frequency", "x_frequency"])
    assert restored.sizes["x"] == nx
    np.testing.assert_allclose(restored.transpose("y", "x").values, da.values, atol=1e-9)


def test_irfft_ignores_recorded_size_once_the_spectrum_is_sliced():
    # A stale source-size attribute must not survive slicing: fall back to 2 * (m - 1).
    da = _temporal_dataarray(n=49)
    spectrum = ekt.temporal.rfft(da).isel(frequency=slice(0, 20))
    assert ekt.temporal.irfft(spectrum).sizes["time"] == 2 * (20 - 1)


def test_irfft_explicit_n_overrides_recorded_size():
    spectrum = ekt.temporal.rfft(_temporal_dataarray(n=49))
    assert ekt.temporal.irfft(spectrum, n=64).sizes["time"] == 64


def test_irfft_without_provenance_falls_back_to_even_length():
    da = _signal_dataarray(n=49, dim="x")
    spectrum = _fourier.rfft(da, dim="x").drop_vars("frequency")
    assert _fourier.irfft(spectrum, dim="frequency").sizes["frequency"] == 2 * (25 - 1)


# ------------------------------------------------------------------------------------------
# The inverse transforms restore the dimension the forward transform recorded
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("time_dim", ["time", "valid_time", "forecast_time"])
def test_temporal_inverse_restores_the_source_time_dim(time_dim):
    # The forward transform is told the time dimension explicitly, so that this exercises the
    # inverse restoring what was recorded rather than the time-dimension detection.
    da = _temporal_dataarray(n=48).rename({"time": time_dim})
    assert ekt.temporal.ifft(ekt.temporal.fft(da, time_dim=time_dim)).dims == (time_dim,)
    assert ekt.temporal.irfft(ekt.temporal.rfft(da, time_dim=time_dim)).dims == (time_dim,)
    assert ekt.temporal.hfft(ekt.temporal.ihfft(da, time_dim=time_dim)).dims == (time_dim,)


def test_temporal_inverse_explicit_time_dim_wins():
    spectrum = ekt.temporal.fft(_temporal_dataarray(n=48).rename({"time": "valid_time"}))
    assert ekt.temporal.ifft(spectrum, time_dim="step").dims == ("step",)


def test_temporal_inverse_falls_back_to_time_without_provenance():
    da = _signal_dataarray(n=16, dim="x")
    spectrum = _fourier.fft(da, dim="x").drop_vars("frequency")
    assert ekt.temporal.ifft(spectrum).dims == ("time",)


# ------------------------------------------------------------------------------------------
# Output dtype matches the input precision, eagerly and under dask
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "complex_dtype", "real_dtype"),
    [(np.float32, np.complex64, np.float32), (np.float64, np.complex128, np.float64)],
)
def test_forward_transforms_preserve_precision(dtype, complex_dtype, real_dtype):
    pytest.importorskip("dask")
    da = xr.DataArray(np.sin(np.arange(16)).astype(dtype), dims=("x",), name="signal")
    for func in (_fourier.fft, _fourier.rfft, _fourier.ihfft):
        eager = func(da, dim="x")
        lazy = func(da.chunk({"x": -1}), dim="x")
        assert eager.dtype == complex_dtype
        # The dask meta must agree with what the graph actually produces.
        assert lazy.dtype == lazy.compute().dtype == complex_dtype
    spectrum = _fourier.rfft(da, dim="x")
    eager = _fourier.irfft(spectrum, dim="frequency")
    lazy = _fourier.irfft(spectrum.chunk({"frequency": -1}), dim="frequency")
    assert eager.dtype == real_dtype
    assert lazy.dtype == lazy.compute().dtype == real_dtype


def test_dataset_output_dtype_promotes_across_variables():
    da32 = xr.DataArray(np.sin(np.arange(16)).astype(np.float32), dims=("x",))
    da64 = xr.DataArray(np.sin(np.arange(16)).astype(np.float64), dims=("x",))
    result = _fourier.fft(xr.Dataset({"a": da32, "b": da64}), dim="x")
    assert result["b"].dtype == np.complex128


# ------------------------------------------------------------------------------------------
# earthkit-data objects are accepted by every data-taking temporal entry point
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fftshift", "ifftshift"],
)
def test_temporal_data_taking_entry_points_are_format_handled(name):
    # Guards against a new entry point being added without the decorator that lets it
    # accept earthkit-data objects, which is invisible until someone passes one in.
    assert hasattr(getattr(ekt.temporal, name), "__wrapped__"), f"temporal.{name} is not format-handled"


# ------------------------------------------------------------------------------------------
# Degenerate and descending coordinates
# ------------------------------------------------------------------------------------------
def test_fft_descending_coordinate_gives_positive_frequencies(caplog):
    da = _temporal_dataarray(n=48)
    with caplog.at_level("WARNING"):
        result = ekt.temporal.fft(da.isel(time=slice(None, None, -1)))
    assert any("descending" in record.message for record in caplog.records)
    np.testing.assert_allclose(result["frequency"].values, np.fft.fftfreq(48, d=3600.0))


def test_fft_zero_spacing_raises_a_named_error():
    da = xr.DataArray(np.ones(4), coords={"x": [1.0, 1.0, 1.0, 1.0]}, dims=("x",))
    with pytest.raises(ValueError, match="sample spacing of zero"):
        _fourier.fft(da, dim="x")


def test_fft_irregular_spacing_warning_reports_the_spread(caplog):
    da = xr.DataArray(np.arange(4.0), coords={"x": [0.0, 1.0, 2.0, 4.0]}, dims=("x",))
    with caplog.at_level("WARNING"):
        _fourier.fft(da, dim="x")
    message = next(r.message for r in caplog.records if "not regularly spaced" in r.message)
    assert "varies between 1 and 2" in message
