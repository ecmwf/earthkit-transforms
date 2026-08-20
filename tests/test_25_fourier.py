import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earthkit import transforms as ekt
from earthkit.transforms import _fourier
from earthkit.transforms._fourier import fft, ifft


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
