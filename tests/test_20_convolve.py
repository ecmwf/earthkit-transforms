import numpy as np
import pytest
import xarray as xr

from earthkit.transforms._convolve import convolve


VALID_COMBINATIONS = [
    ("direct", "zeropad"),
    ("fft", "zeropad"),
    ("fft", "periodic"),
]


@pytest.fixture
def window() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def data_1d() -> xr.DataArray:
    return xr.DataArray(np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0, 8.0, 1.0]), dims=["t"])


@pytest.fixture
def data_3d(data_1d) -> xr.DataArray:
    base_a = xr.DataArray([1.0, -2.0, 0.5], dims=["a"])
    base_b = xr.DataArray([1.0, -1.0, 3.0], dims=["b"])
    return (data_1d * base_a * base_b).transpose("a", "t", "b")


@pytest.mark.parametrize("how_method, how_boundary", VALID_COMBINATIONS)
def test_convolve_dataset_matches_dataarrays(how_method, how_boundary, data_1d, window):
    dataset = xr.Dataset({"temperature": data_1d, "pressure": data_1d * 2})
    result = convolve(dataset, window, "t", how_method=how_method, how_boundary=how_boundary)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == set(dataset.data_vars)
    for var in dataset.data_vars:
        expected = convolve(dataset[var], window, "t", how_method=how_method, how_boundary=how_boundary)
        xr.testing.assert_allclose(result[var], expected, atol=1e-8)


@pytest.mark.parametrize("how_method, how_boundary", VALID_COMBINATIONS)
def test_convolve_matches_impulse_response(how_method, how_boundary):
    window = np.array([1.0, 2.0, 3.0])
    impulse_index = 5
    impulse = np.zeros(11)
    impulse[impulse_index] = 1.0
    impulse_data = xr.DataArray(impulse, dims=["t"])
    # Analytical solution: convolving an impulse with a window just places
    # the window itself, starting at (impulse_index - start).
    expected = np.zeros_like(impulse_data)
    lo = impulse_index - (window.size - 1) // 2
    expected[lo : lo + window.size] = window
    result = convolve(impulse_data, window, "t", how_method=how_method, how_boundary=how_boundary)
    assert result.dims == impulse_data.dims
    np.testing.assert_allclose(result.values, expected, atol=1e-8)


def test_convolve_fft_periodic_matches_analytical_sinusoid():
    n = 19
    freq = 3
    window = np.array([1.0, 1.0, 1.0])
    start = (window.size - 1) // 2
    t = np.arange(n)
    data = xr.DataArray(np.cos(2 * np.pi * freq * t / n), dims=["t"])
    # Same frequency, rescaled and phase-shifted by the window's frequency
    # response plus a phase term from the centering roll
    response = np.fft.rfft(window, n=n)[freq]
    expected = np.abs(response) * np.cos(
        2 * np.pi * freq * t / n + np.angle(response) + 2 * np.pi * freq * start / n
    )
    result = convolve(data, window, "t", how_method="fft", how_boundary="periodic")
    np.testing.assert_allclose(result.values, expected, atol=1e-8)


@pytest.mark.parametrize("window_values", ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]))
def test_convolve_direct_and_fft_agree_for_zeropad(data_1d, window_values):
    window = np.array(window_values)
    direct = convolve(data_1d, window, "t", how_method="direct", how_boundary="zeropad")
    fft = convolve(data_1d, window, "t", how_method="fft", how_boundary="zeropad")
    xr.testing.assert_allclose(direct, fft, atol=1e-8)


@pytest.mark.parametrize("how_method, how_boundary", VALID_COMBINATIONS)
@pytest.mark.parametrize("dim", ["a", "t", "b"])
def test_convolve_along_dim(how_method, how_boundary, dim, data_3d, window):
    result = convolve(data_3d, window, dim, how_method=how_method, how_boundary=how_boundary)
    assert result.dims == data_3d.dims
    assert result.shape == data_3d.shape
    # Check result for first element
    selection = {d: 0 for d in data_3d.dims if d != dim}
    expected_1d = convolve(data_3d.isel(selection), window, dim, how_method=how_method, how_boundary=how_boundary)
    xr.testing.assert_allclose(result.isel(selection), expected_1d, atol=1e-8)


@pytest.mark.parametrize("how_method", ["auto", "direct"])
def test_convolve_preserves_integer_dtype(how_method):
    data = xr.DataArray(np.arange(9), dims=["t"])
    window = np.array([1, 1, 1])
    result = convolve(data, window, "t", how_method=how_method, how_boundary="zeropad")
    assert result.dtype.kind == "i"
    expected = np.array([1, 3, 6, 9, 12, 15, 18, 21, 15])
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("how_method, how_boundary", VALID_COMBINATIONS)
def test_convolve_how_label_renames_dataarray(how_method, how_boundary, data_1d, window):
    named = data_1d.rename("temperature")
    result = convolve(named, window, "t", how_method=how_method, how_boundary=how_boundary, how_label="smoothed")
    assert result.name == "temperature_smoothed"


@pytest.mark.parametrize("how_method, how_boundary", VALID_COMBINATIONS)
def test_convolve_how_label_none_leaves_name_unchanged(how_method, how_boundary, data_1d, window):
    named = data_1d.rename("temperature")
    result = convolve(named, window, "t", how_method=how_method, how_boundary=how_boundary)
    assert result.name == "temperature"


def test_convolve_raises_for_multidimensional_window(data_1d):
    with pytest.raises(ValueError, match="1-dimensional"):
        convolve(data_1d, np.ones((2, 2)), "t")


def test_convolve_raises_for_empty_window(data_1d):
    with pytest.raises(ValueError, match="non-empty"):
        convolve(data_1d, [], "t")
