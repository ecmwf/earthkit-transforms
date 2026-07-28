# Copyright 2026-, European Centre for Medium Range Weather Forecasts.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import xarray as xr

from earthkit.transforms._aggregate import how_label_rename
from earthkit.utils.array import array_namespace


def convolve(dataarray, *_args, **kwargs):
    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _convolve_dataarray(dataarray[var], *_args, **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _convolve_dataarray(dataarray, *_args, **kwargs)


def _convolve_dataarray(
    dataarray,
    window,
    dim,
    *,
    how_boundary="zeropad",
    how_method="direct",
    how_label=None
):
    """Convolution.

    Parameters
    ----------
    dataarray : xarray.DataArray
        First input to the convolution.
    window : array_like
        Second input to the convolution.
    dim : str
        Dimension along which to convolve the inputs.
    how_boundary : "zeropad" | "periodic"
        Boundary handling.
    how_method : "direct" | "fft"
        Implementation of convolution. FFT-based convolution only works for
        float-type inputs without NaNs.
    how_label : str | None
        Label to append to the name of the variable in the convoluted object, default is nothing

    Returns
    -------
    xarray.DataArray
    """
    xp = array_namespace(dataarray.data, window)

    window = xp.asarray(window)
    assert window.ndim == 1

    method = (how_method, how_boundary)
    if method not in _CONVOLVE_METHODS:
        raise NotImplementedError(method)
    convolved = _CONVOLVE_METHODS[method](dataarray, window, dim)

    convolved = how_label_rename(convolved, how_label=how_label)
    return convolved


def _convolve_dataarray_direct_zeropad(dataarray, window, dim):
    window = window[::-1].copy()  # Reverse kernel to get convolution from dot-product
    k = window.size
    window_dim = f"__convolve_dim_{dim}"
    window_da = xr.DataArray(window, dims=[window_dim])
    zero = dataarray.dtype.type(0)
    rolled = dataarray.rolling({dim: k}, center=True).construct(window_dim, fill_value=zero)
    result = (rolled * window_da).sum(dim=window_dim, skipna=False)
    # Multiplying by the (unnamed) window_da drops .name; restore it
    return result.rename(dataarray.name)


def _convolve_array_fft(signal, window, axis, n):
    if signal.dtype.kind != "f" or window.dtype.kind != "f":
        warnings.warn("fft-based convolution casts inputs to float")
    xp = array_namespace(signal, window)
    window_axis_pad = (xp.newaxis,) * (signal.ndim - axis - 1)
    fft_sig = xp.fft.rfft(signal, axis=axis, n=n)
    fft_win = xp.fft.rfft(window, n=n)[:, *window_axis_pad]  # TODO py311 only
    return xp.fft.irfft(fft_sig * fft_win, axis=axis, n=n)


def _convolve_dataarray_fft_zeropad(dataarray, window, dim):
    nsig = dataarray.sizes[dim]
    nwin = window.size
    nfft = nsig + nwin - 1
    axis = dataarray.get_axis_num(dim)
    convolved = _convolve_array_fft(dataarray.data, window, axis=axis, n=nfft)
    # Consistent with direct implementation and centering convention of xarray
    start = (nwin - 1) // 2
    slicer = [slice(None)] * dataarray.ndim
    slicer[axis] = slice(start, start + nsig)
    return dataarray.copy(data=convolved[tuple(slicer)])


def _convolve_dataarray_fft_periodic(dataarray, window, dim):
    xp = array_namespace(dataarray.data, window)
    nsig = dataarray.sizes[dim]
    start = (window.size - 1) // 2
    axis = dataarray.get_axis_num(dim)
    convolved = _convolve_array_fft(dataarray.data, window, axis=axis, n=nsig)
    convolved = xp.roll(convolved, -start, axis=axis)
    return dataarray.copy(data=convolved)


_CONVOLVE_METHODS = {
    ("direct", "zeropad"): _convolve_dataarray_direct_zeropad,
    ("fft", "zeropad"): _convolve_dataarray_fft_zeropad,
    ("fft", "periodic"): _convolve_dataarray_fft_periodic,
}
