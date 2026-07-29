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
from typing import Literal

import xarray as xr

from earthkit.transforms._aggregate import how_label_rename
from earthkit.utils.array import array_namespace


def convolve(dataarray: xr.DataArray | xr.Dataset, *_args, **kwargs):
    r"""Convolve an xarray.dataarray or xarray.dataset with a 1-D window along a dimension.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        First input to the convolution.
    window : array_like
        Second input to the convolution, a 1-D kernel.
    dim : str
        Dimension along which to convolve the inputs.
    how_boundary : {"zeropad", "periodic"}, default: "zeropad"
        How the signal is extended where the window overhangs:

        - ``"zeropad"``: the signal is extended with zeros.
        - ``"periodic"``: the signal wraps around.
    how_method : {"auto", "direct", "fft"}, default: "auto"
        How the convolution is evaluated:

        - ``"auto"``: automatically select a method based on the inputs.
        - ``"direct"``: implementation as a windowed dot product. Preserves the
          input dtype and propagates ``NaN`` locally.
        - ``"fft"``: evaluated in the frequency domain via the convolution
          theorem. Fastest for longer windows. Casts all inputs to float and
          spreads ``NaN`` values across the entire convolution axis.
    how_label : str | None
        Label to append to the name of the variable in the convoluted object,
        default is nothing.

    Returns
    -------
    xarray.DataArray
        The result of the convolution.

    Notes
    -----
    For a signal :math:`f` of length :math:`n` and a window :math:`g` of length
    :math:`k`, the output along ``dim`` is the centered discrete convolution

    .. math::

        (f * g)_i = \sum_{j=0}^{k-1} g_j \, f_{i + s - j},
        \qquad s = \left\lfloor \frac{k - 1}{2} \right\rfloor,

    for :math:`i = 0, \dots, n - 1`. Note that signal and window indices run in
    opposite directions to obtain true convolution instead of cross-correlation.
    """
    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _convolve_dataarray(dataarray[var], *_args, **kwargs)
            out_ds[out_da.name] = out_da
        return out_ds
    else:
        return _convolve_dataarray(dataarray, *_args, **kwargs)


def _convolve_dataarray(
    dataarray: xr.DataArray,
    window: "array_like",
    dim: str,
    *,
    how_boundary: Literal["zeropad"] | Literal["periodic"] = "zeropad",
    how_method: Literal["auto"] | Literal["direct"] | Literal["fft"] = "auto",
    how_label: str = None,
):
    r"""Convolve a data array with a 1-D window along a single dimension.

    Parameters
    ----------
    dataarray : xarray.DataArray
        First input to the convolution.
    window : array_like
        Second input to the convolution, a 1-D kernel.
    dim : str
        Dimension along which to convolve the inputs.
    how_boundary : {"zeropad", "periodic"}, default: "zeropad"
        How the signal is extended where the window overhangs.
    how_method : {"auto", "direct", "fft"}, default: "auto"
        How the convolution is evaluated.
    how_label : str | None
        Label to append to the name of the variable in the convoluted object,
        default is nothing.

    Returns
    -------
    xarray.DataArray
        The result of the convolution.
    """
    if dim not in dataarray.dims:
        raise ValueError(f"dim={dim!r} not found in dataarray dimensions {dataarray.dims}")

    xp = array_namespace(dataarray.data)
    window = xp.asarray(window)
    if window.ndim != 1:
        raise ValueError(f"window must be 1-dimensional, got window.ndim={window.ndim}")
    if window.size == 0:
        raise ValueError("window must be non-empty")

    if how_method == "auto":
        # FFT is float-only, so choose direct method when result has any other type
        how_method_proposed = "fft" if dataarray.dtype.kind == "f" or window.dtype.kind == "f" else "direct"
        if (how_method_proposed, how_boundary) not in _CONVOLVE_METHODS:
            raise RuntimeError(
                f"Unable to auto-select a method for input and boundary {how_boundary!r}. "
                "Please select a method and boundary combination explicitly."
            )
        how_method = how_method_proposed

    method = (how_method, how_boundary)
    if method not in _CONVOLVE_METHODS:
        available = ", ".join(f"{m!r} and {b!r}" for m, b in _CONVOLVE_METHODS)
        raise ValueError(
            f"Unsupported combination of method and boundary: {how_method!r} and {how_boundary!r}. "
            f"Available combinations are: {available}"
        )

    convolved = _CONVOLVE_METHODS[method](dataarray, window, dim)
    convolved = how_label_rename(convolved, how_label=how_label)
    return convolved


def _convolve_dataarray_direct_zeropad(dataarray, window, dim):
    """Rolling dot product-based convolution with zero-padding at the boundary."""
    window = window[::-1].copy()  # reverse kernel for true convolution
    k = window.size
    window_dim = f"__convolve_dim_{dim}"
    window_da = xr.DataArray(window, dims=[window_dim])
    return (
        dataarray.rolling({dim: k}, center=True)
        .construct(window_dim, fill_value=dataarray.dtype.type(0))
        .dot(window_da, dim=window_dim)
    )


def _convolve_array_fft(signal, window, axis, n, xp):
    """Generic FFT-based convolution."""
    if signal.dtype.kind != "f" or window.dtype.kind != "f":
        warnings.warn("FFT-based convolution casts inputs to float")
    window_axis_pad = (xp.newaxis,) * (signal.ndim - axis - 1)
    fft_sig = xp.fft.rfft(signal, axis=axis, n=n)
    fft_win = xp.fft.rfft(window, n=n)[(slice(None), *window_axis_pad)]
    return xp.fft.irfft(fft_sig * fft_win, axis=axis, n=n)


def _convolve_dataarray_fft_zeropad(dataarray, window, dim):
    """FFT-based convolution with zero-padding at the boundary."""
    xp = array_namespace(dataarray.data, window)
    nsig = dataarray.sizes[dim]
    nwin = window.size
    nfft = nsig + nwin - 1
    axis = dataarray.get_axis_num(dim)
    convolved = _convolve_array_fft(dataarray.data, window, axis=axis, n=nfft, xp=xp)
    # Consistent with direct implementation and centering convention of xarray
    start = (nwin - 1) // 2
    slicer = [slice(None)] * dataarray.ndim
    slicer[axis] = slice(start, start + nsig)
    return dataarray.copy(data=convolved[tuple(slicer)])


def _convolve_dataarray_fft_periodic(dataarray, window, dim):
    """FFT-based convolution with periodic boundary condition."""
    xp = array_namespace(dataarray.data, window)
    nsig = dataarray.sizes[dim]
    start = (window.size - 1) // 2
    axis = dataarray.get_axis_num(dim)
    convolved = _convolve_array_fft(dataarray.data, window, axis=axis, n=nsig, xp=xp)
    convolved = xp.roll(convolved, -start, axis=axis)
    return dataarray.copy(data=convolved)


_CONVOLVE_METHODS = {
    ("direct", "zeropad"): _convolve_dataarray_direct_zeropad,
    ("fft", "zeropad"): _convolve_dataarray_fft_zeropad,
    ("fft", "periodic"): _convolve_dataarray_fft_periodic,
}
