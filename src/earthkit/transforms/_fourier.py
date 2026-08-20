# Copyright 2024-, European Centre for Medium Range Weather Forecasts.
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

"""Fast Fourier Transform (FFT) transformations for earthkit data objects.

This module provides ``xarray`` wrappers for every function in the ``fft`` extension of
the Python array API standard
(https://data-apis.org/array-api/latest/extensions/fourier_transform_functions.html):

- one-dimensional transforms: :func:`fft`, :func:`ifft`, :func:`rfft`, :func:`irfft`,
  :func:`hfft`, :func:`ihfft`
- n-dimensional transforms: :func:`fftn`, :func:`ifftn`, :func:`rfftn`, :func:`irfftn`
- sample-frequency helpers: :func:`fftfreq`, :func:`rfftfreq`
- spectrum shifts: :func:`fftshift`, :func:`ifftshift`

The transforms are computed using the ``fft`` extension of the array namespace of the input
data, so they run on the native backend of the data (e.g. NumPy or a GPU-backed array
library) and return ``xarray.DataArray``/``xarray.Dataset`` objects.
"""

import logging
import typing as T

import numpy as np
import xarray as xr
from earthkit.utils.array import array_namespace

logger = logging.getLogger(__name__)

#: Attribute used to record the dimension a frequency coordinate was derived from.
SOURCE_DIM_ATTR = "earthkit_fft_source_dim"


# ------------------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------------------
def _resolve_xp(xp: T.Any, array: T.Any) -> T.Any:
    """Return the explicit array namespace, or infer it from ``array``."""
    return xp if xp is not None else array_namespace(array)


def _infer_sample_spacing(dataarray: xr.Dataset | xr.DataArray, dim: str) -> float:
    """Infer the sample spacing of a dimension from its coordinate values.

    Datetime coordinates are converted to a spacing in seconds. If the coordinate
    is missing or has fewer than two values a spacing of ``1.0`` is returned.
    """
    if dim not in dataarray.coords:
        return 1.0
    coord = np.asarray(dataarray.coords[dim].values)
    if coord.size < 2:
        return 1.0
    if np.issubdtype(coord.dtype, np.datetime64):
        deltas = np.diff(coord) / np.timedelta64(1, "s")
        return float(np.mean(deltas))
    return float(np.mean(np.diff(coord)))


def _ensure_dims(dataarray: xr.Dataset | xr.DataArray, dims: str | T.Sequence[str] | None) -> list[str]:
    """Normalise ``dims`` to a list of dimension names, validating membership."""
    if dims is None:
        resolved = list(dataarray.dims)
    elif isinstance(dims, str):
        resolved = [dims]
    else:
        resolved = list(dims)
    for dim in resolved:
        if dim not in dataarray.dims:
            raise ValueError(f"Dimension '{dim}' not found in data object dimensions: {list(dataarray.dims)}")
    return resolved


def _resolve_per_dim(value: T.Any, dims: list[str]) -> list:
    """Broadcast a scalar/None to a per-dimension list, or validate a provided sequence."""
    if value is None or np.isscalar(value):
        return [value] * len(dims)
    value = list(value)
    if len(value) != len(dims):
        raise ValueError(f"Expected one value per dimension ({len(dims)}), got {len(value)}.")
    return value


def _forward_1d(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    xp_func_name: str,
    freq_func_name: str,
    n: int | None,
    freq_dim: str,
    sample_spacing: float | None,
    norm: str,
    xp: T.Any,
) -> xr.Dataset | xr.DataArray:
    """Apply a forward 1-D transform along ``dim`` producing a frequency dimension."""
    if dim not in dataarray.dims:
        raise ValueError(f"Dimension '{dim}' not found in data object dimensions: {list(dataarray.dims)}")

    size = int(dataarray.sizes[dim])
    n_in = int(n) if n is not None else size
    spacing = sample_spacing if sample_spacing is not None else _infer_sample_spacing(dataarray, dim)
    freqs = getattr(np.fft, freq_func_name)(n_in, d=spacing)
    out_len = int(freqs.shape[0])

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, n=n_in, axis=-1, norm=norm)

    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[[dim]],
        output_core_dims=[[freq_dim]],
        exclude_dims={dim} if freq_dim == dim else set(),
        dask="parallelized",
        output_dtypes=[complex],
        dask_gufunc_kwargs={"output_sizes": {freq_dim: out_len}},
    )
    result = result.assign_coords({freq_dim: freqs})
    result[freq_dim].attrs.update({"long_name": "frequency", SOURCE_DIM_ATTR: dim})
    return result


def _inverse_1d(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    xp_func_name: str,
    n: int | None,
    default_n: T.Callable[[int], int],
    output_dim: str | None,
    output_coord: T.Any,
    norm: str,
    xp: T.Any,
    out_dtype: type,
) -> xr.Dataset | xr.DataArray:
    """Apply an inverse 1-D transform along ``dim`` producing a signal dimension."""
    if dim not in dataarray.dims:
        raise ValueError(f"Dimension '{dim}' not found in data object dimensions: {list(dataarray.dims)}")

    size = int(dataarray.sizes[dim])
    n_out = int(n) if n is not None else default_n(size)
    if output_dim is None:
        output_dim = dataarray.coords[dim].attrs.get(SOURCE_DIM_ATTR, dim) if dim in dataarray.coords else dim

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, n=n_out, axis=-1, norm=norm)

    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[[dim]],
        output_core_dims=[[output_dim]],
        exclude_dims={dim} if output_dim == dim else set(),
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"output_sizes": {output_dim: n_out}},
    )
    if output_coord is not None:
        result = result.assign_coords({output_dim: output_coord})
    return result


def _forward_nd(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None,
    xp_func_name: str,
    real_input: bool,
    s: T.Sequence[int] | None,
    freq_dims: T.Sequence[str] | None,
    sample_spacing: T.Any,
    norm: str,
    xp: T.Any,
) -> xr.Dataset | xr.DataArray:
    """Apply a forward n-D transform over ``dims`` producing frequency dimensions."""
    dims = _ensure_dims(dataarray, dims)
    n_axes = len(dims)
    sizes = [int(dataarray.sizes[d]) for d in dims]
    s_list = [int(v) if v is not None else sizes[i] for i, v in enumerate(_resolve_per_dim(s, dims))]
    spacings = [
        v if v is not None else _infer_sample_spacing(dataarray, dims[i])
        for i, v in enumerate(_resolve_per_dim(sample_spacing, dims))
    ]

    freqs = []
    for i in range(n_axes):
        if real_input and i == n_axes - 1:
            freqs.append(np.fft.rfftfreq(s_list[i], d=spacings[i]))
        else:
            freqs.append(np.fft.fftfreq(s_list[i], d=spacings[i]))
    out_sizes = [int(f.shape[0]) for f in freqs]

    out_dims = [f"{d}_frequency" for d in dims] if freq_dims is None else list(freq_dims)
    axes = tuple(range(-n_axes, 0))

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, s=tuple(s_list), axes=axes, norm=norm)

    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[dims],
        output_core_dims=[out_dims],
        exclude_dims={d for d in dims if d in out_dims},
        dask="parallelized",
        output_dtypes=[complex],
        dask_gufunc_kwargs={"output_sizes": dict(zip(out_dims, out_sizes))},
    )
    for out_dim, freq, src_dim in zip(out_dims, freqs, dims):
        result = result.assign_coords({out_dim: freq})
        result[out_dim].attrs.update({"long_name": "frequency", SOURCE_DIM_ATTR: src_dim})
    return result


def _inverse_nd(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None,
    xp_func_name: str,
    real_output: bool,
    s: T.Sequence[int] | None,
    output_dims: T.Sequence[str] | None,
    output_coords: T.Mapping[str, T.Any] | None,
    norm: str,
    xp: T.Any,
    out_dtype: type,
) -> xr.Dataset | xr.DataArray:
    """Apply an inverse n-D transform over ``dims`` producing signal dimensions."""
    dims = _ensure_dims(dataarray, dims)
    n_axes = len(dims)
    sizes = [int(dataarray.sizes[d]) for d in dims]

    s_values = _resolve_per_dim(s, dims)
    out_sizes = []
    for i in range(n_axes):
        if s_values[i] is not None:
            out_sizes.append(int(s_values[i]))
        elif real_output and i == n_axes - 1:
            out_sizes.append(2 * (sizes[i] - 1))
        else:
            out_sizes.append(sizes[i])

    if output_dims is None:
        out_dims = [dataarray.coords[d].attrs.get(SOURCE_DIM_ATTR, d) if d in dataarray.coords else d for d in dims]
    else:
        out_dims = list(output_dims)
    axes = tuple(range(-n_axes, 0))

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, s=tuple(out_sizes), axes=axes, norm=norm)

    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[dims],
        output_core_dims=[out_dims],
        exclude_dims={d for d in dims if d in out_dims},
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"output_sizes": dict(zip(out_dims, out_sizes))},
    )
    if output_coords is not None:
        result = result.assign_coords({k: v for k, v in output_coords.items() if k in out_dims})
    return result


# ------------------------------------------------------------------------------------------
# One-dimensional transforms
# ------------------------------------------------------------------------------------------
def fft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional discrete Fourier transform along a dimension.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform.
    dim : str
        Name of the dimension along which to compute the transform.
    n : int, optional
        Length of the transformed axis. If larger than the input the axis is zero-padded,
        if smaller it is truncated. Defaults to the size of ``dim``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along ``dim``, used to build the frequency coordinate. If not
        provided it is inferred from the coordinate values of ``dim``; datetime coordinates are
        converted to a spacing in seconds so that the frequencies are expressed in Hz.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim``.

    """
    return _forward_1d(dataarray, dim, "fft", "fftfreq", n, freq_dim, sample_spacing, norm, xp)


def ifft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    output_dim: str | None = None,
    output_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional inverse discrete Fourier transform along a dimension.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform, typically the output of :func:`fft`.
    dim : str
        Name of the (frequency) dimension along which to compute the inverse transform.
    n : int, optional
        Length of the transformed axis. Defaults to the size of ``dim``.
    output_dim : str, optional
        Name of the dimension created in the output. If not provided, the source dimension
        recorded by :func:`fft` is used when available, otherwise ``dim``.
    output_coord : array-like, optional
        Coordinate values to assign to ``output_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
        Must match the ``norm`` used for the forward transform to recover the original data.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued inverse transform, with ``dim`` replaced by ``output_dim``.

    """
    return _inverse_1d(dataarray, dim, "ifft", n, lambda size: size, output_dim, output_coord, norm, xp, complex)


def rfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional discrete Fourier transform of real-valued input.

    Only the non-negative frequency terms are returned, so the frequency dimension has length
    ``n // 2 + 1`` where ``n`` is the transform length.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    dim : str
        Name of the dimension along which to compute the transform.
    n : int, optional
        Number of input points used along ``dim``. Defaults to the size of ``dim``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along ``dim``, used to build the frequency coordinate. If not
        provided it is inferred from the coordinate values of ``dim``.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim`` of length ``n // 2 + 1``.

    """
    return _forward_1d(dataarray, dim, "rfft", "rfftfreq", n, freq_dim, sample_spacing, norm, xp)


def irfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    output_dim: str | None = None,
    output_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional inverse of :func:`rfft` for complex-valued input.

    The output is real-valued. Because the last input point does not carry information about
    the length of the original real signal, the output length defaults to ``2 * (m - 1)`` where
    ``m`` is the size of ``dim``; use ``n`` to specify the exact output length.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object to transform, typically the output of :func:`rfft`.
    dim : str
        Name of the (frequency) dimension along which to compute the inverse transform.
    n : int, optional
        Length of the output signal. Defaults to ``2 * (size - 1)``.
    output_dim : str, optional
        Name of the dimension created in the output. If not provided, the source dimension
        recorded by :func:`rfft` is used when available, otherwise ``dim``.
    output_coord : array-like, optional
        Coordinate values to assign to ``output_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued inverse transform, with ``dim`` replaced by ``output_dim``.

    """
    return _inverse_1d(
        dataarray, dim, "irfft", n, lambda size: 2 * (size - 1), output_dim, output_coord, norm, xp, float
    )


def hfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    output_dim: str | None = None,
    output_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional FFT of a signal that has Hermitian symmetry.

    The input represents the non-negative-frequency half of a Hermitian-symmetric signal and
    the output is real-valued. The output length defaults to ``2 * (m - 1)`` where ``m`` is the
    size of ``dim``; use ``n`` to specify the exact output length.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object with Hermitian symmetry.
    dim : str
        Name of the dimension along which to compute the transform.
    n : int, optional
        Length of the output. Defaults to ``2 * (size - 1)``.
    output_dim : str, optional
        Name of the dimension created in the output. If not provided, the source dimension
        recorded by a preceding transform is used when available, otherwise ``dim``.
    output_coord : array-like, optional
        Coordinate values to assign to ``output_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued transform, with ``dim`` replaced by ``output_dim``.

    """
    return _inverse_1d(
        dataarray, dim, "hfft", n, lambda size: 2 * (size - 1), output_dim, output_coord, norm, xp, float
    )


def ihfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the one-dimensional inverse FFT of a signal that has Hermitian symmetry.

    The input is real-valued and the complex output contains only the non-negative frequency
    terms, so the frequency dimension has length ``n // 2 + 1`` where ``n`` is the transform length.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    dim : str
        Name of the dimension along which to compute the transform.
    n : int, optional
        Number of input points used along ``dim``. Defaults to the size of ``dim``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along ``dim``, used to build the frequency coordinate. If not
        provided it is inferred from the coordinate values of ``dim``.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim`` of length ``n // 2 + 1``.

    """
    return _forward_1d(dataarray, dim, "ihfft", "rfftfreq", n, freq_dim, sample_spacing, norm, xp)


# ------------------------------------------------------------------------------------------
# N-dimensional transforms
# ------------------------------------------------------------------------------------------
def fftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    freq_dims: T.Sequence[str] | None = None,
    sample_spacing: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional discrete Fourier transform over several dimensions.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform.
    dims : str or sequence of str, optional
        Dimensions over which to compute the transform. Defaults to all dimensions.
    s : sequence of int, optional
        Transform length for each dimension in ``dims``. Defaults to the sizes of ``dims``.
    freq_dims : sequence of str, optional
        Names of the frequency dimensions created in the output. Defaults to
        ``"<dim>_frequency"`` for each transformed dimension.
    sample_spacing : float or sequence of float, optional
        Sample spacing for each dimension, used to build the frequency coordinates. A scalar is
        applied to all dimensions. If not provided the spacing is inferred per dimension.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension.

    """
    return _forward_nd(dataarray, dims, "fftn", False, s, freq_dims, sample_spacing, norm, xp)


def ifftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    output_dims: T.Sequence[str] | None = None,
    output_coords: T.Mapping[str, T.Any] | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional inverse discrete Fourier transform over several dimensions.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform, typically the output of :func:`fftn`.
    dims : str or sequence of str, optional
        (Frequency) dimensions over which to compute the inverse transform. Defaults to all
        dimensions.
    s : sequence of int, optional
        Output length for each dimension in ``dims``. Defaults to the sizes of ``dims``.
    output_dims : sequence of str, optional
        Names of the dimensions created in the output. If not provided, the source dimensions
        recorded by :func:`fftn` are used when available, otherwise the input dimension names.
    output_coords : mapping, optional
        Mapping of output dimension name to coordinate values to assign in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
        Must match the ``norm`` used for the forward transform.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued inverse transform, with each transformed dimension replaced by an
        output dimension.

    """
    return _inverse_nd(dataarray, dims, "ifftn", False, s, output_dims, output_coords, norm, xp, complex)


def rfftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    freq_dims: T.Sequence[str] | None = None,
    sample_spacing: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional discrete Fourier transform of real-valued input.

    The transform over the last dimension in ``dims`` returns only the non-negative frequency
    terms (length ``n // 2 + 1``); the remaining dimensions are transformed as by :func:`fftn`.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    dims : str or sequence of str, optional
        Dimensions over which to compute the transform. Defaults to all dimensions.
    s : sequence of int, optional
        Transform length for each dimension in ``dims``. Defaults to the sizes of ``dims``.
    freq_dims : sequence of str, optional
        Names of the frequency dimensions created in the output. Defaults to
        ``"<dim>_frequency"`` for each transformed dimension.
    sample_spacing : float or sequence of float, optional
        Sample spacing for each dimension, used to build the frequency coordinates.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension and the last transformed dimension of length ``n // 2 + 1``.

    """
    return _forward_nd(dataarray, dims, "rfftn", True, s, freq_dims, sample_spacing, norm, xp)


def irfftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    output_dims: T.Sequence[str] | None = None,
    output_coords: T.Mapping[str, T.Any] | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional inverse of :func:`rfftn` for complex-valued input.

    The output is real-valued. The length of the last transformed dimension defaults to
    ``2 * (m - 1)`` where ``m`` is its input size; use ``s`` to specify exact output lengths.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object to transform, typically the output of :func:`rfftn`.
    dims : str or sequence of str, optional
        (Frequency) dimensions over which to compute the inverse transform. Defaults to all
        dimensions.
    s : sequence of int, optional
        Output length for each dimension in ``dims``. Defaults to the input sizes, except the
        last transformed dimension which defaults to ``2 * (size - 1)``.
    output_dims : sequence of str, optional
        Names of the dimensions created in the output. If not provided, the source dimensions
        recorded by :func:`rfftn` are used when available, otherwise the input dimension names.
    output_coords : mapping, optional
        Mapping of output dimension name to coordinate values to assign in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued inverse transform, with each transformed dimension replaced by an output
        dimension.

    """
    return _inverse_nd(dataarray, dims, "irfftn", True, s, output_dims, output_coords, norm, xp, float)


# ------------------------------------------------------------------------------------------
# Sample-frequency helpers
# ------------------------------------------------------------------------------------------
def fftfreq(n: int, sample_spacing: float = 1.0, dim: str = "frequency") -> xr.DataArray:
    """Return the discrete Fourier transform sample frequencies as a DataArray.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is ``"frequency"``.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n`` containing the sample frequencies, indexed by ``dim``.

    """
    freqs = np.fft.fftfreq(int(n), d=sample_spacing)
    return xr.DataArray(freqs, dims=[dim], coords={dim: freqs}, name=dim, attrs={"long_name": "frequency"})


def rfftfreq(n: int, sample_spacing: float = 1.0, dim: str = "frequency") -> xr.DataArray:
    """Return the sample frequencies for :func:`rfft`/:func:`irfft` as a DataArray.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is ``"frequency"``.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n // 2 + 1`` containing the non-negative sample frequencies,
        indexed by ``dim``.

    """
    freqs = np.fft.rfftfreq(int(n), d=sample_spacing)
    return xr.DataArray(freqs, dims=[dim], coords={dim: freqs}, name=dim, attrs={"long_name": "frequency"})


# ------------------------------------------------------------------------------------------
# Spectrum shifts
# ------------------------------------------------------------------------------------------
def fftshift(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str | T.Sequence[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Shift the zero-frequency component to the centre of the spectrum.

    The data and the associated coordinate(s) are reordered along the requested dimension(s).

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to shift, typically a frequency-domain object.
    dim : str or sequence of str, optional
        Dimension(s) over which to shift. Defaults to all dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted data object.

    """
    dims = _ensure_dims(dataarray, dim)
    indexers = {d: np.fft.fftshift(np.arange(int(dataarray.sizes[d]))) for d in dims}
    return dataarray.isel(indexers)


def ifftshift(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str | T.Sequence[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Inverse of :func:`fftshift`.

    The data and the associated coordinate(s) are reordered along the requested dimension(s).

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to shift.
    dim : str or sequence of str, optional
        Dimension(s) over which to shift. Defaults to all dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted data object.

    """
    dims = _ensure_dims(dataarray, dim)
    indexers = {d: np.fft.ifftshift(np.arange(int(dataarray.sizes[d]))) for d in dims}
    return dataarray.isel(indexers)
