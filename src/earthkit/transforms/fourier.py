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

The frequency coordinate is built from the sample spacing of the transformed dimension. That
spacing is in seconds — and the frequencies therefore in Hz — only when it was inferred from a
datetime coordinate (``numpy.datetime64`` or ``cftime``). For a numeric coordinate, or an
explicit ``sample_spacing``, the frequencies are in the reciprocal of whatever units that
coordinate or value is in, and the coordinate is left unlabelled unless
``sample_spacing_units`` says otherwise.

Each frequency coordinate also records the dimension and the transform length it was derived
from, so that the inverse transforms can restore both without being told. This is what allows
:func:`irfft` and :func:`hfft` to recover a signal of odd length, which is otherwise ambiguous.
"""

import logging
import typing as T

import numpy as np
import xarray as xr
from earthkit.utils.array import array_namespace

logger = logging.getLogger(__name__)

#: Attribute used to record the dimension a frequency coordinate was derived from.
SOURCE_DIM_ATTR = "earthkit_fft_source_dim"

#: Attribute used to record the transform length a frequency coordinate was derived from.
#: The inverse transforms use it to restore the original signal length exactly, which is
#: otherwise ambiguous for the real and Hermitian transforms (a spectrum of length ``m``
#: corresponds to a signal of length ``2 * m - 2`` or ``2 * m - 1``).
SOURCE_SIZE_ATTR = "earthkit_fft_source_size"

#: Frequency unit to advertise for each known sample-spacing unit. Any other spacing unit
#: is expressed as its reciprocal, e.g. ``"day"`` gives a frequency in ``"day-1"``.
_KNOWN_FREQUENCY_UNITS = {"s": "Hz"}


# ------------------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------------------
def _resolve_xp(xp: T.Any, array: T.Any) -> T.Any:
    """Return the explicit array namespace, or infer it from ``array``."""
    return xp if xp is not None else array_namespace(array)


def _infer_sample_spacing(dataarray: xr.Dataset | xr.DataArray, dim: str) -> tuple[float, str | None]:
    """Infer the sample spacing of a dimension from its coordinate values.

    Datetime coordinates are converted to a spacing in seconds; this covers both
    ``numpy.datetime64`` coordinates and ``cftime`` coordinates, i.e. those using a
    non-standard calendar such as ``360_day`` or ``noleap``. If the coordinate
    is missing or has fewer than two values a spacing of ``1.0`` is returned. A warning
    is emitted if the coordinate is not regularly spaced, since the FFT assumes uniform
    sampling and the returned (mean) spacing then yields an approximate frequency axis.

    Returns
    -------
    tuple of (float, str or None)
        The mean sample spacing and its units. The units are ``"s"`` when the spacing was
        derived from a datetime coordinate and ``None`` when it is in the (unknown) units of
        a numeric coordinate, so that callers only claim a frequency unit when they know one.

    """
    if dim not in dataarray.coords:
        return 1.0, None
    coord = np.asarray(dataarray.coords[dim].values)
    if coord.size < 2:
        return 1.0, None
    deltas = np.diff(coord)
    units: str | None = None
    if np.issubdtype(deltas.dtype, np.timedelta64):
        # numpy.datetime64 coordinates difference to timedelta64
        deltas = deltas / np.timedelta64(1, "s")
        units = "s"
    elif deltas.dtype == object and all(hasattr(delta, "total_seconds") for delta in deltas):
        # cftime (and python datetime) coordinates difference to datetime.timedelta
        deltas = np.array([delta.total_seconds() for delta in deltas], dtype=float)
        units = "s"
    mean_delta = float(np.mean(deltas))
    if mean_delta != 0.0 and not np.allclose(deltas, mean_delta, rtol=1e-3):
        # Naming the observed spread keeps the warning actionable: unequal calendar months,
        # for instance, are expected to vary by a few percent, whereas a missing time step is not.
        logger.warning(
            "Coordinate '%s' is not regularly spaced (spacing varies between %g and %g, mean %g); "
            "the FFT assumes uniform sampling, so the frequency coordinate derived from the mean "
            "spacing is only approximate.",
            dim,
            float(np.min(deltas)),
            float(np.max(deltas)),
            mean_delta,
        )
    if mean_delta == 0.0:
        raise ValueError(
            f"Coordinate '{dim}' has a sample spacing of zero, so no frequency coordinate can be "
            "derived from it. Provide an explicit 'sample_spacing' if the coordinate values are "
            "not meaningful."
        )
    if mean_delta < 0.0:
        logger.warning(
            "Coordinate '%s' is descending; using the absolute sample spacing (%g) so that the "
            "frequency coordinate is positive.",
            dim,
            abs(mean_delta),
        )
        mean_delta = abs(mean_delta)
    return mean_delta, units


def _frequency_units(spacing_units: str | None) -> str | None:
    """Return the frequency unit corresponding to a sample-spacing unit, if one is known."""
    if not spacing_units:
        return None
    return _KNOWN_FREQUENCY_UNITS.get(spacing_units, f"{spacing_units}-1")


def _resolve_spacing(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    sample_spacing: float | None,
    sample_spacing_units: str | None,
) -> tuple[float, str | None]:
    """Return the sample spacing to use along ``dim``, and the units it is expressed in."""
    if sample_spacing is not None:
        return float(sample_spacing), sample_spacing_units
    return _infer_sample_spacing(dataarray, dim)


def _frequency_attrs(source_dim: str | None, source_size: int, spacing_units: str | None) -> dict:
    """Build the attributes recorded on a frequency coordinate.

    ``source_dim`` is ``None`` for a standalone frequency axis built by :func:`fftfreq` or
    :func:`rfftfreq`, which has a window length but no dimension it was derived from.
    """
    attrs: dict = {"long_name": "frequency", SOURCE_SIZE_ATTR: int(source_size)}
    if source_dim is not None:
        attrs[SOURCE_DIM_ATTR] = source_dim
    units = _frequency_units(spacing_units)
    if units is not None:
        attrs["units"] = units
    return attrs


def _recorded_source_dim(dataarray: xr.Dataset | xr.DataArray, dim: str) -> str | None:
    """Return the dimension a frequency coordinate was derived from, if it was recorded."""
    if dim not in dataarray.coords:
        return None
    source_dim = dataarray.coords[dim].attrs.get(SOURCE_DIM_ATTR)
    return str(source_dim) if source_dim is not None else None


def _recorded_source_size(
    dataarray: xr.Dataset | xr.DataArray, dim: str, spectrum_len: T.Callable[[int], int], size: int
) -> int | None:
    """Return the transform length recorded on a frequency coordinate, if it is still valid.

    The recorded length is only trusted when it is consistent with the current size of the
    frequency dimension, i.e. when ``spectrum_len(recorded) == size``. A spectrum that has
    been sliced or filtered keeps the (now stale) attribute, and falling back to the generic
    default is safer than silently returning an output of the pre-slicing length.
    """
    if dim not in dataarray.coords:
        return None
    recorded = dataarray.coords[dim].attrs.get(SOURCE_SIZE_ATTR)
    if recorded is None:
        return None
    try:
        recorded = int(recorded)
    except (TypeError, ValueError):
        return None
    return recorded if recorded > 0 and spectrum_len(recorded) == size else None


def _input_dtype(dataarray: xr.Dataset | xr.DataArray, dims: T.Sequence[str]) -> np.dtype:
    """Return the dtype the transform will see, promoting across the variables of a Dataset.

    Only variables carrying every transform dimension are considered; the rest are passed
    through untouched and must not influence the output dtype.
    """
    if isinstance(dataarray, xr.Dataset):
        dtypes = [v.dtype for v in dataarray.data_vars.values() if all(d in v.dims for d in dims)]
        if not dtypes:
            return np.dtype(float)
        try:
            return np.result_type(*dtypes)
        except TypeError:
            return np.dtype(float)
    return dataarray.dtype


def _complex_dtype(dataarray: xr.Dataset | xr.DataArray, dims: T.Sequence[str]) -> np.dtype:
    """Return the complex dtype a transform of ``dataarray`` produces, preserving precision."""
    try:
        return np.result_type(_input_dtype(dataarray, dims), np.complex64)
    except TypeError:
        return np.dtype(complex)


def _real_dtype(dataarray: xr.Dataset | xr.DataArray, dims: T.Sequence[str]) -> np.dtype:
    """Return the real dtype a real-output transform of ``dataarray`` produces."""
    return np.zeros((), dtype=_complex_dtype(dataarray, dims)).real.dtype


def _ensure_dims(dataarray: xr.Dataset | xr.DataArray, dims: str | T.Sequence[str] | None) -> list[str]:
    """Normalise ``dims`` to a list of dimension names, validating membership."""
    if dims is None:
        resolved = [str(d) for d in dataarray.dims]
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
    sample_spacing_units: str | None,
    norm: str,
    xp: T.Any,
) -> xr.Dataset | xr.DataArray:
    """Apply a forward 1-D transform along ``dim`` producing a frequency dimension."""
    if dim not in dataarray.dims:
        raise ValueError(f"Dimension '{dim}' not found in data object dimensions: {list(dataarray.dims)}")

    size = int(dataarray.sizes[dim])
    n_in = int(n) if n is not None else size
    spacing, spacing_units = _resolve_spacing(dataarray, dim, sample_spacing, sample_spacing_units)
    # Frequency coordinates are built with numpy to keep them host-side, as xarray coordinates.
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
        output_dtypes=[_complex_dtype(dataarray, [dim])],
        dask_gufunc_kwargs={"output_sizes": {freq_dim: out_len}},
        on_missing_core_dim="copy",
    )
    result = result.assign_coords({freq_dim: freqs})
    result[freq_dim].attrs.update(_frequency_attrs(dim, n_in, spacing_units))
    return result


def _inverse_1d(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    xp_func_name: str,
    n: int | None,
    default_n: T.Callable[[int], int],
    spectrum_len: T.Callable[[int], int],
    output_dim: str | None,
    output_coord: T.Any,
    norm: str,
    xp: T.Any,
    real_output: bool,
) -> xr.Dataset | xr.DataArray:
    """Apply an inverse 1-D transform along ``dim`` producing a signal dimension."""
    if dim not in dataarray.dims:
        raise ValueError(f"Dimension '{dim}' not found in data object dimensions: {list(dataarray.dims)}")

    size = int(dataarray.sizes[dim])
    if n is not None:
        n_out = int(n)
    else:
        # Prefer the transform length recorded by the forward transform, which resolves the
        # even/odd ambiguity of the real and Hermitian transforms, and fall back to the
        # generic default for spectra that carry no provenance.
        recorded = _recorded_source_size(dataarray, dim, spectrum_len, size)
        n_out = recorded if recorded is not None else default_n(size)
    if output_dim is None:
        output_dim = _recorded_source_dim(dataarray, dim) or dim

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, n=n_out, axis=-1, norm=norm)

    out_dtype = _real_dtype(dataarray, [dim]) if real_output else _complex_dtype(dataarray, [dim])
    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[[dim]],
        output_core_dims=[[output_dim]],
        exclude_dims={dim} if output_dim == dim else set(),
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"output_sizes": {output_dim: n_out}},
        on_missing_core_dim="copy",
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
    sample_spacing_units: T.Any,
    norm: str,
    xp: T.Any,
) -> xr.Dataset | xr.DataArray:
    """Apply a forward n-D transform over ``dims`` producing frequency dimensions."""
    dims = _ensure_dims(dataarray, dims)
    n_axes = len(dims)
    sizes = [int(dataarray.sizes[d]) for d in dims]
    s_list = [int(v) if v is not None else sizes[i] for i, v in enumerate(_resolve_per_dim(s, dims))]
    spacing_values = _resolve_per_dim(sample_spacing, dims)
    spacing_unit_values = _resolve_per_dim(sample_spacing_units, dims)
    resolved = [_resolve_spacing(dataarray, dims[i], spacing_values[i], spacing_unit_values[i]) for i in range(n_axes)]
    spacings = [spacing for spacing, _ in resolved]
    spacing_units = [units for _, units in resolved]

    freqs = []
    for i in range(n_axes):
        # Frequency coordinates are built with numpy to keep them host-side, as xarray coordinates.
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
        output_dtypes=[_complex_dtype(dataarray, dims)],
        dask_gufunc_kwargs={"output_sizes": dict(zip(out_dims, out_sizes))},
        on_missing_core_dim="copy",
    )
    for out_dim, freq, src_dim, src_size, units in zip(out_dims, freqs, dims, s_list, spacing_units):
        result = result.assign_coords({out_dim: freq})
        result[out_dim].attrs.update(_frequency_attrs(src_dim, src_size, units))
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
) -> xr.Dataset | xr.DataArray:
    """Apply an inverse n-D transform over ``dims`` producing signal dimensions."""
    dims = _ensure_dims(dataarray, dims)
    n_axes = len(dims)
    sizes = [int(dataarray.sizes[d]) for d in dims]

    s_values = _resolve_per_dim(s, dims)
    out_sizes = []
    for i in range(n_axes):
        halved = real_output and i == n_axes - 1
        # The last axis of a real transform is halved, the others keep their length.
        spectrum_len = (lambda n: n // 2 + 1) if halved else (lambda n: n)
        recorded = _recorded_source_size(dataarray, dims[i], spectrum_len, sizes[i])
        if s_values[i] is not None:
            out_sizes.append(int(s_values[i]))
        elif recorded is not None:
            out_sizes.append(recorded)
        elif halved:
            out_sizes.append(2 * (sizes[i] - 1))
        else:
            out_sizes.append(sizes[i])

    if output_dims is None:
        out_dims = [_recorded_source_dim(dataarray, d) or d for d in dims]
    else:
        out_dims = list(output_dims)
    axes = tuple(range(-n_axes, 0))

    def _apply(array):
        return getattr(_resolve_xp(xp, array).fft, xp_func_name)(array, s=tuple(out_sizes), axes=axes, norm=norm)

    out_dtype = _real_dtype(dataarray, dims) if real_output else _complex_dtype(dataarray, dims)
    result = xr.apply_ufunc(
        _apply,
        dataarray,
        input_core_dims=[dims],
        output_core_dims=[out_dims],
        exclude_dims={d for d in dims if d in out_dims},
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"output_sizes": dict(zip(out_dims, out_sizes))},
        on_missing_core_dim="copy",
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
    sample_spacing_units: str | None = None,
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
        provided it is inferred from the coordinate values of ``dim``; datetime coordinates
        (including ``cftime`` calendars) are converted to a spacing in seconds, so that the
        frequencies are expressed in Hz. For a numeric coordinate the spacing is in the units
        of the coordinate itself, whatever those are.
    sample_spacing_units : str, optional
        Units of ``sample_spacing``, used to label the frequency coordinate. Only meaningful
        alongside an explicit ``sample_spacing``, since an inferred spacing carries its own
        units. ``"s"`` labels the frequency coordinate in Hz and any other unit ``u`` labels it
        ``"u-1"``. If not provided the frequency coordinate is left unlabelled.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim``. The frequency
        coordinate records the source dimension and transform length, so that :func:`ifft`
        can restore both without being told.

    Notes
    -----
    Variables of a Dataset that do not have ``dim`` are passed through untransformed. The
    transformed dimension becomes the last dimension of the result.

    """
    return _forward_1d(dataarray, dim, "fft", "fftfreq", n, freq_dim, sample_spacing, sample_spacing_units, norm, xp)


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
        Length of the transformed axis. Defaults to the transform length recorded by
        :func:`fft` when the spectrum still carries it, otherwise to the size of ``dim``.
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
    return _inverse_1d(
        dataarray,
        dim,
        "ifft",
        n,
        lambda size: size,
        lambda length: length,
        output_dim,
        output_coord,
        norm,
        xp,
        real_output=False,
    )


def rfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    sample_spacing_units: str | None = None,
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
        provided it is inferred from the coordinate values of ``dim``; datetime coordinates
        (including ``cftime`` calendars) give a spacing in seconds.
    sample_spacing_units : str, optional
        Units of ``sample_spacing``, used to label the frequency coordinate. See :func:`fft`.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim`` of length ``n // 2 + 1``.
        The frequency coordinate records the transform length, so that :func:`irfft` recovers the
        original signal length even when it is odd.

    """
    return _forward_1d(dataarray, dim, "rfft", "rfftfreq", n, freq_dim, sample_spacing, sample_spacing_units, norm, xp)


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

    The output is real-valued. A spectrum of length ``m`` corresponds to a real signal of
    length ``2 * m - 2`` or ``2 * m - 1``, so the output length is ambiguous in general. A
    spectrum produced by :func:`rfft` carries the transform length it was built from and is
    inverted exactly; otherwise the output length defaults to ``2 * (m - 1)``, and ``n`` can be
    used to specify it.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object to transform, typically the output of :func:`rfft`.
    dim : str
        Name of the (frequency) dimension along which to compute the inverse transform.
    n : int, optional
        Length of the output signal. Defaults to the transform length recorded by :func:`rfft`
        when the spectrum still carries it, otherwise to ``2 * (size - 1)``. The recorded
        length is ignored if it is no longer consistent with the size of ``dim``, e.g. after
        the spectrum has been sliced.
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
        dataarray,
        dim,
        "irfft",
        n,
        lambda size: 2 * (size - 1),
        lambda length: length // 2 + 1,
        output_dim,
        output_coord,
        norm,
        xp,
        real_output=True,
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
    the output is real-valued. As for :func:`irfft` the output length is ambiguous: input
    produced by :func:`ihfft` carries the transform length it was built from and is inverted
    exactly, otherwise the output length defaults to ``2 * (m - 1)`` where ``m`` is the size of
    ``dim``, and ``n`` can be used to specify it.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object with Hermitian symmetry.
    dim : str
        Name of the dimension along which to compute the transform.
    n : int, optional
        Length of the output. Defaults to the transform length recorded by :func:`ihfft` when
        the input still carries it, otherwise to ``2 * (size - 1)``.
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
        dataarray,
        dim,
        "hfft",
        n,
        lambda size: 2 * (size - 1),
        lambda length: length // 2 + 1,
        output_dim,
        output_coord,
        norm,
        xp,
        real_output=True,
    )


def ihfft(
    dataarray: xr.Dataset | xr.DataArray,
    dim: str,
    n: int | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    sample_spacing_units: str | None = None,
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
        provided it is inferred from the coordinate values of ``dim``; datetime coordinates
        (including ``cftime`` calendars) give a spacing in seconds.
    sample_spacing_units : str, optional
        Units of ``sample_spacing``, used to label the frequency coordinate. See :func:`fft`.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with ``dim`` replaced by ``freq_dim`` of length ``n // 2 + 1``.
        The frequency coordinate records the transform length, so that :func:`hfft` recovers the
        original signal length even when it is odd.

    """
    return _forward_1d(dataarray, dim, "ihfft", "rfftfreq", n, freq_dim, sample_spacing, sample_spacing_units, norm, xp)


# ------------------------------------------------------------------------------------------
# N-dimensional transforms
# ------------------------------------------------------------------------------------------
def fftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    freq_dims: T.Sequence[str] | None = None,
    sample_spacing: T.Any = None,
    sample_spacing_units: T.Any = None,
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
    sample_spacing_units : str or sequence of str, optional
        Units of ``sample_spacing``, used to label the frequency coordinates. A scalar is
        applied to all dimensions. See :func:`fft`.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension. Each frequency coordinate records its source dimension and transform length,
        so that :func:`ifftn` can restore both without being told.

    """
    return _forward_nd(dataarray, dims, "fftn", False, s, freq_dims, sample_spacing, sample_spacing_units, norm, xp)


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
        Output length for each dimension in ``dims``. Defaults to the transform lengths recorded
        by :func:`fftn` when the spectrum still carries them, otherwise to the sizes of ``dims``.
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
    return _inverse_nd(dataarray, dims, "ifftn", False, s, output_dims, output_coords, norm, xp)


def rfftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    freq_dims: T.Sequence[str] | None = None,
    sample_spacing: T.Any = None,
    sample_spacing_units: T.Any = None,
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
    sample_spacing_units : str or sequence of str, optional
        Units of ``sample_spacing``, used to label the frequency coordinates. A scalar is
        applied to all dimensions. See :func:`fft`.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from each array as it is transformed.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension and the last transformed dimension of length ``n // 2 + 1``. Each frequency
        coordinate records its transform length, so that :func:`irfftn` recovers the original
        lengths even when the last one is odd.

    """
    return _forward_nd(dataarray, dims, "rfftn", True, s, freq_dims, sample_spacing, sample_spacing_units, norm, xp)


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

    The output is real-valued. A spectrum produced by :func:`rfftn` carries the transform
    lengths it was built from and is inverted exactly; otherwise the length of the last
    transformed dimension defaults to ``2 * (m - 1)`` where ``m`` is its input size, and ``s``
    can be used to specify exact output lengths.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object to transform, typically the output of :func:`rfftn`.
    dims : str or sequence of str, optional
        (Frequency) dimensions over which to compute the inverse transform. Defaults to all
        dimensions.
    s : sequence of int, optional
        Output length for each dimension in ``dims``. Defaults to the transform lengths recorded
        by :func:`rfftn` when the spectrum still carries them, otherwise to the input sizes,
        except the last transformed dimension which defaults to ``2 * (size - 1)``.
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
    return _inverse_nd(dataarray, dims, "irfftn", True, s, output_dims, output_coords, norm, xp)


# ------------------------------------------------------------------------------------------
# Sample-frequency helpers
# ------------------------------------------------------------------------------------------
def fftfreq(
    n: int,
    sample_spacing: float = 1.0,
    dim: str = "frequency",
    sample_spacing_units: str | None = None,
) -> xr.DataArray:
    """Return the discrete Fourier transform sample frequencies as a DataArray.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is ``"frequency"``.
    sample_spacing_units : str, optional
        Units of ``sample_spacing``, used to label the frequency coordinate. See :func:`fft`.
        If not provided the frequency coordinate is left unlabelled.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n`` containing the sample frequencies, indexed by ``dim``.

    """
    # Frequency coordinates are built with numpy to keep them host-side, as xarray coordinates.
    freqs = np.fft.fftfreq(int(n), d=sample_spacing)
    attrs = _frequency_attrs(None, int(n), sample_spacing_units)
    result = xr.DataArray(freqs, dims=[dim], coords={dim: freqs}, name=dim, attrs=attrs)
    # The coordinate carries the attributes too, so that it is indistinguishable from the
    # frequency coordinate the transforms produce.
    result[dim].attrs.update(attrs)
    return result


def rfftfreq(
    n: int,
    sample_spacing: float = 1.0,
    dim: str = "frequency",
    sample_spacing_units: str | None = None,
) -> xr.DataArray:
    """Return the sample frequencies for :func:`rfft`/:func:`irfft` as a DataArray.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is ``"frequency"``.
    sample_spacing_units : str, optional
        Units of ``sample_spacing``, used to label the frequency coordinate. See :func:`fft`.
        If not provided the frequency coordinate is left unlabelled.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n // 2 + 1`` containing the non-negative sample frequencies,
        indexed by ``dim``.

    """
    # Frequency coordinates are built with numpy to keep them host-side, as xarray coordinates.
    freqs = np.fft.rfftfreq(int(n), d=sample_spacing)
    attrs = _frequency_attrs(None, int(n), sample_spacing_units)
    result = xr.DataArray(freqs, dims=[dim], coords={dim: freqs}, name=dim, attrs=attrs)
    # The coordinate carries the attributes too, so that it is indistinguishable from the
    # frequency coordinate the transforms produce.
    result[dim].attrs.update(attrs)
    return result


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
    # Integer indexers are built with numpy and applied host-side via .isel(); the backend gathers the data.
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
    # Integer indexers are built with numpy and applied host-side via .isel(); the backend gathers the data.
    indexers = {d: np.fft.ifftshift(np.arange(int(dataarray.sizes[d]))) for d in dims}
    return dataarray.isel(indexers)
