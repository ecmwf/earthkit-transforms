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

"""Temporal Fast Fourier Transform (FFT) transformations for earthkit data objects.

These are thin wrappers around :mod:`earthkit.transforms._fourier` which detect the
time dimension automatically from the metadata of the data object.
"""

import logging
import typing as T

import numpy as np
import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms import _fourier, _tools

logger = logging.getLogger(__name__)


def _add_period_coord(
    result: xr.Dataset | xr.DataArray, freq_dim: str, units: str | None = "s"
) -> xr.Dataset | xr.DataArray:
    """Attach a ``period`` (1 / frequency) convenience coordinate along ``freq_dim``.

    The period is expressed in the reciprocal units of the frequency coordinate (seconds
    when the sample spacing is inferred from datetime coordinates). The zero-frequency term
    has no finite period and is set to NaN.
    """
    if freq_dim not in result.coords:
        return result
    # The period coordinate is derived with numpy to keep it host-side, as an xarray coordinate.
    freqs = np.asarray(result.coords[freq_dim].values, dtype=float)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs != 0.0, 1.0 / freqs, np.nan)
    result = result.assign_coords({"period": (freq_dim, periods)})
    attrs = {"long_name": "period"}
    if units is not None:
        attrs["units"] = units
    result["period"].attrs.update(attrs)
    return result


# ------------------------------------------------------------------------------------------
# One-dimensional transforms
# ------------------------------------------------------------------------------------------
@format_handler()
@_tools.time_dim_decorator
def fft(
    dataarray: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the discrete Fourier Transform of an xarray object along the time dimension.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.fft` which detects
    the time dimension automatically from the metadata of the data object.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the
        calculation. Default behaviour is to deduce the time dimension from the
        attributes of the coordinates, then fall back to ``"time"``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along the time dimension, used to compute the frequency
        coordinate. If not provided it is inferred from the time coordinate values and
        expressed in seconds, so that the frequencies are in Hz.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued Fourier Transform of the input, with the time dimension
        replaced by ``freq_dim``, a frequency coordinate in Hz and a ``period`` coordinate
        (``1 / frequency``, in seconds) for convenience.

    """
    result = _fourier.fft(
        dataarray,
        dim=time_dim,
        freq_dim=freq_dim,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )
    return _add_period_coord(result, freq_dim)


@format_handler()
def ifft(
    dataarray: xr.Dataset | xr.DataArray,
    freq_dim: str = "frequency",
    time_dim: str = "time",
    time_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the inverse Fourier Transform of an xarray object along the frequency dimension.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.ifft` for data in the
    frequency domain, e.g. the output of :func:`fft`.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform, typically the output of :func:`fft`.
    freq_dim : str, optional
        Name of the frequency dimension along which to compute the inverse transform.
        Default is ``"frequency"``.
    time_dim : str, optional
        Name of the time dimension created in the output. Default is ``"time"``.
    time_coord : array-like, optional
        Time coordinate values to assign to ``time_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
        Must match the ``norm`` used for the forward transform.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The inverse Fourier Transform of the input, with ``freq_dim`` replaced by ``time_dim``.

    """
    return _fourier.ifft(
        dataarray,
        dim=freq_dim,
        output_dim=time_dim,
        output_coord=time_coord,
        norm=norm,
        xp=xp,
    )


@format_handler()
@_tools.time_dim_decorator
def rfft(
    dataarray: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the Fourier Transform of a real-valued xarray object along the time dimension.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.rfft` which detects
    the time dimension automatically from the metadata of the data object. Only the non-negative
    frequency terms are returned, so the frequency dimension has length ``n // 2 + 1``.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the
        calculation. Default behaviour is to deduce the time dimension from the
        attributes of the coordinates, then fall back to ``"time"``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along the time dimension, used to compute the frequency
        coordinate. If not provided it is inferred from the time coordinate values and
        expressed in seconds, so that the frequencies are in Hz.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued Fourier Transform of the input, with the time dimension replaced by
        ``freq_dim`` of length ``n // 2 + 1``, a frequency coordinate in Hz and a ``period``
        coordinate (``1 / frequency``, in seconds) for convenience.

    """
    result = _fourier.rfft(
        dataarray,
        dim=time_dim,
        freq_dim=freq_dim,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )
    return _add_period_coord(result, freq_dim)


@format_handler()
def irfft(
    dataarray: xr.Dataset | xr.DataArray,
    freq_dim: str = "frequency",
    time_dim: str = "time",
    time_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the inverse of :func:`rfft`, producing a real-valued signal along the time dimension.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.irfft` for data in the
    frequency domain, e.g. the output of :func:`rfft`.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object to transform, typically the output of :func:`rfft`.
    freq_dim : str, optional
        Name of the frequency dimension along which to compute the inverse transform.
        Default is ``"frequency"``.
    time_dim : str, optional
        Name of the time dimension created in the output. Default is ``"time"``.
    time_coord : array-like, optional
        Time coordinate values to assign to ``time_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
        Must match the ``norm`` used for the forward transform.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued inverse transform of the input, with ``freq_dim`` replaced by ``time_dim``.

    """
    return _fourier.irfft(
        dataarray,
        dim=freq_dim,
        output_dim=time_dim,
        output_coord=time_coord,
        norm=norm,
        xp=xp,
    )


@format_handler()
def hfft(
    dataarray: xr.Dataset | xr.DataArray,
    freq_dim: str = "frequency",
    time_dim: str = "time",
    time_coord: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the FFT of a Hermitian-symmetric signal, producing a real-valued time series.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.hfft`. The input
    represents the non-negative-frequency half of a Hermitian-symmetric signal and the output is
    real-valued.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Complex-valued data object with Hermitian symmetry.
    freq_dim : str, optional
        Name of the frequency dimension along which to compute the transform.
        Default is ``"frequency"``.
    time_dim : str, optional
        Name of the time dimension created in the output. Default is ``"time"``.
    time_coord : array-like, optional
        Time coordinate values to assign to ``time_dim`` in the result.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued transform of the input, with ``freq_dim`` replaced by ``time_dim``.

    """
    return _fourier.hfft(
        dataarray,
        dim=freq_dim,
        output_dim=time_dim,
        output_coord=time_coord,
        norm=norm,
        xp=xp,
    )


@format_handler()
@_tools.time_dim_decorator
def ihfft(
    dataarray: xr.Dataset | xr.DataArray,
    time_dim: str | None = None,
    freq_dim: str = "frequency",
    sample_spacing: float | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the inverse FFT of a Hermitian-symmetric signal along the time dimension.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.ihfft` which detects
    the time dimension automatically from the metadata of the data object. The input is
    real-valued and only the non-negative frequency terms are returned, so the frequency dimension
    has length ``n // 2 + 1``.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the
        calculation. Default behaviour is to deduce the time dimension from the
        attributes of the coordinates, then fall back to ``"time"``.
    freq_dim : str, optional
        Name of the frequency dimension created in the output. Default is ``"frequency"``.
    sample_spacing : float, optional
        Spacing between samples along the time dimension, used to compute the frequency
        coordinate. If not provided it is inferred from the time coordinate values and
        expressed in seconds, so that the frequencies are in Hz.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform of the input, with the time dimension replaced by
        ``freq_dim`` of length ``n // 2 + 1``, a frequency coordinate in Hz and a ``period``
        coordinate (``1 / frequency``, in seconds) for convenience.

    """
    result = _fourier.ihfft(
        dataarray,
        dim=time_dim,
        freq_dim=freq_dim,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )
    return _add_period_coord(result, freq_dim)


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
    """Compute the n-dimensional discrete Fourier Transform, over the time dimension by default.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.fftn`. When ``dims``
    is not provided the time dimension is detected automatically from the metadata of the data
    object and used as the (single) transform dimension.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to transform.
    dims : str or sequence of str, optional
        Dimensions over which to compute the transform. Defaults to the detected time dimension.
    s : sequence of int, optional
        Transform length for each dimension in ``dims``. Defaults to the sizes of ``dims``.
    freq_dims : sequence of str, optional
        Names of the frequency dimensions created in the output. Defaults to
        ``"<dim>_frequency"`` for each transformed dimension.
    sample_spacing : float or sequence of float, optional
        Sample spacing for each dimension, used to build the frequency coordinates. If not
        provided the spacing is inferred per dimension.
    norm : str, optional
        Normalisation mode, one of ``"backward"`` (default), ``"ortho"`` or ``"forward"``.
    xp : module, optional
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension.

    """
    if dims is None:
        dims = _tools.get_dim_key(dataarray, "t")
    return _fourier.fftn(
        dataarray,
        dims=dims,
        s=s,
        freq_dims=freq_dims,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )


def ifftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    output_dims: T.Sequence[str] | None = None,
    output_coords: T.Mapping[str, T.Any] | None = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional inverse discrete Fourier Transform.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.ifftn` for data in
    the frequency domain, e.g. the output of :func:`fftn`.

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
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued inverse transform, with each transformed dimension replaced by an
        output dimension.

    """
    return _fourier.ifftn(
        dataarray,
        dims=dims,
        s=s,
        output_dims=output_dims,
        output_coords=output_coords,
        norm=norm,
        xp=xp,
    )


def rfftn(
    dataarray: xr.Dataset | xr.DataArray,
    dims: str | T.Sequence[str] | None = None,
    s: T.Sequence[int] | None = None,
    freq_dims: T.Sequence[str] | None = None,
    sample_spacing: T.Any = None,
    norm: str = "backward",
    xp: T.Any = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the n-dimensional Fourier Transform of real input, over time by default.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.rfftn`. When ``dims``
    is not provided the time dimension is detected automatically from the metadata of the data
    object and used as the (single) transform dimension. The transform over the last dimension in
    ``dims`` returns only the non-negative frequency terms (length ``n // 2 + 1``).

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Real-valued data object to transform.
    dims : str or sequence of str, optional
        Dimensions over which to compute the transform. Defaults to the detected time dimension.
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
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The complex-valued transform, with each transformed dimension replaced by a frequency
        dimension and the last transformed dimension of length ``n // 2 + 1``.

    """
    if dims is None:
        dims = _tools.get_dim_key(dataarray, "t")
    return _fourier.rfftn(
        dataarray,
        dims=dims,
        s=s,
        freq_dims=freq_dims,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )


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

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.irfftn` for data in
    the frequency domain, e.g. the output of :func:`rfftn`. The output is real-valued.

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
        The array namespace to use. If None, it is inferred from the data object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The real-valued inverse transform, with each transformed dimension replaced by an output
        dimension.

    """
    return _fourier.irfftn(
        dataarray,
        dims=dims,
        s=s,
        output_dims=output_dims,
        output_coords=output_coords,
        norm=norm,
        xp=xp,
    )


# ------------------------------------------------------------------------------------------
# Sample-frequency helpers
# ------------------------------------------------------------------------------------------
def fftfreq(n: int, sample_spacing: float = 1.0, freq_dim: str = "frequency") -> xr.DataArray:
    """Return the discrete Fourier Transform sample frequencies as a DataArray.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.fftfreq`.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    freq_dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is
        ``"frequency"``.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n`` containing the sample frequencies, indexed by ``freq_dim``,
        with a ``period`` coordinate (``1 / frequency``) for convenience.

    """
    result = _fourier.fftfreq(n, sample_spacing=sample_spacing, dim=freq_dim)
    return T.cast(xr.DataArray, _add_period_coord(result, freq_dim, units=None))


def rfftfreq(n: int, sample_spacing: float = 1.0, freq_dim: str = "frequency") -> xr.DataArray:
    """Return the sample frequencies for :func:`rfft`/:func:`irfft` as a DataArray.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.rfftfreq`.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    sample_spacing : float, optional
        Sample spacing (inverse of the sampling rate). Default is ``1.0``.
    freq_dim : str, optional
        Name of the dimension and coordinate of the returned DataArray. Default is
        ``"frequency"``.

    Returns
    -------
    xarray.DataArray
        A 1-D DataArray of length ``n // 2 + 1`` containing the non-negative sample frequencies,
        indexed by ``freq_dim``, with a ``period`` coordinate (``1 / frequency``) for convenience.

    """
    result = _fourier.rfftfreq(n, sample_spacing=sample_spacing, dim=freq_dim)
    return T.cast(xr.DataArray, _add_period_coord(result, freq_dim, units=None))


# ------------------------------------------------------------------------------------------
# Spectrum shifts
# ------------------------------------------------------------------------------------------
def fftshift(
    dataarray: xr.Dataset | xr.DataArray,
    freq_dim: str | T.Sequence[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Shift the zero-frequency component to the centre of the spectrum.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.fftshift`.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to shift, typically a frequency-domain object.
    freq_dim : str or sequence of str, optional
        Dimension(s) over which to shift. Defaults to all dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted data object.

    """
    return _fourier.fftshift(dataarray, dim=freq_dim)


def ifftshift(
    dataarray: xr.Dataset | xr.DataArray,
    freq_dim: str | T.Sequence[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Inverse of :func:`fftshift`.

    This is a convenience wrapper around :func:`earthkit.transforms._fourier.ifftshift`.

    Parameters
    ----------
    dataarray : xarray.Dataset or xarray.DataArray
        Data object to shift.
    freq_dim : str or sequence of str, optional
        Dimension(s) over which to shift. Defaults to all dimensions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The shifted data object.

    """
    return _fourier.ifftshift(dataarray, dim=freq_dim)
