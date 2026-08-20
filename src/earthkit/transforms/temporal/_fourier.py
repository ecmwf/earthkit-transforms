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

import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms import _fourier, _tools

logger = logging.getLogger(__name__)


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
        replaced by ``freq_dim`` and a frequency coordinate in Hz.

    """
    return _fourier.fft(
        dataarray,
        dim=time_dim,
        freq_dim=freq_dim,
        sample_spacing=sample_spacing,
        norm=norm,
        xp=xp,
    )


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
