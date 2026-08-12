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

from typing import TypeVar

import numpy as np
import xarray as xr
from earthkit.utils.decorators import format_handler
from numpy.typing import ArrayLike

from earthkit.transforms import _tools
from earthkit.transforms._convolve import convolve as _convolve

@format_handler()
def convolve(
    dataarray: xr.Dataset | xr.DataArray,
    window: np.typing.ArrayLike,
    time_dim: str | None = None,
    remove_partial_periods: bool = False,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Centred convolution along the time dimension.

    Time series are zero-padded at the boundaries.

    Parameters
    ----------
    dataarray : xarray.DataArray | xarray.Dataset
        First input to the convolution.
    window : array_like
        Second input to the convolution, a 1-D kernel.
    time_dim : str
        Name of the time dimension, or coordinate, in the xarray object,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    how_method : {"auto", "direct", "fft"}, default: "auto"
        How the convolution is evaluated:

        - ``"auto"``: automatically select a method based on the inputs.
        - ``"direct"``: implementation as a windowed dot product. Propagates
          ``NaN`` values locally.
        - ``"fft"``: evaluated in the frequency domain via the convolution
          theorem. Fastest for longer windows. Casts all inputs to float and
          spreads ``NaN`` values across the time axis.
    how_label : str | None, default: None
        Label to append to the name of the variable in the convoluted object,
        default is nothing.
    remove_partial_periods : bool, default: False
        If True, remove time steps affected by padding at the start and end.
    **kwargs
        Keyword arguments passed to :func:`earthkit.transforms.convolve`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset (as provided)
    """
    dim = _tools.get_dim_key(dataarray, "t") if time_dim is None else time_dim
    kwargs["dim"] = dim
    kwargs["how_boundary"] = "zeropad"
    result = _convolve(dataarray, window, **kwargs)
    if remove_partial_periods and (k := np.asarray(window).size) > 1:
        start = k // 2
        end = -((k - 1) // 2) or None  # avoid -0 for k==2
        result = result.isel({dim: slice(start, end)})
    return result
