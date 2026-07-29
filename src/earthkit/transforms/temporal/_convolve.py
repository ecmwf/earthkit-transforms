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

from typing import Literal

import xarray as xr
from earthkit.utils.decorators import format_handler

from earthkit.transforms import _tools
from earthkit.transforms._convolve import convolve as _convolve


@format_handler()
def convolve(
    dataarray: xr.Dataset | xr.DataArray,
    window: "array_like",
    *_args,
    time_dim: str | None = None,
    how_boundary: Literal["zeropad"] = "zeropad",
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Convolution along the time dimension.

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
    how_boundary : {"zeropad", "periodic"}, default: "zeropad"
        How the signal is extended where the window overhangs:

        - ``"zeropad"``: the signal is extended with zeros.
    how_method : {"auto", "direct", "fft"}, default: "auto"
        How the convolution is evaluated:

        - ``"auto"``: automatically select a method based on the inputs.
        - ``"direct"``: implementation as a windowed dot product. Preserves the
          input dtype and propagates ``NaN`` locally.
        - ``"fft"``: evaluated in the frequency domain via the convolution
          theorem. Fastest for longer windows. Casts all inputs to float and
          spreads ``NaN`` values across the time axis.
    how_label : str | None
        Label to append to the name of the variable in the convoluted object,
        default is nothing.

    Returns
    -------
    xarray.DataArray | xarray.Dataset
        dataarray convolved with the given window along the time dimension.
    """
    kwargs["dim"] = _tools.get_dim_key(dataarray, "t") if time_dim is None else time_dim
    kwargs["how_boundary"] = how_boundary
    return _convolve(dataarray, window, *_args, **kwargs)
