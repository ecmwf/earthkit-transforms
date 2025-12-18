# Copyright 2025, European Centre for Medium Range Weather Forecasts.
#
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
import typing as T

import pandas as pd
import xarray as xr

from earthkit.transforms import _tools

@_tools.time_dim_decorator
@_tools.transform_inputs_decorator()
def accumulation_to_rate(
    dataarray: xr.DataArray,
    step: None | float | int = None,
    step_units: str = "hours",
    rate_units: str = "seconds",
    xp: T.Any = None,
    time_dim: str | None = None,
    accumulation_type: str = "start_of_step",
    from_first_step: bool = True,
    provenance: bool = True,
):
    """Convert a variable accumulated from the beginning of the forecast to a rate.

    The rate is computed by considering first-order discrete differences in data
    along the time (or leadtime, if time is not a dimension)
    axis, divided by the number of seconds in the step.

    # Check it does this
    If time[k] - time[k-1] != step, then the corresponding output values will be NaN.

    Example:
        Compute snowfall rate in m/s based on accumulated snowfall, using a time step of 24 hours.

        >>> snow_fall_rate = accumulation_to_rate(snow_fall, step=24)

    Parameters
    ----------
    dataarray : xr.Dataset or xr.DataArray
        Data accumulated along time to be converted into rate (per second).
    step : float | int, optional
        Interval between consecutive time steps in hours. If not provided, the function
        will infer the step from the first two time steps in the data.
    step_units : str, optional
        Units of the `step` parameter (if provided), default is 'hours'.
    rate_units : str, optional
        Units for the output rate, it can be any valid pandas time frequency string
        (e.g., '15min', '3H', 'D') or simple units like 'seconds', 'minutes', 'hours', 'days',
        or if set to 'step_length', the rate will be accumulation per time step.
        The default is 'seconds'.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the calculation,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
        If you do not want to aggregate along the time dimension use earthkit.transforms.aggregate.reduce
    accumulation_type : str, optional
        Type of accumulation used in the input data. Options are:
        - "start_of_step": accumulation restarts at the beginning of each time step.
        - "start_of_forecast": accumulation starts at the beginning of the forecast and continues
          throughout the forecast period.
        - "start_of_day": accumulation restarts at the beginning of each day (00:00 UTC).
        Default is "start_of_step".
    from_first_step : bool, optional
        Only used if `accumulation_type` is "start_of_forecast". If True, the first time step's rate is 
        calculated by dividing the first accumulation value by the step duration. Default is True.
    provenance : bool, optional
        If True, appends a history entry to the output dataarray's attributes indicating
        that the transformation was applied. Default is True.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Data object with rate calculated based on the accumulation data.

    """
    if xp is None:
        xp = _tools.array_namespace_from_object(dataarray)

    time_dim_array = dataarray[time_dim]

    if step is None:
        if len(time_dim_array) < 2:
            raise ValueError("Cannot infer step from time dimension with less than two entries.")
        # assume uniform step based on first time steps
        step_obj = time_dim_array[1] - time_dim_array[0]
    else:
        _step = float(step)
        step_obj = pd.to_timedelta(_step, step_units)
        step_float = float(step_obj)

    if rate_units == "step_length":
        rate_scale_factor = 1.
        rate_units_str = ""  # Do not append anything to units attribute
    else:
        try:
            # Handle cases like '15min', '3H', etc.
            rate_obj = pd.to_timedelta(rate_units)
            rate_units_str = f"({rate_units})^-1"
        except ValueError:
            # Handle simple units like 'seconds', 'minutes', 'hours', 'days'
            rate_obj = pd.to_timedelta(1, rate_units)
            rate_units_str = f"{rate_units}^-1"
        rate_scale_factor = step_obj / rate_obj

    # Tidy up the rate_units_str for common abbreviations
    rate_units_str = rate_units_str.replace("seconds", "s").replace("minutes", "min")

    match accumulation_type:
        case "start_of_step":
            # Simple case: rate = accumulation / step
            output = dataarray / rate_scale_factor

        case "start_of_forecast":
            # Compute forward differences
            diff_data = dataarray.diff(time_dim, label="upper")
            time_diff = time_dim_array.diff(time_dim, label="upper")
            mask = xp.isclose(
                time_diff.astype("float64"), step_float
            )
            output = xr.where(
                mask, diff_data / rate_scale_factor, xp.nan
            )
            # The first time step corresponds to the first accumulation value divided by the step
            if from_first_step:
                first_value = dataarray.isel({time_dim: 0}) / rate_scale_factor
                output = xr.concat([first_value.expand_dims(time_dim), output], dim=time_dim)

        case "start_of_day":
            # Mask for midnight steps of the day (True = midnight)
            midnight_mask = time_dim_array.dt.floor("H").dt.hour == 0
            # To make a first_step_of_day_mask that corresponds to the diffed array below,
            # shift the mask back one step and then remove the first element in array,
            # or, more simply, remove the final element
            first_step_of_day_mask = midnight_mask.data[:-1]

            # If first step is midnight, we set to NaN as no prior data
            if midnight_mask.data[0]:
                dataarray[0] = xp.nan

            # Compute forward differences
            diff_data = dataarray.diff(time_dim, label="upper")
            time_diff = time_dim_array.diff(time_dim, label="upper")
            mask = xp.isclose(
                time_diff.astype("float64"), step_float
            )

            # Calculate the rate for values, excluding the first step of each day
            output = xr.where(mask & ~first_step_of_day_mask, diff_data / rate_scale_factor, xp.nan)

            # Now calculate the rate for the first step of each day using the original dataarray
            output = xr.where(
                first_step_of_day_mask,
                dataarray.sel(**{time_dim: time_diff[time_dim]}) / rate_scale_factor,
                output
            )

        case _:
            raise ValueError(f"Unknown accumulation_type: {accumulation_type}")

    output.name = dataarray.name + '_rate'
    if 'units' in dataarray.attrs:
        output.attrs.update({'units': dataarray.attrs['units'] + rate_units_str})
    if 'long_name' in dataarray.attrs:
        output.attrs['long_name'] = dataarray.attrs['long_name'] + ' rate'
    if provenance or "history" in dataarray.attrs:
        output.attrs["history"] = dataarray.attrs.get("history", "") + (
            f"\nConverted from accumulation to rate using earthkit.transforms.temporal.accumulation_to_rate."
        )

    return output
