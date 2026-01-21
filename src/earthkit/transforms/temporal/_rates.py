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
import logging
import typing as T

import pandas as pd
import xarray as xr
from earthkit.transforms import _tools

logger = logging.getLogger(__name__)


def deaccumulate(
    dataarray: xr.Dataset | xr.DataArray,
    *_args,
    **_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Alias for `accumulation_to_rate` function with rate_units set to 'step_length'.

    The returned object will preserve the units and long_name attributes of the input dataarray.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        Data accumulated along time to be converted into rate (per time step).
    step : timedelta | str , optional
        Interval between consecutive time steps.
        If a string, it should be a valid pandas time frequency string (e.g., '15min', '3h', '1 day').
        If not provided, the will be inferred from the data.
    rate_units : timedelta | str, optional
        Units for the output rate. If a string, it must be a valid pandas time frequency string
        (e.g., '15min', '3h', '1 day') or simple units like 'seconds', 'minutes', 'hours', 'days'.
        If set to 'step_length', the rate will be accumulation per time step ("deaccumulated") and the
        returned object will preserve the units and long_name attributes of the input dataarray.
        The default is 'seconds'.
    rate_label: str | None = None, optional
        Suffix to append to the name of the output dataarray. If None, defaults to
        'rate' or 'per_step' depending on the rate_units.
    xp : T.Any
        The array namespace to use for the reduction. If None, it will be inferred from the dataarray.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the calculation,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    accumulation_type : str, optional
        Type of accumulation used in the input data. Default is "start_of_forecast". Options are:
        - "start_of_step": accumulation restarts at the beginning of each time step.
        - "start_of_forecast": accumulation starts at the beginning of the forecast and continues
          throughout the forecast period.
        - "start_of_day": accumulation restarts at the beginning of each day (00:00 UTC).
        Default is "start_of_step".
    from_first_step : bool, optional
        Only used if `accumulation_type` is "start_of_forecast". If True, the first time step's rate is
        calculated by dividing the first accumulation value by the step duration. Default is False.
    provenance : bool, optional
        If True, appends a history entry to the output dataarray's attributes indicating
        that the transformation was applied. Default is True.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Data object with deaccumulation data.

    """
    if "rate_units" in _kwargs:
        logger.warning("The 'rate_units' parameter is not applicable for `deaccumulate` and will be ignored.")
    _kwargs["rate_units"] = "step_length"
    _kwargs.setdefault("accumulation_type", "start_of_forecast")
    return accumulation_to_rate(dataarray, *_args, **_kwargs)


@_tools.time_dim_decorator
@_tools.transform_inputs_decorator()
def accumulation_to_rate(
    dataarray: xr.Dataset | xr.DataArray,
    *_args,
    **_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Convert a variable accumulated from the beginning of the forecast to a rate.

    The rate is computed by considering first-order discrete differences in data
    along the inferred, or specified, time dimension.
    The difference are converted to a rate by dividing by the time step duration, unless
    specified otherwise.

    Parameters
    ----------
    dataarray : xr.DataArray | xr.Dataset
        Data accumulated along time to be converted into rate (per second).
    step : timedelta | str , optional
        Interval between consecutive time steps.
        If a string, it should be a valid pandas time frequency string (e.g., '15min', '3h', '1 day').
        If not provided, the will be inferred from the data.
    rate_units : timedelta | str, optional
        Units for the output rate. If a string, it must be a valid pandas time frequency string
        (e.g., '15min', '3h', '1 day') or simple units like 'seconds', 'minutes', 'hours', 'days'.
        If set to 'step_length', the rate will be accumulation per time step ("deaccumulated") and the
        returned object will preserve the units and long_name attributes of the input dataarray.
        The default is 'seconds'.
    rate_label: str | None = None, optional
        Suffix to append to the name of the output dataarray. If None, defaults to
        'rate' or 'per_step' depending on the rate_units.
    xp : T.Any
        The array namespace to use for the reduction. If None, it will be inferred from the dataarray.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the calculation,
        default behaviour is to deduce time dimension from
        attributes of coordinates, then fall back to `"time"`.
    accumulation_type : str, optional
        Type of accumulation used in the input data. Default is "start_of_step". Options are:
        - "start_of_step": accumulation restarts at the beginning of each time step.
        - "start_of_forecast": accumulation starts at the beginning of the forecast and continues
          throughout the forecast period.
        - "start_of_day": accumulation restarts at the beginning of each day (00:00 UTC).
        Default is "start_of_step".
    from_first_step : bool, optional
        Only used if `accumulation_type` is "start_of_forecast". If True, the first time step's rate is
        calculated by dividing the first accumulation value by the step duration. Default is False.
    provenance : bool, optional
        If True, appends a history entry to the output dataarray's attributes indicating
        that the transformation was applied. Default is True.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Data object with rate calculated based on the accumulation data.

    """
    if isinstance(dataarray, xr.Dataset):
        out_ds = xr.Dataset().assign_attrs(dataarray.attrs)
        for var in dataarray.data_vars:
            out_da = _accumulation_to_rate_dataarray(dataarray[var], *_args, **_kwargs)
            out_ds[out_da.name] = out_da
        return out_ds

    return _accumulation_to_rate_dataarray(dataarray, *_args, **_kwargs)


def _accumulation_to_rate_dataarray(
    dataarray: xr.DataArray,
    step: None | str | pd.Timedelta = None,
    rate_units: None | str | pd.Timedelta = "seconds",
    xp: T.Any = None,
    time_dim: str | None = None,
    accumulation_type: str = "start_of_step",
    from_first_step: bool = False,
    provenance: bool = True,
    rate_label: str | None = None,
) -> xr.DataArray:
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
    dataarray : xr.DataArray
        Data accumulated along time to be converted into rate (per second).
    accumulation_type : str, optional
        Type of accumulation used in the input data. Options are:
        - "start_of_step": accumulation restarts at the beginning of each time step.
        - "start_of_forecast": accumulation starts at the beginning of the forecast and continues
          throughout the forecast period.
        - "start_of_day": accumulation restarts at the beginning of each day (00:00 UTC).
        Default is "start_of_step".
    step : timedelta | str , optional
        Interval between consecutive time steps.
        If a string, it should be a valid pandas time frequency string (e.g., '15min', '3h', '1 day').
        If not provided, the will be inferred from the data.
    rate_units : timedelta | str, optional
        Units for the output rate. If a string, it must be a valid pandas time frequency string
        (e.g., '15min', '3h', '1 day') or simple units like 'seconds', 'minutes', 'hours', 'days'.
        If set to 'step_length', the rate will be accumulation per time step ("deaccumulated") and the
        returned object will preserve the units and long_name attributes of the input dataarray.
        The default is 'seconds'.
    rate_label: str | None = None, optional
        Suffix to append to the name of the output dataarray name, as _{rate_label}.
        If None, defaults to 'rate' or 'per_step' depending on the rate_units.
    xp : T.Any
        The array namespace to use for the reduction. If None, it will be inferred from the dataarray.
    time_dim : str, optional
        Name of the time dimension, or coordinate, in the xarray object to use for the calculation,
        default behaviour is to deduce time dimension from attributes of coordinates,
        then fall back to `"time"`.
    from_first_step : bool, optional
        Only used if `accumulation_type` is "start_of_forecast" or "start_of_day" and the first step is not
        midnight. If True, the first time step's rate is calculated by dividing the first accumulation
        value by the step duration. Default is False.
    provenance : bool, optional
        If True, appends a history entry to the output dataarray's attributes indicating
        that the transformation was applied. Default is True.

    Returns
    -------
    xr.DataArray
        Data object with rate calculated based on the accumulation data.

    """
    if xp is None:
        xp = _tools.array_namespace_from_object(dataarray)

    time_dim_array = dataarray[time_dim]
    # If the time_dim_array is a "forecast_reference_time", then we see if there is a "forecast_period"
    period_dim_array = None
    if (
        time_dim_array.attrs.get("standard_name") == "forecast_reference_time"
        or time_dim_array.name == "forecast_reference_time"
    ):
        # Check for forecast_period as standard_name or name in coords
        for coord in dataarray.coords.values():
            if coord.attrs.get("standard_name") == "forecast_period" or coord.name == "forecast_period":
                period_dim_array = coord
                break
        else:
            for name in ["step", "leadtime", "forecast_period"]:
                if name in dataarray.coords:
                    period_dim_array = dataarray[name]
                    break

    # decide whether to use period_dim_array or time_dim_array for step calculation
    if period_dim_array is not None:
        step_dim_array = period_dim_array
    else:
        step_dim_array = time_dim_array

    step_dim = step_dim_array.name

    if step is None:
        if len(step_dim_array) < 2:
            if period_dim_array is not None:
                # This means that there is only one forecast period per forecast reference time
                # So we can treat the period_dim_array as the step_obj directly
                step_obj = step_dim_array
            else:
                raise ValueError("Cannot infer step from time dimension with less than two entries.")
        else:
            # step_obj is an array of time differences
            step_obj = step_dim_array.diff(step_dim, label="upper")

        if from_first_step or accumulation_type in ["start_of_step"]:
            # Prepend the first step assuming it's the same as the second step
            first_step = step_obj.isel(**{step_dim: 0}).expand_dims(step_dim)
            first_step = first_step.assign_coords(
                {step_dim: step_dim_array.isel(**{step_dim: 0}).expand_dims(step_dim)}
            )
            step_obj = xr.concat([first_step, step_obj], dim=step_dim)
    else:
        # Convert to timedelta
        step_obj = pd.to_timedelta(step)

    if rate_units == "step_length":
        rate_scale_factor = 1.0
        rate_units_str = ""  # Do not append anything to units attribute
        if rate_label is None:
            rate_label = "per_step"
    else:
        if isinstance(rate_units, str) and not rate_units[0].isnumeric():
            rate_units = "1 " + rate_units

        rate_obj = pd.to_timedelta(rate_units)
        rate_units_str = f" {_tools.timedelta_to_largest_unit(rate_obj)}^-1"
        rate_scale_factor = step_obj / rate_obj
        if rate_label is None:
            rate_label = "rate"

    # Tidy up the rate_units_str for common abbreviations
    rate_units_str = rate_units_str.replace("seconds", "s").replace("minutes", "min")

    match accumulation_type:
        case "start_of_step":
            # Simple case: rate = accumulation / step
            output = dataarray / rate_scale_factor

        case "start_of_forecast":
            # Compute forward differences
            diff_data = dataarray.diff(step_dim, label="upper")
            if from_first_step:
                # Prepend diff with first value being the same as the first dataarray value
                first_diff = dataarray.isel({step_dim: 0}).expand_dims(step_dim)
                diff_data = xr.concat([first_diff, diff_data], dim=step_dim)

            output = diff_data / rate_scale_factor

        case "start_of_day":
            if period_dim_array is not None:
                # This sceanario should be treated as "start_of_forecast", but we check that no step is > 24h
                if step_dim_array.max() > pd.Timedelta("1 days"):
                    logger.warning(
                        "For accumulation_type 'start_of_day' with time represented as forecast periods, "
                        "the step between periods should not be greater than 24 hours. "
                        "accumulation_to_rate is proceeding treating accumulation as 'start_of_forecast'. "
                        "Please check that you are performing the correct transformation."
                    )
                #  where the start of forecast is at the start of each day
                # Compute forward differences
                diff_data = dataarray.diff(step_dim, label="upper")
                if from_first_step:
                    # Prepend diff with first value being the same as the first dataarray value
                    first_diff = dataarray.isel({step_dim: 0}).expand_dims(step_dim)
                    diff_data = xr.concat([first_diff, diff_data], dim=step_dim)

                output = diff_data / rate_scale_factor

            else:
                # TODO: think of a more robust check to include here instead of a blanket warning
                logger.warning(
                    "Please be aware that using accumulation_type='start_of_day' with a 'valid_time' "
                    "representation assumes that the data is contiguous with steps that can be identified"
                    "as midnight in order to reset accumulation."
                )
                # Mask for midnight steps of the day (True = midnight)
                midnight_mask = step_dim_array.dt.floor("h").dt.hour == 0
                # Shift the mask forward one step for first step of day.
                # Leave the first element as NaN as not possible to know if it was the first step of the day,
                # this eventuality is handled explicitly by the "from_first_step" logic below
                first_step_of_day_mask = midnight_mask.shift(**{step_dim: 1})

                # Compute forward differences
                diff_data = dataarray.diff(step_dim, label="upper")
                if from_first_step:
                    if not midnight_mask.data[0]:
                        first_diff = dataarray.isel({step_dim: 0}).expand_dims(step_dim)
                    else:
                        # If the first step is midnight, we do not know the accumulation before it,
                        # so we set it to NaN
                        first_diff = xp.nan * dataarray.isel({step_dim: 0}).expand_dims(step_dim)
                    diff_data = xr.concat([first_diff, diff_data], dim=step_dim)
                else:
                    # We must drop the first element of the first_step_of_day_mask to match the diffed array
                    first_step_of_day_mask = first_step_of_day_mask.isel({step_dim: slice(1, None)})

                # Calculate the rate for values, excluding the first step of each day
                output = diff_data / rate_scale_factor

                # Now calculate the rate for the first step of each day using the original dataarray
                _indexer: dict[str, xr.DataArray] = {step_dim: output[step_dim]}
                output = xr.where(
                    first_step_of_day_mask,
                    dataarray.sel(_indexer) / rate_scale_factor,
                    output,
                )

        case _:
            raise ValueError(f"Unknown accumulation_type: {accumulation_type}")

    new_name = "_".join(filter(None, [dataarray.name, rate_label]))
    output = output.rename(new_name)
    # Clear any existing attributes and set new ones
    output.attrs = {}
    if "units" in dataarray.attrs:
        output.attrs.update({"units": dataarray.attrs["units"] + rate_units_str})
    if "long_name" in dataarray.attrs:
        output.attrs["long_name"] = " ".join(
            filter(None, [dataarray.attrs["long_name"], rate_label.replace("_", " ")])
        )
    if provenance or "history" in dataarray.attrs:
        output.attrs["history"] = "\n".join(
            filter(
                None,
                [
                    dataarray.attrs.get("history", ""),
                    (
                        "Converted from accumulation to rate using "
                        "earthkit.transforms.temporal.accumulation_to_rate."
                    ),
                ],
            )
        )

    return output
