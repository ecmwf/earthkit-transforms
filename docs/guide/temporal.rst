earthkit.transforms.temporal
--------------------------------------

The :doc:`../../_api/transforms/temporal/index` module includes methods
for transforming data with respect to the temporal coordinate(s).
This includes aggregating the data in time dimension to a single value,
daily values, or monthly values and calculating rates from accumulated data.

Aggregation methods
^^^^^^^^^^^^^^^^^^^^^^

To aggregate the data in time dimension to a single value you can use the `temporal.reduce`,
`temporal.mean`, `temporal.sum`, `temporal.min`, `temporal.max` functions. These functions
take an xarray data object and return the aggregated value of the data. The time dimension
is automatically detected based on the metadata of the data object, to override this you can
use the `time_dim` parameter.

To calculate calculate daily/monthly values you can use the `temporal.daily_reduce` or
`temporal.monthly_reduce` functions. These functions reduce the data in time dimension to
daily or monthly values respectively. The `how` parameter can be used to specify the aggregation
method. The default is `mean`.

.. autofunction:: earthkit.transforms.temporal.daily_reduce
.. autofunction:: earthkit.transforms.temporal.monthly_reduce


In addition to the `XXX_reduce` functions, the temporal module also includes several methods
which calculate the desired reduction, without the "how" parameter. These methods are
wrappers of the `daily_reduce` and `monthly_reduce` methods and are documented
in the API reference guide: :doc:`../../_api/transforms/temporal/index`:

- `temporal.daily_mean`
- `temporal.daily_median`
- `temporal.daily_min`
- `temporal.daily_max`
- `temporal.daily_sum`
- `temporal.daily_std`
- `temporal.monthly_mean`
- `temporal.monthly_median`
- `temporal.monthly_min`
- `temporal.monthly_max`
- `temporal.monthly_sum`
- `temporal.monthly_std`


Rate calculations
^^^^^^^^^^^^^^^^^^^^^

To calculate rates from accumulated data you can use the `temporal.accumulation_to_rate` function.
This function takes an xarray data object and returns the rate of change of the data. The time dimension
is automatically detected based on the metadata of the data object, to override this
you can use the `time_dim` parameter. Similarly, the step between time points is automatically
detected, but can be overridden using the `step` parameter. If using the step parameter, the value
should be provided in hours unless specifying a different unit with the `step_units` parameter.
`step_units` can take any time units recognised by pandas, e.g. "minutes", "days", "15min", "3H", etc.

By default the function will calculate the rate per second, but this can be changed using the
`rate_units` parameter. `rate_units` can take any time units recognised by pandas,
e.g. "minutes", "days", "15min", "3H", etc. If you are only interested in "deaccumulating" the data,
i.e. converting accumulated values to step values, you can set `rate_units` to "step_length" such that the
`rate_units` will be equal to the step length between time points.

The `accumulation_type` parameter is used to specify the type of accumulation used in the input data.
The options are:
- "start_of_step": accumulation restarts at the beginning of each time step, e.g. ERA5.
- "start_of_forecast": accumulation restarts at the beginning of each forecast, e.g. Seeason forecasts.
- "start_of_day": accumulation restarts at the beginning of each day, e.g. ERA5-land.

.. autofunction:: earthkit.transforms.temporal.accumulation_to_rate
