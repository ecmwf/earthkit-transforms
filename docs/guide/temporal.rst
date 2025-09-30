earthkit.transforms.temporal
--------------------------------------

The :doc:`../../_api/transforms/temporal/index` module includes methods
for aggregating data in time. This includes reducing the data in time dimension
to a single value, daily values or monthly values.

To reduce the data in time dimension to a single value you can use the `temporal.reduce`,
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
