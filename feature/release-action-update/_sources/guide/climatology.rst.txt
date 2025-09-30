earthkit.transforms.climatology
-----------------------------------------

The :doc:`../../_api/transforms/climatology/index` module includes methods
for aggregating climatologies and anomalies of data, including grouping the data in
time dimension to produce climatological daily or monthly values.
The API follows a similar pattern to the temporal aggregation methods, where there is a generic
`climatology.reduce`` method which allows all parameters to be passed, and a series of
convenience methods which wrap the `climatology.reduce` method and set the `how` parameter
and/or the `frequency` parameter.

.. autofunction:: earthkit.transforms.climatology.reduce


The convenience methods are all documented in the API reference guide:
:doc:`../../_api/transforms/temporal/index`, and users are advised
to refer to the notebook examples.
