Climatologies and anomalies
===========================

What is a climatology?
----------------------

A climatology is a long-term average of a variable, such as temperature or precipitation,
over a specific period of time.
Typically, climatologies are calculated over a 30-year period, but the specific period can vary depending
on the context and the data available.
Climatologies are often used to understand the typical conditions of a region
and to identify anomalies, which are deviations from the long-term average.
Often climatologies are calculated for a given frequency, such as daily or monthly,
to understand the typical seasonal conditions for each day of the year or each month.


Calculating climatologies and anomalies with **earthkit.transforms**
--------------------------------------------------------------------

The :doc:`../autodocs/earthkit.transforms.climatology` module includes methods
for aggregating climatologies and anomalies of data, including grouping the data in
the time dimension to produce climatological daily or monthly values.
The API follows a similar pattern to the temporal aggregation methods, where there is a generic
`climatology.reduce`` method which allows all parameters to be passed, and a series of
convenience methods which wrap the `climatology.reduce` method and set the `how` parameter
and/or the `frequency` parameter.

.. dropdown:: Show API documentation for ``reduce``

   .. autofunction:: earthkit.transforms.climatology.reduce
      :no-index:


The convenience methods are all documented in the API reference guide:
:doc:`../autodocs/earthkit.transforms.climatology`, and users are advised
to refer to the notebook examples.


Quantile categories
-------------------

Climatological quantiles are often used to place values in categories, for example the below-normal,
near-normal and above-normal terciles. The ``climatology.quantile_categories`` method assigns each value
the number of climatological quantiles that it is greater than or equal to. When the quantiles are
grouped by a time frequency, such as ``month``, each time step is compared with the quantiles of its group:

.. code-block:: python

    from earthkit.transforms import climatology

    terciles = climatology.quantiles(
        data, [1 / 3, 2 / 3], frequency="month", climatology_range=("1991", "2020")
    )
    categories = climatology.quantile_categories(data, terciles)  # 0, 1 or 2
