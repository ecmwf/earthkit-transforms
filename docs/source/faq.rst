Frequently asked questions
==========================


How do I calculate daily statistics such as the daily mean, maximum, minimum, etc.?
-----------------------------------------------------------------------------------

Please see the :doc:`../how-tos/temporal/howto_calculate_daily_mean_era5` example for calculating the daily mean of ERA5 data.
The same approach can be used to calculate other statistics such as the daily maximum and minimum by using the appropriate
aggregation function (e.g., `daily_max` for maximum, `daily_min` for minimum) instead of `daily_mean` in the code.


How are the dimensions to perform computations over detected?
-------------------------------------------------------------

Please see the :doc:`../concepts/dimension-detection` page for more information on how dimensions and coordinates
are detected in **earthkit.transforms**.


How do I calculate climatologies?
---------------------------------

Please see the :doc:`../how-tos/climatology/howto_climatology_mean_era5` example for a basic example of calculating climatologies of data.
For a more detailed explanations and examples, please see the
:doc:`../concepts/climatology` concept page and the
:doc:`../tutorials/climatology/01-era5-climatology` tutorial notebook.
