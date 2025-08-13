"""Temporal transformations for earthkit data objects.

Typically this is done with an xarray representation of data. Some pandas methods are used
for indexing and selecting data.
"""

from earthkit.transforms.temporal._aggregate import (
    daily_max,
    daily_mean,
    daily_median,
    daily_min,
    daily_reduce,
    daily_std,
    daily_sum,
    max,
    mean,
    median,
    min,
    monthly_max,
    monthly_mean,
    monthly_median,
    monthly_min,
    monthly_reduce,
    monthly_std,
    monthly_sum,
    reduce,
    rolling_reduce,
    standardise_time,
    std,
    sum,
)

__all__ = [
    "max",
    "mean",
    "median",
    "min",
    "reduce",
    "rolling_reduce",
    "std",
    "sum",
    "daily_max",
    "daily_mean",
    "daily_median",
    "daily_min",
    "daily_reduce",
    "daily_std",
    "daily_sum",
    "monthly_max",
    "monthly_mean",
    "monthly_median",
    "monthly_min",
    "monthly_reduce",
    "monthly_std",
    "monthly_sum",
    "standardise_time",
]
