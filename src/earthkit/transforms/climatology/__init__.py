"""Climatological transformations for earthkit data objects.

Typically this is done with an xarray representation of data.
"""

from earthkit.transforms.climatology._aggregate import (
    anomalazy,
    anomaly,
    daily_max,
    daily_mean,
    daily_median,
    daily_min,
    daily_std,
    max,
    mean,
    median,
    min,
    monthly_max,
    monthly_mean,
    monthly_median,
    monthly_min,
    monthly_std,
    percentiles,
    quantiles,
    relative_anomaly,
    std,
)

__all__ = [
    "mean",
    "max",
    "min",
    "std",
    "median",
    "quantiles",
    "percentiles",
    "daily_max",
    "daily_mean",
    "daily_median",
    "daily_min",
    "daily_std",
    "monthly_max",
    "monthly_mean",
    "monthly_median",
    "monthly_min",
    "monthly_std",
    "anomalazy",
    "anomaly",
    "relative_anomaly",
]
