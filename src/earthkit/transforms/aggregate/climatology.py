"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms.climatology import _aggregate as _src

from ._deprecate import _deprecated

# Explicitly wrap each function with a deprecation warning
reduce = _deprecated(_src.reduce, new_module="earthkit.transforms.climatology")
mean = _deprecated(_src.mean, new_module="earthkit.transforms.climatology")
median = _deprecated(_src.median, new_module="earthkit.transforms.climatology")
min = _deprecated(_src.min, new_module="earthkit.transforms.climatology")
max = _deprecated(_src.max, new_module="earthkit.transforms.climatology")
std = _deprecated(_src.std, new_module="earthkit.transforms.climatology")

daily_reduce = _deprecated(_src.daily_reduce, new_module="earthkit.transforms.climatology")
daily_mean = _deprecated(_src.daily_mean, new_module="earthkit.transforms.climatology")
daily_median = _deprecated(_src.daily_median, new_module="earthkit.transforms.climatology")
daily_min = _deprecated(_src.daily_min, new_module="earthkit.transforms.climatology")
daily_max = _deprecated(_src.daily_max, new_module="earthkit.transforms.climatology")
daily_std = _deprecated(_src.daily_std, new_module="earthkit.transforms.climatology")

monthly_reduce = _deprecated(_src.monthly_reduce, new_module="earthkit.transforms.climatology")
monthly_mean = _deprecated(_src.monthly_mean, new_module="earthkit.transforms.climatology")
monthly_median = _deprecated(_src.monthly_median, new_module="earthkit.transforms.climatology")
monthly_min = _deprecated(_src.monthly_min, new_module="earthkit.transforms.climatology")
monthly_max = _deprecated(_src.monthly_max, new_module="earthkit.transforms.climatology")
monthly_std = _deprecated(_src.monthly_std, new_module="earthkit.transforms.climatology")

quantiles = _deprecated(_src.quantiles, new_module="earthkit.transforms.climatology")
percentiles = _deprecated(_src.percentiles, new_module="earthkit.transforms.climatology")
anomaly = _deprecated(_src.anomaly, new_module="earthkit.transforms.climatology")
relative_anomaly = _deprecated(_src.relative_anomaly, new_module="earthkit.transforms.climatology")
auto_anomaly = _deprecated(_src.auto_anomaly, new_module="earthkit.transforms.climatology")

__all__ = [
    "reduce",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "daily_reduce",
    "daily_mean",
    "daily_median",
    "daily_min",
    "daily_max",
    "daily_std",
    "monthly_reduce",
    "monthly_mean",
    "monthly_median",
    "monthly_min",
    "monthly_max",
    "monthly_std",
    "quantiles",
    "percentiles",
    "anomaly",
    "relative_anomaly",
    "auto_anomaly",
]
