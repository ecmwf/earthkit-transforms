"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms.temporal import _aggregate as _src

from ._deprecate import _deprecated

# Explicitly wrap each function with a deprecation warning
standardise_time = _deprecated(_src.standardise_time, new_module="earthkit.transforms.temporal")
reduce = _deprecated(_src.reduce, new_module="earthkit.transforms.temporal")
mean = _deprecated(_src.mean, new_module="earthkit.transforms.temporal")
median = _deprecated(_src.median, new_module="earthkit.transforms.temporal")
min = _deprecated(_src.min, new_module="earthkit.transforms.temporal")
max = _deprecated(_src.max, new_module="earthkit.transforms.temporal")
std = _deprecated(_src.std, new_module="earthkit.transforms.temporal")
sum = _deprecated(_src.sum, new_module="earthkit.transforms.temporal")

daily_reduce = _deprecated(_src.daily_reduce, new_module="earthkit.transforms.temporal")
daily_mean = _deprecated(_src.daily_mean, new_module="earthkit.transforms.temporal")
daily_median = _deprecated(_src.daily_median, new_module="earthkit.transforms.temporal")
daily_max = _deprecated(_src.daily_max, new_module="earthkit.transforms.temporal")
daily_min = _deprecated(_src.daily_min, new_module="earthkit.transforms.temporal")
daily_std = _deprecated(_src.daily_std, new_module="earthkit.transforms.temporal")
daily_sum = _deprecated(_src.daily_sum, new_module="earthkit.transforms.temporal")

monthly_reduce = _deprecated(_src.monthly_reduce, new_module="earthkit.transforms.temporal")
monthly_mean = _deprecated(_src.monthly_mean, new_module="earthkit.transforms.temporal")
monthly_median = _deprecated(_src.monthly_median, new_module="earthkit.transforms.temporal")
monthly_min = _deprecated(_src.monthly_min, new_module="earthkit.transforms.temporal")
monthly_max = _deprecated(_src.monthly_max, new_module="earthkit.transforms.temporal")
monthly_std = _deprecated(_src.monthly_std, new_module="earthkit.transforms.temporal")
monthly_sum = _deprecated(_src.monthly_sum, new_module="earthkit.transforms.temporal")

rolling_reduce = _deprecated(_src.rolling_reduce, new_module="earthkit.transforms.temporal")

# Explicitly declare __all__ for clean exports
__all__ = [
    "standardise_time",
    "reduce",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "sum",
    "daily_reduce",
    "daily_mean",
    "daily_median",
    "daily_max",
    "daily_min",
    "daily_std",
    "daily_sum",
    "monthly_reduce",
    "monthly_mean",
    "monthly_median",
    "monthly_min",
    "monthly_max",
    "monthly_std",
    "monthly_sum",
    "rolling_reduce",
]
