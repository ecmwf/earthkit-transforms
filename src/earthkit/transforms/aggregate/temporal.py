"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""
from earthkit.transforms.temporal import _aggregate as _src

from ._deprecate import _deprecated

_all_names = [
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

# Dynamically wrap and export each function
globals().update(
    {
        name: _deprecated(getattr(_src, name), old_name=name, new_module="earthkit.transforms.temporal")
        for name in _all_names
    }
)
