"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms.climatology import _aggregate as _src

from ._deprecate import _deprecated

# List all the public functions we want to re-export with warnings
_all_names = [
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

# Dynamically wrap and export each function
globals().update(
    {
        name: _deprecated(getattr(_src, name), old_name=name, new_module="earthkit.transforms.climatology")
        for name in _all_names
    }
)
