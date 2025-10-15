"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""
from earthkit.transforms.ensemble import _aggregate as _src

from ._deprecate import _deprecated

_all_names = ["mean", "std", "min", "max", "reduce"]

# Dynamically wrap and export each function
globals().update(
    {
        name: _deprecated(getattr(_src, name), old_name=name, new_module="earthkit.transforms.climatology")
        for name in _all_names
    }
)

globals()["standard_deviation"] = globals()["std"]
