"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms.ensemble import _aggregate as _src

from ._deprecate import _deprecated

# Explicitly wrap each function with a deprecations warning
mean = _deprecated(_src.mean, new_module="earthkit.transforms.ensemble")
std = _deprecated(_src.std, new_module="earthkit.transforms.ensemble")
min = _deprecated(_src.min, new_module="earthkit.transforms.ensemble")
max = _deprecated(_src.max, new_module="earthkit.transforms.ensemble")
reduce = _deprecated(_src.reduce, new_module="earthkit.transforms.ensemble")

# Explicitly declare __all__ for clean exports
__all__ = ["mean", "std", "min", "max", "reduce"]
