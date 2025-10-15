"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms import _aggregate as _src

from ._deprecate import _deprecated

# Explicitly wrap each function with a deprecation warning
how_label_rename = _deprecated(_src.how_label_rename, new_module="earthkit.transforms")
reduce = _deprecated(_src.reduce, new_module="earthkit.transforms")
rolling_reduce = _deprecated(_src.rolling_reduce, new_module="earthkit.transforms")
resample = _deprecated(_src.resample, new_module="earthkit.transforms")

# Explicitly declare __all__ for clean exports
__all__ = ["how_label_rename", "reduce", "rolling_reduce", "resample"]
