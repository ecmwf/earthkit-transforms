"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms import _aggregate as _src

from ._deprecate import _deprecated

_all_names = [
    "how_label_rename",
    "reduce",
    "rolling_reduce",
    "resample",
]

# Dynamically wrap and export each function
__all__ = [_deprecated(getattr(_src, name), new_module="earthkit.transforms") for name in _all_names]
