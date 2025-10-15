"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms import _aggregate as _src

from ._deprecate import _deprecated

_all_names = [
    "how_label_rename",
    "_reduce_dataarray",
    "reduce",
    "rolling_reduce",
    "_rolling_reduce_dataarray",
    "_dropna",
    "resample",
]

# Dynamically wrap and export each function
globals().update(
    {
        name: _deprecated(getattr(_src, name), old_name=name, new_module="earthkit.transforms")
        for name in _all_names
    }
)
