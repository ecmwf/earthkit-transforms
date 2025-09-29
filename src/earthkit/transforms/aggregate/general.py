"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms import _aggregate
from earthkit.transforms._aggregate import (
    reduce,
    resample,
    rolling_reduce,
)

try:
    __all__ = _aggregate.__all__
except AttributeError:
    __all__ = [name for name in globals()]

__all__.extend([resample, reduce, rolling_reduce])
