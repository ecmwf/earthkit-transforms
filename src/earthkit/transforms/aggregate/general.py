"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""

from earthkit.transforms._aggregate import *  # noqa: F403
from earthkit.transforms._aggregate import (
    reduce,
    resample,
    rolling_reduce,
)

__all__ = [name for name in globals()]

__all__.extend([resample, reduce, rolling_reduce])
