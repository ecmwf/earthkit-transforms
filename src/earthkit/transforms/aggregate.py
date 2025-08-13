"""Temporary deprecated sub-module namespace."""

import warnings

from earthkit.transforms import _aggregate as general
from earthkit.transforms import climatology, ensemble, spatial, temporal
from earthkit.transforms._aggregate import reduce, resample, rolling_reduce

warnings.warn(
    "The 'earthkit.transforms.aggregate' module is deprecated and will be removed "
    "in version 2.X of earthkit.transforms. Please import the from earthkit.transforms, e.g.: "
    "from earthkit.transforms import spatial",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "climatology",
    "ensemble",
    "general",
    "spatial",
    "temporal",
    "reduce",
    "resample",
    "rolling_reduce",
]
