"""Temporary deprecated sub-module namespace."""

import warnings

from earthkit.transforms._aggregate import reduce, resample, rolling_reduce

from . import climatology, ensemble, general, spatial, temporal

warnings.warn(
    "The 'earthkit.transforms.aggregate' module is deprecated and will be removed "
    "in version 2.X of earthkit.transforms. Please import the from earthkit.transforms, e.g.: "
    "from earthkit.transforms import spatial",
    FutureWarning,
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
