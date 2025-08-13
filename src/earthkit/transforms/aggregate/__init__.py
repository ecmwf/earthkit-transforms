"""Temporary deprecated sub-module namespace."""

import warnings

from . import general
from earthkit.transforms.spatial import _aggregate as spatial
from earthkit.transforms.climatology import _aggregate as climatology
from earthkit.transforms.ensemble import _aggregate as ensemble
from earthkit.transforms.temporal import _aggregate as temporal
from earthkit.transforms._aggregate import reduce, resample, rolling_reduce

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
