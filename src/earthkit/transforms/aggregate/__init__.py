"""Temporary deprecated sub-module namespace."""


from earthkit.transforms._aggregate import reduce, resample, rolling_reduce

from . import climatology, ensemble, general, spatial, temporal

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
