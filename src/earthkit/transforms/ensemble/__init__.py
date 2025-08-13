"""Ensemble transformations for earthkit data objects.

Typically this is done with an xarray representation of data.
"""

from earthkit.transforms.ensemble._aggregate import max, mean, min, reduce, std, sum

__all__ = [
    "mean",
    "std",
    "min",
    "max",
    "sum",
    "reduce",
]
