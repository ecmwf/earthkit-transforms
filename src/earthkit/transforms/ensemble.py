"""
Ensemble transformations for earthkit data objects.

Typically this is done with an xarray representation of data.
"""

from earthkit.transforms._aggregate.ensemble import (
    mean, std, min, max, reduce
)

all = [
    "mean",
    "std",
    "min",
    "max",
    "sum",
    "reduce",
]