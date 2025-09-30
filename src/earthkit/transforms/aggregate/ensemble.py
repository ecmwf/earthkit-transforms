"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""
from earthkit.transforms.ensemble._aggregate import (
    max,
    mean,
    min,
    reduce,
    std,
)

standard_deviation = std

__all__ = ["mean", "standard_deviation", "min", "max", "reduce"]
