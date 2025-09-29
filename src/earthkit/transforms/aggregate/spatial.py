"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""
from earthkit.transforms.spatial import mask, reduce
from earthkit.transforms.spatial._aggregate import masks

__all__ = [mask, masks, reduce]
