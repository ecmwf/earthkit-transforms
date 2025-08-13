"""Spatial transformations for earthkit data objects.

Typically this is done with an xarray representation of data and a geopandas representation of geometries.
"""

from earthkit.transforms._aggregate.spatial import mask, reduce

__all__ = ["mask", "reduce"]
