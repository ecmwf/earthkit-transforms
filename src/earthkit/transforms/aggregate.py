"""Temporary deprecated sub-module namespace."""

import warnings

from earthkit.transforms._aggregate import climatology, ensemble, general, spatial, temporal
from earthkit.transforms._aggregate.general import reduce, resample, rolling_reduce

warnings.warn(
    "The 'earthkit.transforms.aggregate' module is deprecated and will be removed "
    "in version 2.X of earthkit.transforms. Please import the from earthkit.transforms, e.g.: "
    "from earthkit.transforms import spatial",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from earthkit.data.utils.module_inputs_wrapper import (
        transform_function_inputs,
        transform_module_inputs,
    )
except ImportError:
    pass
else:
    general = transform_module_inputs(general)
    temporal = transform_module_inputs(temporal)
    climatology = transform_module_inputs(climatology)
    ensemble = transform_module_inputs(ensemble)
    spatial = transform_module_inputs(spatial)
    reduce = transform_function_inputs(reduce)
    rolling_reduce = transform_function_inputs(rolling_reduce)
    resample = transform_function_inputs(resample)

__all__ = [
    "ensemble",
    "general",
    "temporal",
    "climatology",
    "spatial",
    "reduce",
    "resample",
    "rolling_reduce",
]
