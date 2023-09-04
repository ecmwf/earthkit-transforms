"""Aggregation tools for meteorological and climate data.."""


from earthkit.climate.aggregate import climatology, spatial, temporal

try:
    from earthkit.data.utils.module_inputs_wrapper import transform_module_inputs
except ImportError:
    pass
else:
    temporal = transform_module_inputs(temporal)

    climatology = transform_module_inputs(climatology)

    spatial = transform_module_inputs(spatial)

__all__ = ["__version__", "temporal", "climatology", "spatial"]
