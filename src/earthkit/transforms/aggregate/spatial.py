"""Module for preservation of the legacy API, it will be removed in version 2.X of earthkit.transforms."""
import logging

from earthkit.transforms.spatial._aggregate import mask, reduce

from ._deprecate import _deprecated

logger = logging.getLogger(__name__)

mask = _deprecated(mask, new_module="earthkit.transforms.spatial")
reduce = _deprecated(reduce, new_module="earthkit.transforms.spatial")


def masks(
    *_args,
    **_kwargs,
):
    logger.warning(
        "earthkit.transforms.aggregate.spatial.masks is deprecated, "
        "please use earthkit.transforms.aggregate.spatial.mask instead."
    )
    return mask(*_args, **_kwargs)


__all__ = [mask, "masks", reduce]
