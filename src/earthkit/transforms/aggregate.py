"""Temporary deprecated sub-module namespace."""
import warnings

from earthkit.transforms import _aggregate

warnings.warn(
    "The 'earthkit.transforms.aggregate' module is deprecated and will be removed "
    "in version 2.X of earthkit.transforms. Please import the from earthkit.transforms, e.g.: "
    "from earthkit.transforms import spatial",
    DeprecationWarning,
    stacklevel=2,
)

# Ensure the same __all__ as _aggregate, if defined
try:
    __all__ = _aggregate.__all__
except AttributeError:
    # Fallback: export everything that's not private
    __all__ = [name for name in globals() if not name.startswith("_")]
