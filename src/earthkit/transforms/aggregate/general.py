# import warnings

from earthkit.transforms._aggregate import *
from earthkit.transforms import _aggregate

# warnings.warn(
#     "The 'earthkit.transforms.aggregate' module is deprecated and will be removed "
#     "in version 2.X of earthkit.transforms. Please import the from earthkit.transforms, e.g.: "
#     "from earthkit.transforms import spatial",
#     FutureWarning,
#     stacklevel=2,
# )

try:
    __all__ = _aggregate.__all__
except AttributeError:
    __all__ = [name for name in globals()]