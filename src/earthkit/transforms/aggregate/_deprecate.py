# Common function to deprecate a method/function
import warnings
from functools import wraps


def _deprecated(func, old_name=None, new_module="earthkit.transforms"):
    """Return a wrapper that emits a deprecation warning on first call."""
    old_name = old_name or func.__name__
    msg = (
        f"The function '{old_name}' from the legacy aggregate module is deprecated "
        f"and will be removed in version 2.X of earthkit.transforms. "
        f"Use '{new_module}.{func.__name__}' instead."
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper
