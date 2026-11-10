"""Warn at the first caller outside the timenet package."""

import inspect
import warnings


def warn_at_caller(warning: Warning) -> None:
    """Emit a warning at the first frame of the call stack whose module is outside the ``timenet`` package.

    Args:
        warning: The warning to emit.
    """
    # The walk starts at this function, which is level 1 of the stack and lies in the package.
    frame = inspect.currentframe()
    level = 1
    while frame is not None and _in_timenet(frame.f_globals.get("__name__", "")):
        frame = frame.f_back
        level += 1
    warnings.warn(warning, stacklevel=level)


def _in_timenet(module: str) -> bool:
    """Return whether a module name lies in the ``timenet`` package."""
    return module == "timenet" or module.startswith("timenet.")
