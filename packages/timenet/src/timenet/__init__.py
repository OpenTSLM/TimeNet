"""TimeNet: register, query, download, convert, and explore time series datasets.

This package exports ``TimeNet``, ``LocalRegistry``, ``InstalledConnectors``, ``build``, ``plan_build``, ``publish``,
``Plan``, and ``Step``.
"""

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.client import TimeNet
    from timenet.coordinator.plan import Plan, Step
    from timenet.producer import build, plan_build, publish
    from timenet.registry import LocalRegistry
    from timenet.sources import InstalledConnectors


__all__ = ["InstalledConnectors", "LocalRegistry", "Plan", "Step", "TimeNet", "build", "plan_build", "publish"]


def __getattr__(name: str) -> object:
    """Load an exported name from its module on first use (PEP 562).

    Args:
        name: The name of the attribute.

    Returns:
        The exported object.

    Raises:
        AttributeError: If the package exports no such name.
    """
    if name == "TimeNet":
        module = "timenet.client"
    elif name == "LocalRegistry":
        module = "timenet.registry"
    elif name == "InstalledConnectors":
        module = "timenet.sources"
    elif name in {"Plan", "Step"}:
        module = "timenet.coordinator.plan"
    elif name in {"build", "plan_build", "publish"}:
        module = "timenet.producer"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    # Later lookups find the name without this function.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the attributes of the package, the exported names included before their first use (PEP 562).

    Returns:
        The names, sorted.
    """
    return sorted({*globals(), *__all__})
