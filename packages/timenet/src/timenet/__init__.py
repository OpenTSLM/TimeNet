"""Build, publish, and read time series datasets."""

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.client import TimeNet
    from timenet.coordinator.plan import BuildStep, CopyStep, Plan, ReadStep, Step
    from timenet.producer import build, plan_build, publish
    from timenet.registry import LocalRegistry
    from timenet.sources import InstalledConnectors


__all__ = [
    "BuildStep",
    "CopyStep",
    "InstalledConnectors",
    "LocalRegistry",
    "Plan",
    "ReadStep",
    "Step",
    "TimeNet",
    "build",
    "plan_build",
    "publish",
]


def __getattr__(name: str) -> object:
    """Load an exported name on first access, so ``import timenet`` stays cheap.

    Args:
        name: The attribute name.

    Returns:
        The exported object.

    Raises:
        AttributeError: If the package does not export the name.
    """
    module = {
        "BuildStep": "timenet.coordinator.plan",
        "CopyStep": "timenet.coordinator.plan",
        "InstalledConnectors": "timenet.sources",
        "LocalRegistry": "timenet.registry",
        "Plan": "timenet.coordinator.plan",
        "ReadStep": "timenet.coordinator.plan",
        "Step": "timenet.coordinator.plan",
        "TimeNet": "timenet.client",
        "build": "timenet.producer",
        "plan_build": "timenet.producer",
        "publish": "timenet.producer",
    }.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    # Later lookups find the name without this function.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List package attributes, including exports that have not loaded yet.

    Returns:
        The attribute names in sorted order.
    """
    return sorted({*globals(), *__all__})
