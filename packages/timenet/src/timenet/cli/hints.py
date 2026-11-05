"""Add CLI error hints and validate registry options.

A flag that names no registry is a usage error: it exits with code 2 and names the flag. A bad ``TIMENET_*``
variable stays a TimeNet error, which exits with code 1 and names the variable.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import typer

from timenet._hints import Facts, explain
from timenet._registry_uri import local_registry_path, registry_location
from timenet.client import default_sources, default_store
from timenet.errors import TimeNetRegistryError
from timenet.registry import LocalRegistry, WritableRegistry, open_registry
from timenet.sources import ReleaseSource


def cli_hints(facts: Facts) -> AbstractContextManager[None]:
    """Create a context manager that adds CLI hints to TimeNet errors.

    Args:
        facts: Details of the command's operation.

    Returns:
        The context manager.
    """
    return explain(facts, "cli")


def registry_flag(value: str, flag: str, *, store: Path | None = None) -> WritableRegistry:
    """Open the registry that a flag names.

    Args:
        value: The value of the flag.
        flag: The flag, such as ``--dest``.
        store: The store directory whose copies of the same releases a remote registry reads, or ``None``.

    Returns:
        The registry.

    Raises:
        BadParameter: If the value names no registry.
    """
    try:
        return open_registry(value, store=store, label=flag)
    except TimeNetRegistryError as error:
        raise typer.BadParameter(str(error)) from error


def store_from_flag(local_registry: str | None) -> LocalRegistry:
    """Open the store that ``--local-registry`` names, else the default store.

    Args:
        local_registry: The value of ``--local-registry``, or ``None``.

    Returns:
        The store.
    """
    _check(local_registry, "--local-registry", local_registry_path)
    return default_store(local_registry=local_registry)


def sources_from_flags(
    local_registry: str | None, remote_registry: str | None, auto_build: bool | None
) -> list[ReleaseSource]:
    """Select release sources from CLI options and configured defaults.

    Args:
        local_registry: The value of ``--local-registry``, or ``None``.
        remote_registry: The value of ``--remote-registry``, or ``None``.
        auto_build: The value of ``--auto-build``, or ``None``.

    Returns:
        The store, the registry and the optional installed connectors, in lookup order.
    """
    _check(local_registry, "--local-registry", local_registry_path)
    _check(remote_registry, "--remote-registry", registry_location)
    return default_sources(local_registry=local_registry, remote_registry=remote_registry, auto_build=auto_build)


def _check(value: str | None, flag: str, parse: Callable[..., object]) -> None:
    """Check that a flag names a registry, and raise ``BadParameter`` if not."""  # ruff: ignore[docstring-missing-exception]
    if value is None:
        return
    try:
        parse(value, label=flag)
    except TimeNetRegistryError as error:
        raise typer.BadParameter(str(error)) from error
