"""Validate the registry options of the CLI.

A flag that names no registry is a usage error: it exits with code 2 and names the flag. A bad ``TIMENET_*``
variable stays a TimeNet error, which exits with code 1 and names the variable.
"""

import typer

from timenet._registry_uri import local_registry_path
from timenet.client import _sources_after_local_registry, default_local_registry
from timenet.config import settings
from timenet.errors import TimeNetRegistryError
from timenet.registry import LocalRegistry, WritableRegistry, open_registry
from timenet.sources import ReleaseSource


def registry_flag(value: str, flag: str) -> WritableRegistry:
    """Open the registry that a flag names.

    Args:
        value: The value of the flag.
        flag: The flag, such as ``--dest``.

    Returns:
        The registry.

    Raises:
        BadParameter: If the value names no registry.
    """
    try:
        return open_registry(value, label=flag)
    except TimeNetRegistryError as error:
        raise typer.BadParameter(str(error)) from error


def local_registry_from_flag(local_registry: str | None) -> LocalRegistry:
    """Open the local registry that ``--local-registry`` names, else the default one.

    Args:
        local_registry: The value of ``--local-registry``, or ``None``.

    Returns:
        The local registry.

    Raises:
        BadParameter: If the value names no local directory.
    """
    if local_registry is None:
        return default_local_registry()
    try:
        path = local_registry_path(local_registry, label="--local-registry")
    except TimeNetRegistryError as error:
        raise typer.BadParameter(str(error)) from error
    return default_local_registry(local_registry=path)


def sources_from_flags(
    local_registry: LocalRegistry, remote_registry: str | None, auto_build: bool | None
) -> list[ReleaseSource]:
    """Select release sources from CLI options and configured defaults.

    Args:
        local_registry: The local registry of the command, from :func:`local_registry_from_flag`.
        remote_registry: The value of ``--remote-registry``, or ``None``.
        auto_build: The value of ``--auto-build``, or ``None``.

    Returns:
        The local registry, the remote registry and the optional installed connectors, in lookup order.
    """
    remote = None if remote_registry is None else registry_flag(remote_registry, "--remote-registry")
    # Both registries are arguments too, so a bad variable that a flag replaces raises nothing.
    cfg = settings(local_registry=local_registry.root, remote_registry=remote_registry, auto_build=auto_build)
    return [local_registry, *_sources_after_local_registry(cfg, remote)]
