"""The :func:`open_registry` factory: dispatch a URI to the right registry backend."""

from pathlib import Path
from urllib.parse import urlparse

from timenet.registry.base import BaseRegistry
from timenet.registry.local import LocalRegistry
from timenet.registry.remote import RemoteRegistry
from timenet.registry.s3 import S3Registry
from timenet.registry.writable import WritableRegistry


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted TimeNet registry that the ``timenet://`` scheme is an alias for."""


def open_registry(uri: str | Path) -> BaseRegistry:
    """Open a registry from a URI or path.

    Dispatches by scheme: ``http(s)://`` and ``timenet://`` open a :class:`RemoteRegistry`
    (``timenet://`` is an alias for the hosted :data:`TIMENET_REGISTRY_URL`), ``s3://`` opens an
    :class:`S3Registry`, and ``file://`` or a plain path opens a :class:`LocalRegistry`.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.

    Returns:
        The matching registry backend.
    """
    text = str(uri)
    if text.startswith("timenet://"):
        rest = text.removeprefix("timenet://").strip("/")
        return RemoteRegistry(f"{TIMENET_REGISTRY_URL}/{rest}" if rest else TIMENET_REGISTRY_URL)
    if text.startswith(("http://", "https://")):
        return RemoteRegistry(text)
    if text.startswith("s3://"):
        return S3Registry(text)
    if text.startswith("file://"):
        return LocalRegistry(Path(urlparse(text).path))
    return LocalRegistry(Path(text))


def open_writable_registry(uri: str | Path) -> WritableRegistry:
    """Open a registry that supports :meth:`~WritableRegistry.store`, for curation to publish into.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.

    Returns:
        The matching writable registry backend.

    Raises:
        ValueError: If the resolved backend does not support writing.
    """
    registry = open_registry(uri)
    if not isinstance(registry, WritableRegistry):
        raise ValueError(f"registry {uri!r} is not writable")
    return registry
