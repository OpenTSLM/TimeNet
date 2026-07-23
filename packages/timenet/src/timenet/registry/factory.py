"""The :func:`open_registry` factory: dispatch a URI to the right registry backend."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from timenet.config import settings
from timenet.errors import RegistryError
from timenet.registry.base import BaseRegistry
from timenet.registry.local import LocalRegistry
from timenet.registry.remote import RemoteRegistry
from timenet.registry.s3 import S3Registry
from timenet.registry.writable import WritableRegistry


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted TimeNet registry that the ``timenet://`` scheme is an alias for."""

_REMOTE_SCHEMES = ("timenet://", "http://", "https://", "s3://")


def open_registry(uri: str | Path) -> BaseRegistry:
    """Open a registry from a URI or path.

    Dispatches by scheme: ``http(s)://`` and ``timenet://`` open a :class:`RemoteRegistry`
    (``timenet://`` is an alias for the hosted :data:`TIMENET_REGISTRY_URL`), ``s3://`` opens an
    :class:`S3Registry`, and ``file://`` or a plain path opens a :class:`LocalRegistry`.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.

    Returns:
        The matching registry backend.

    Raises:
        ValueError: If ``uri`` carries a scheme no backend handles, or is a ``file://`` URI with a
            host component (which would silently drop the host).
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
        parsed = urlparse(text)
        if parsed.netloc:
            raise ValueError(f"file:// registry URI must be absolute (three slashes), got {text!r}")
        return LocalRegistry(Path(url2pathname(parsed.path)).expanduser())
    if "://" in text:
        raise ValueError(f"unsupported registry scheme in {text!r}")
    return LocalRegistry(Path(text).expanduser())


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


def local_registry_path(uri: str | Path) -> Path:
    """Resolve a registry URI to the local directory it names, for curation to write into.

    Every backend is a :class:`WritableRegistry`, so :func:`open_writable_registry` cannot tell a
    directory the engine can write to from a remote stub. This can.

    Args:
        uri: A ``file://`` URI or local path.

    Returns:
        The local directory the URI names, with ``~`` expanded.

    Raises:
        RegistryError: If the URI names a remote backend or carries an unsupported scheme, which
            curation cannot write to.
    """
    text = str(uri)
    if text.startswith(_REMOTE_SCHEMES):
        raise RegistryError(f"registry {text!r} is remote; curation writes to a local directory")
    if text.startswith("file://"):
        parsed = urlparse(text)
        if parsed.netloc:
            raise RegistryError(f"file:// registry URI must be absolute (three slashes), got {text!r}")
        return Path(url2pathname(parsed.path)).expanduser()
    if "://" in text:
        raise RegistryError(f"unsupported registry scheme in {text!r}")
    return Path(text).expanduser()


def default_registry_path() -> Path:
    """Resolve the local registry directory a build writes to (and the SDK reads from) by default.

    Honors ``$TIMENET_REGISTRY`` when it names a local directory, otherwise falls back to the default
    ``<TIMENET_HOME>/registry``. The single source of truth shared by the curate CLI and the
    ``timenet_connectors`` build/load helpers, so producer and consumer never disagree on where a
    dataset lands. Propagates :class:`~timenet.errors.RegistryError` from :func:`local_registry_path`
    when ``$TIMENET_REGISTRY`` names a remote registry, which cannot be built into.

    Returns:
        The local registry directory.
    """
    cfg = settings()
    return cfg.registry_path if cfg.registry is None else local_registry_path(cfg.registry)
