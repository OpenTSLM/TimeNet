"""The :func:`open_registry` factory: dispatch a URI to the right registry backend."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

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
        return LocalRegistry(Path(url2pathname(parsed.path)))
    if "://" in text:
        raise ValueError(f"unsupported registry scheme in {text!r}")
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
        if not parsed.path:
            raise RegistryError("file:// registry URI must name an absolute path")
        return Path(url2pathname(parsed.path)).expanduser()
    if "://" in text:
        raise RegistryError(f"unsupported registry scheme in {text!r}")
    return Path(text).expanduser()
