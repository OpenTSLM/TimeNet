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


def open_registry(uri: str | Path, *, cache_dir: str | Path | None = None) -> BaseRegistry:
    """Open a registry from a URI or path.

    The scheme selects the backend. ``http(s)://`` and ``timenet://`` open a :class:`RemoteRegistry`,
    and ``timenet://`` is an alias for the hosted :data:`TIMENET_REGISTRY_URL`. ``s3://`` opens an
    :class:`S3Registry`. ``file://`` or a plain path opens a :class:`LocalRegistry`.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        cache_dir: Where a remote backend caches downloads; ignored by the local backend. Defaults to
            the configured storage directory when ``None``.

    Returns:
        The matching registry backend.

    Raises:
        ValueError: If ``uri`` carries a scheme no backend handles, if a ``timenet://`` URI carries a
            path, or if ``uri`` is a ``file://`` URI with a host component (the host would be dropped
            without notice).
    """
    text = str(uri)
    if text.startswith("timenet://"):
        # timenet:// is a bare alias for the hosted service root; it takes no path (dataset ids are
        # passed to the client methods, not folded into the registry URI). Reject a stray remainder
        # rather than drop it silently, so a mistyped URI fails loudly instead of hitting the default host.
        if text[len("timenet://") :].strip("/"):
            raise ValueError(f"timenet:// takes no path; got {text!r}")
        return RemoteRegistry(TIMENET_REGISTRY_URL, cache_dir=cache_dir)
    if text.startswith(("http://", "https://")):
        return RemoteRegistry(text, cache_dir=cache_dir)
    if text.startswith("s3://"):
        return S3Registry(text, cache_dir=cache_dir)
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

    Every backend is a :class:`WritableRegistry`, so :func:`open_writable_registry` cannot separate a
    directory the engine can write to from a remote stub. This function can.

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


def default_registry_path() -> Path:
    """Resolve the local registry directory a build writes to (and the SDK reads from) by default.

    If ``$TIMENET_REGISTRY`` names a local directory, use it. Otherwise use the default
    ``<TIMENET_HOME>/registry``. The curate CLI and the ``timenet_connectors`` build and load helpers
    all call this function, so producer and consumer agree on where a dataset lands. If
    ``$TIMENET_REGISTRY`` names a remote registry, this function propagates the
    :class:`~timenet.errors.RegistryError` from :func:`local_registry_path`, because a build cannot
    write to a remote registry.

    Returns:
        The local registry directory.
    """
    cfg = settings()
    return cfg.registry_path if cfg.registry is None else local_registry_path(cfg.registry)
