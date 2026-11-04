"""The :func:`open_registry` factory: dispatch a URI to the right registry backend."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from timenet.config import settings
from timenet.errors import TimeNetRegistryError
from timenet.registry.local import LocalRegistry
from timenet.registry.remote import RemoteRegistry
from timenet.registry.s3 import S3Registry
from timenet.registry.writable import WritableRegistry


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted TimeNet registry that the ``timenet://`` scheme is an alias for."""

_REMOTE_SCHEMES = ("timenet://", "http://", "https://", "s3://")


_SCHEMES = "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."


def open_registry(uri: str | Path, *, cache_dir: str | Path | None = None, label: str = "registry") -> WritableRegistry:
    """Open a registry from a URI or path.

    The scheme selects the backend. ``http(s)://`` and ``timenet://`` open a :class:`RemoteRegistry`,
    and ``timenet://`` is an alias for the hosted :data:`TIMENET_REGISTRY_URL`. ``s3://`` opens an
    :class:`S3Registry`. ``file://`` or a plain path opens a :class:`LocalRegistry`.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        cache_dir: Where a remote backend caches downloads. The local backend ignores it. Defaults to
            the configured storage directory when ``None``.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        The matching registry backend.

    Raises:
        TimeNetRegistryError: If ``uri`` is blank or carries a scheme no backend handles, if a
            ``timenet://`` URI carries a path, or if ``uri`` is a ``file://`` URI with a host component
            (the host would be dropped without notice) or without a path.
    """
    text = str(uri)
    if text.startswith("timenet://"):
        # timenet:// is a bare alias for the hosted service root; it takes no path (dataset ids are
        # passed to the client methods, not folded into the registry URI). Reject a stray remainder
        # rather than drop it silently, so a mistyped URI fails loudly instead of hitting the default host.
        if text[len("timenet://") :].strip("/"):
            raise TimeNetRegistryError(f"{label} {text!r}: timenet:// takes no path")
        return RemoteRegistry(TIMENET_REGISTRY_URL, cache_dir=cache_dir)
    if text.startswith(("http://", "https://")):
        return RemoteRegistry(text, cache_dir=cache_dir)
    if text.startswith("s3://"):
        return S3Registry(text, cache_dir=cache_dir)
    return LocalRegistry(local_registry_path(text, label=label))


def local_registry_path(uri: str | Path, *, label: str = "registry") -> Path:
    """Resolve a registry URI to the local directory it names, for build to write into.

    Args:
        uri: A ``file://`` URI or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        The local directory the URI names, with ``~`` expanded.

    Raises:
        TimeNetRegistryError: If the value is blank, names a remote backend, or is not a valid local
            registry URI.
    """
    text = str(uri)
    if not text.strip():
        # Path("") is the current directory, which a blank value never means.
        raise TimeNetRegistryError(f"{label} is blank; give a registry URL or a local path")
    if text.startswith(_REMOTE_SCHEMES):
        raise TimeNetRegistryError(f"{label} {text!r} is remote; build writes to a local directory")
    if text.startswith("file://"):
        parsed = urlparse(text)
        if parsed.netloc:
            raise TimeNetRegistryError(f"{label} {text!r} must be an absolute file:// URI (three slashes). {_SCHEMES}")
        if not parsed.path:
            raise TimeNetRegistryError(f"{label} {text!r} must name an absolute path")
        return Path(url2pathname(parsed.path)).expanduser()
    if "://" in text:
        raise TimeNetRegistryError(f"{label} {text!r} has an unsupported registry scheme. {_SCHEMES}")
    return Path(text).expanduser()


def default_registry_path() -> Path:
    """Resolve the local registry directory a build writes to (and the SDK reads from) by default.

    If ``$TIMENET_REGISTRY`` names a local directory, use it. Otherwise use the default
    ``<TIMENET_HOME>/registry``. The build CLI and the ``timenet_connectors`` build and load helpers
    all call this function, so producer and consumer agree on where a dataset lands. If
    ``$TIMENET_REGISTRY`` names a remote registry, this function propagates the
    :class:`~timenet.errors.TimeNetRegistryError` from :func:`local_registry_path`, because a build cannot
    write to a remote registry.

    Returns:
        The local registry directory.
    """
    cfg = settings()
    return cfg.registry_path if cfg.registry is None else local_registry_path(cfg.registry)
