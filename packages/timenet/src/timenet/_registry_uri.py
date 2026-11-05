"""Parse the value that names a registry: a URL, a ``file://`` URI or a local path.

This module imports only TimeNet's errors, so that the settings can parse the store path.
"""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from timenet.errors import TimeNetRegistryError


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted TimeNet registry that the ``timenet://`` scheme is an alias for."""

_REMOTE_SCHEMES = ("timenet://", "http://", "https://", "s3://")

_SCHEMES = "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."


def registry_location(uri: str | Path, *, label: str = "registry") -> str | Path:
    """Parse a registry value into the URL of a remote registry or the directory of a local one.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        An ``http(s)://`` or ``s3://`` URL, with ``timenet://`` as the hosted registry, or a local directory
        with ``~`` expanded.

    Raises:
        TimeNetRegistryError: If the value is blank, has a scheme that no backend handles, is a
            ``timenet://`` URI with a path, or is a ``file://`` URI with a host.
    """
    text = str(uri)
    if text.startswith("timenet://"):
        # timenet:// is a bare alias for the hosted service root; it takes no path (dataset ids are
        # passed to the client methods, not folded into the registry URI). Reject a stray remainder
        # rather than drop it silently, so a mistyped URI fails loudly instead of hitting the default host.
        if text[len("timenet://") :].strip("/"):
            raise TimeNetRegistryError(f"{label} {text!r}: timenet:// takes no path")
        return TIMENET_REGISTRY_URL
    if text.startswith(_REMOTE_SCHEMES):
        return text
    return local_registry_path(text, label=label)


def local_registry_path(uri: str | Path, *, label: str = "registry") -> Path:
    """Resolve a registry URI to the local directory it names, for example the directory of the store.

    Args:
        uri: A ``file://`` URI or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        The local directory path with ``~`` expanded.

    Raises:
        TimeNetRegistryError: If the value is blank, names a remote backend, or is not a valid local
            registry URI.
    """
    text = str(uri)
    if not text.strip():
        # Path("") is the current directory, which a blank value never means.
        raise TimeNetRegistryError(f"{label} is blank; give a registry URL or a local path")
    if text.startswith(_REMOTE_SCHEMES):
        raise TimeNetRegistryError(f"{label} {text!r} is remote; the local registry needs a local directory")
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
