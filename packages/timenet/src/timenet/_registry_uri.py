"""Parse registry URLs and local directory paths."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from timenet.errors import TimeNetRegistryError


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted registry URL represented by ``timenet://``."""

_REMOTE_SCHEMES = ("timenet://", "http://", "https://", "s3://")

_SCHEMES = "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."


def registry_location(uri: str | Path, *, label: str = "registry") -> str | Path:
    """Parse a registry value into the URL of a remote registry or the directory of a local one.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        A remote URL with ``timenet://`` resolved, or a local path with ``~`` expanded.

    Raises:
        TimeNetRegistryError: If the value is blank or uses an unsupported scheme or invalid registry URI.
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
    """Resolve a local registry address to a directory path.

    Args:
        uri: A ``file://`` URI or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        The local directory path with ``~`` expanded.

    Raises:
        TimeNetRegistryError: If the value is blank, remote or an invalid local registry URI.
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
