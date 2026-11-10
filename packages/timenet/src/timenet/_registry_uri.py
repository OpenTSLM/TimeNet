"""Parse registry URLs and local directory paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from timenet.errors import TimeNetRegistryError


TIMENET_REGISTRY_URL = "https://registry.timenet.ai"
"""The hosted registry URL represented by ``timenet://``."""

_REMOTE_SCHEMES = ("timenet://", "http://", "https://", "s3://")

_SCHEMES = "Use an http(s)://, s3://, timenet:// or file:// URL, or a local path."


@dataclass(frozen=True)
class RemoteLocation:
    """The URL of a remote registry and the kind of service behind it."""

    url: str
    """The URL, with ``timenet://`` resolved to the hosted registry."""
    scheme: Literal["http", "s3"]
    """``"s3"`` for an S3 bucket, ``"http"`` for a TimeNet HTTP service."""


def registry_location(uri: str | Path, *, label: str = "registry") -> RemoteLocation | Path:
    """Parse a registry value into the location of a remote registry or the directory of a local one.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        A remote location with ``timenet://`` resolved, or a local path with ``~`` expanded.

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
        return RemoteLocation(TIMENET_REGISTRY_URL, "http")
    if text.startswith("s3://"):
        return RemoteLocation(text, "s3")
    if text.startswith(_REMOTE_SCHEMES):
        return RemoteLocation(text, "http")
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
