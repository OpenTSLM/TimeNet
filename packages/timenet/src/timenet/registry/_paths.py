"""Guard version-relative manifest paths against directory traversal.

Manifest file paths come from a registry (a hosted service, an object store, or a local
directory), so they are untrusted. A crafted or corrupt manifest can name an absolute path, or one
with ``..`` that resolves outside the version directory. Every place that joins a manifest-declared
relpath onto a local base goes through :func:`safe_version_path`.
"""

from pathlib import Path

from timenet.errors import TimeFFormatError


def safe_version_path(base: Path, relpath: str) -> Path:
    """Join ``relpath`` under ``base``, rejecting any path that escapes ``base``.

    Args:
        base: The directory the file must stay within.
        relpath: The version-relative file path taken from a manifest.

    Returns:
        The joined ``base / relpath``.

    Raises:
        TimeFFormatError: If ``relpath`` is absolute or uses ``..`` to resolve outside ``base``.
    """
    destination = base / relpath
    if not destination.resolve().is_relative_to(base.resolve()):
        raise TimeFFormatError(f"manifest file path {relpath!r} escapes the version directory")
    return destination
