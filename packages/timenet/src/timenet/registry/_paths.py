"""Build registry release paths and guard version-relative manifest paths.

Manifest file paths come from a registry (a hosted service, an object store, or a local
directory), so they are untrusted. A crafted or corrupt manifest can name an absolute path, or one
with ``..`` that resolves outside the version directory. Every place that joins a manifest-declared
relpath onto a local base goes through :func:`safe_version_path`.
"""

from pathlib import Path

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types.references import DatasetRef


def release_dir(root: Path, ref: DatasetRef) -> Path:
    """Return the registry directory for an exact release."""
    return root / ref.dataset_id / str(ref.version)


def split_release_dir(path: Path) -> tuple[Path, DatasetRef]:
    """Return a release directory's registry root and exact reference.

    Raises:
        TimeFValidationError: If the path does not end in ``<org>/<name>/<version>``.
    """
    directory = path.expanduser().resolve()
    try:
        ref = DatasetRef.parse(f"{directory.parent.parent.name}/{directory.parent.name}@{directory.name}")
    except TimeFValidationError as error:
        raise TimeFValidationError(f"release directory {path} must end in <org>/<name>/<version>: {error}") from error
    return directory.parents[2], ref


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
