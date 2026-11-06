"""Inspect and clear the cache and the stored releases.

Cache removal requires TimeNet's ``CACHEDIR.TAG`` marker.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import NamedTuple

from timenet.config import is_cache_folder, lies_in
from timenet.errors import TimeFValidationError
from timenet.format.constants import MANIFEST_FILE
from timenet.registry import LocalRegistry
from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class StoredRelease:
    """A stored release with its path and size in bytes."""

    release: DatasetRef
    path: Path
    size_bytes: int


class ClearedCache(NamedTuple):
    """The bytes freed and paths removed or kept by a cache clear."""

    freed: int
    """The number of bytes freed."""
    removed: list[Path]
    """The paths successfully removed."""
    kept: list[Path]
    """The cache paths kept because they lack TimeNet's cache marker."""


class ClearedReleases(NamedTuple):
    """The bytes freed and release folders removed by a release clear."""

    freed: int
    """The number of bytes freed."""
    removed: list[Path]
    """The release folders successfully removed."""


def human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        A string like ``"512 B"``, ``"1.5 KB"``, or ``"3.0 GB"``.
    """
    step = 1024  # bytes per unit step
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < step:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} TB"


def stored_releases(store: LocalRegistry) -> list[StoredRelease]:
    """List stored releases and their sizes in bytes.

    Releases have a ``<dataset_id>/<version>/manifest.json`` file outside hidden or temporary directories.

    Args:
        store: The local registry to inspect.

    Returns:
        The releases, sorted by dataset ID and version.
    """
    root = store.root
    found: list[StoredRelease] = []
    for manifest_path in root.rglob(MANIFEST_FILE):
        version_dir = manifest_path.parent
        parts = version_dir.relative_to(root).parts
        # A release needs a dataset ID before its version, so a manifest at the root or one level below is none.
        if not parts[:-1] or any(part.startswith(".") or ".tmp-" in part for part in parts):
            continue
        try:
            release = DatasetRef.parse(f"{'/'.join(parts[:-1])}@{parts[-1]}")
        except TimeFValidationError:
            continue  # for example a directory whose name is not a version
        found.append(StoredRelease(release, version_dir, _dir_size(version_dir)))
    return sorted(found, key=lambda item: (item.release.dataset_id, item.release.version))


def cache_size(cache_dir: Path) -> int:
    """Measure cached downloads and control databases marked as TimeNet's.

    Only ``downloads`` and ``control`` folders with TimeNet's ``CACHEDIR.TAG`` file count.
    A symlink contributes its own size rather than its target's size.

    Args:
        cache_dir: The cache directory.

    Returns:
        The total size in bytes, or ``0`` if no marked cache folders exist.
    """
    total = 0
    for folder in _cache_folders(cache_dir):
        if is_cache_folder(folder):
            total += folder.lstat().st_size if folder.is_symlink() else _dir_size(folder)
    return total


def clear_cache(cache_dir: Path, *, store: LocalRegistry) -> ClearedCache:
    """Clear cached raw source files and control databases.

    Only folders with TimeNet's ``CACHEDIR.TAG`` marker are removed.
    Removing a symlink keeps its target.

    Args:
        cache_dir: The cache directory.
        store: The store, which must not overlap a cache folder selected for deletion.

    Returns:
        The bytes freed, paths successfully removed and cache paths kept.

    Raises:
        TimeFValidationError: If the store overlaps a cache folder selected for deletion, before any deletion.
        OSError: If a folder cannot be removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (OSError is raised by _delete_and_measure)
    folders = _cache_folders(cache_dir)
    marked = [folder for folder in folders if is_cache_folder(folder)]
    for folder in marked:
        if not folder.is_symlink() and (lies_in(store.root, folder) or lies_in(folder, store.root)):
            raise TimeFValidationError(
                f"the store {store.root} overlaps {folder}, which clear_cache() deletes. Move the store out of "
                "the cache"
            )
    freed = 0
    removed: list[Path] = []
    for folder in marked:
        if folder.is_symlink():
            freed += folder.lstat().st_size
            folder.unlink()  # a symlink goes, and its target stays
        else:
            freed += _delete_and_measure(folder)
        removed.append(folder)
    return ClearedCache(freed, removed, [folder for folder in folders if folder not in marked])


def clear_releases(store: LocalRegistry) -> ClearedReleases:
    """Delete stored releases and the dataset directories left empty.

    Other files in the store stay.

    Args:
        store: The store.

    Returns:
        The bytes freed and the release folders removed.

    Raises:
        OSError: If a folder cannot be removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (OSError is raised by _delete_and_measure)
    freed = 0
    removed: list[Path] = []
    for item in stored_releases(store):
        freed += _delete_and_measure(item.path)
        removed.append(item.path)
        _remove_empty_dataset_dirs(item.path.parent, store.root)
    return ClearedReleases(freed, removed)


def _cache_folders(cache_dir: Path) -> list[Path]:
    """Find existing download and control cache paths.

    Returns:
        The ``downloads`` and ``control`` paths that exist, including symlinks.
    """
    folders = (cache_dir / "downloads", cache_dir / "control")
    return [folder for folder in folders if folder.is_symlink() or folder.exists()]


def _remove_empty_dataset_dirs(directory: Path, root: Path) -> None:
    """Remove empty dataset directories up to but excluding the store root.

    Args:
        directory: The first directory to remove.
        root: The store directory to preserve.
    """
    while root in directory.parents and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def _delete_and_measure(path: Path) -> int:
    """Delete a directory tree and return the bytes it held. It stats each file only once.

    One file pass both measures and unlinks each file. Then :func:`shutil.rmtree` removes the empty
    directory skeleton. This avoids a second walk over every file.

    An unreadable directory or a failed removal raises an ``OSError``.

    Args:
        path: The directory to delete.

    Returns:
        The total size in bytes of the files removed.
    """

    def fail(error: OSError) -> None:
        raise error  # os.walk otherwise skips a directory it cannot read, silently leaving it behind

    freed = 0
    for parent, _dirs, files in os.walk(path, topdown=False, onerror=fail):  # os.walk: Path.walk is 3.12+
        parent_dir = Path(parent)
        for name in files:
            entry = parent_dir / name
            # lstat: a symlink counts as itself, not as its (possibly huge, possibly external) target.
            freed += entry.lstat().st_size
            entry.unlink(missing_ok=True)  # a concurrent build may have removed it between walk and unlink
    shutil.rmtree(path)
    return freed


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of all files under a directory."""
    return sum(entry.lstat().st_size for entry in path.rglob("*") if not entry.is_dir())
