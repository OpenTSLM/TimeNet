"""Inspect and clear the cache and the stored releases.

Cache removal requires TimeNet's ``CACHEDIR.TAG`` marker.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from timenet._fs import delete_and_measure, lies_in
from timenet.config import TimeNetSettings, is_cache_folder
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class StoredRelease:
    """A stored release with its path and size in bytes."""

    release: DatasetRef
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class ClearedCache:
    """The bytes freed and paths removed or kept by a cache clear."""

    freed: int
    """The number of bytes freed."""
    removed: list[Path]
    """The paths successfully removed."""
    kept: list[Path]
    """The cache paths kept because they lack TimeNet's cache marker."""


@dataclass(frozen=True)
class ClearedReleases:
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
    """List the releases of a local registry and their sizes in bytes.

    Args:
        store: The local registry to inspect.

    Returns:
        The releases, sorted by dataset ID and version.
    """
    found: list[StoredRelease] = []
    for release in store.releases():
        path = release_dir(store.root, release)
        found.append(StoredRelease(release, path, _dir_size(path)))
    return found


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


def clear_cache(cache_dir: Path, *, local_registry: LocalRegistry) -> ClearedCache:
    """Clear cached raw source files and control databases.

    Only folders with TimeNet's ``CACHEDIR.TAG`` marker are removed.
    Removing a symlink keeps its target.

    Args:
        cache_dir: The cache directory.
        local_registry: The local registry, which must not overlap a cache folder selected for deletion.

    Returns:
        The bytes freed, paths successfully removed and cache paths kept.

    Raises:
        TimeFValidationError: Before any deletion, if the cache is the home folder or a folder above it, or the local
            registry overlaps a cache folder selected for deletion.
        OSError: If a folder cannot be removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _check_cache_deletion and delete_and_measure)
    folders = _cache_folders(cache_dir)
    marked = [folder for folder in folders if is_cache_folder(folder)]
    if marked:
        # A symlink goes and its target stays, so only the other folders can hold the local registry.
        deleted = [folder for folder in marked if not folder.is_symlink()]
        _check_cache_deletion(cache_dir, deleted, local_registry=local_registry.root)
    freed = 0
    removed: list[Path] = []
    for folder in marked:
        if folder.is_symlink():
            freed += folder.lstat().st_size
            folder.unlink()  # a symlink goes, and its target stays
        else:
            freed += delete_and_measure(folder)
        removed.append(folder)
    return ClearedCache(freed, removed, [folder for folder in folders if folder not in marked])


def clear_releases(store: LocalRegistry) -> ClearedReleases:
    """Delete stored releases and the dataset directories left empty.

    Other files in the local registry stay.

    Args:
        store: The local registry.

    Returns:
        The bytes freed and the release folders removed.

    Raises:
        TimeNetRegistryError: If the local registry is read-only.
        OSError: If a folder cannot be removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by remove_release)
    freed = 0
    removed: list[Path] = []
    for release in store.releases():
        freed += store.remove_release(release)
        removed.append(release_dir(store.root, release))
    return ClearedReleases(freed, removed)


def _check_cache_deletion(cache_dir: Path, folders: Iterable[Path], *, local_registry: Path) -> None:
    """Check that TimeNet may delete folders of the cache.

    Args:
        cache_dir: The cache directory.
        folders: The folders of the cache to delete.
        local_registry: The local registry directory, which must stay.

    Raises:
        TimeFValidationError: If the cache is the home folder or a folder above it, or a folder and the local registry
            overlap.
    """
    home = _home()
    if home is not None and lies_in(home, cache_dir):
        raise TimeFValidationError(
            f"The cache {cache_dir} is the home folder or a folder above it. TimeNet deletes the downloads and control "
            "folders of the cache, so give TIMENET_CACHE a folder of its own, for example "
            f"{TimeNetSettings.model_fields['cache'].default}."
        )
    for folder in folders:
        if lies_in(local_registry, folder) or lies_in(folder, local_registry):
            raise TimeFValidationError(
                f"the local registry {local_registry} overlaps {folder}, which TimeNet deletes. Move the local "
                "registry out of the cache"
            )


def _home() -> Path | None:
    """Find the user's home directory.

    Returns:
        The home directory, or ``None`` if the system cannot find it.
    """
    try:
        return Path.home()
    except RuntimeError:
        return None


def _cache_folders(cache_dir: Path) -> list[Path]:
    """Find existing download and control cache paths.

    Returns:
        The ``downloads`` and ``control`` paths that exist, including symlinks.
    """
    folders = (cache_dir / "downloads", cache_dir / "control")
    return [folder for folder in folders if folder.is_symlink() or folder.exists()]


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of all files under a directory."""
    return sum(entry.lstat().st_size for entry in path.rglob("*") if not entry.is_dir())
