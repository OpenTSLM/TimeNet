"""Inspect and clear the cache and the downloads and builds in the store.

Cache removal requires TimeNet's ``CACHEDIR.TAG`` marker.
"""

from collections.abc import Collection
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
import shutil
from typing import Literal, NamedTuple, get_args

from timenet.config import is_cache_folder, lies_in
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import read_manifest
from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.registry.origin import Origin, read_origin
from timenet.types.references import DatasetRef


ReleaseKind = Literal["download", "build"]
"""The kind of a stored release: a copy from a registry, or a build. A release without a record is a build."""


@dataclass(frozen=True)
class StoredRelease:
    """A stored release with its path, provenance record, and size on disk."""

    release: DatasetRef
    path: Path
    origin: Origin | None
    """The record of how the release reached the store, or ``None`` if it has no record."""

    @cached_property
    def size_bytes(self) -> int:
        """The release's size in bytes, measured once on first access."""
        return _dir_size(self.path)

    @property
    def kind(self) -> ReleaseKind:
        """The release kind: ``"download"`` for a recorded copy, or ``"build"`` otherwise."""
        return "download" if self.origin is not None and self.origin.origin == "copy" else "build"


class KeptRelease(NamedTuple):
    """A release that a clear keeps, because a release that stays needs it."""

    release: DatasetRef
    """The release that stays."""
    needed_by: DatasetRef
    """A release that stays and needs it, as a payload source or as the parent of a build."""


class ReleasesToClear(NamedTuple):
    """The releases that a clear deletes, and the releases of the same kinds that it keeps."""

    releases: list[StoredRelease]
    """The releases to delete, sorted by dataset ID and version."""
    kept: list[KeptRelease]
    """The releases of the cleared kinds that stay, sorted by dataset ID and version."""


class ClearedCache(NamedTuple):
    """The bytes freed and paths removed or kept by a cache clear."""

    freed: int
    """The number of bytes freed."""
    removed: list[Path]
    """The paths successfully removed."""
    kept: list[Path]
    """The cache paths kept because they lack TimeNet's cache marker."""


class ClearedReleases(NamedTuple):
    """The bytes freed, release folders removed and releases kept by a release clear."""

    freed: int
    """The number of bytes freed."""
    removed: list[Path]
    """The release folders successfully removed."""
    kept: list[KeptRelease]
    """The releases of the cleared kinds that stay, because a release that stays needs them."""


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
    """List stored releases with their provenance records and sizes.

    Each entry has a ``manifest.json`` in a valid ``<dataset_id>/<version>`` directory.
    Hidden directories and unfinished writes are excluded.

    Args:
        store: The store.

    Returns:
        The releases, sorted by dataset ID and version.

    Raises:
        TimeFFormatError: If a provenance record is invalid.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by read_origin)
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
        if version_dir != release_dir(root, release):
            continue  # for example 01.0.0, which parses as 1.0.0
        found.append(StoredRelease(release, version_dir, read_origin(store, release)))
    return sorted(found, key=lambda item: (item.release.dataset_id, item.release.version))


def releases_to_clear(store: LocalRegistry, *, kinds: Collection[ReleaseKind]) -> ReleasesToClear:
    """Find the releases of the given kinds that no release that stays needs.

    A release that stays keeps its payload sources, and a build that stays also keeps its parents. This holds
    for each release that stays, also for one that stays only because another one needs it.

    Args:
        store: The store.
        kinds: The kinds of release to delete.

    Returns:
        The releases to delete, and the releases of those kinds that stay with a release that needs each.

    Raises:
        TimeFValidationError: If a dataset or release folder in the store is a symlink and a kind stays.
        TimeFFormatError: If a provenance record or the manifest of a release that stays is damaged.
        OSError: If a store folder cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by _refuse_symlinks and stored_releases)
    if not set(get_args(ReleaseKind)) <= set(kinds) and store.root.is_dir():
        _refuse_symlinks(store.root)
    releases = stored_releases(store)
    held = {item.release: item for item in releases}
    # Walk from each release that stays to the releases that it needs. If every kind goes, nothing stays.
    staying = [item.release for item in releases if item.kind not in kinds]
    seen = set(staying)
    needed_by: dict[DatasetRef, DatasetRef] = {}
    while staying:
        release = staying.pop()
        manifest = read_manifest(held[release].path / MANIFEST_FILE)
        needs = [pinned.release for pinned in manifest.payload_sources.values()]
        if held[release].kind == "build":
            needs.extend(manifest.metadata.parents.values())
        for need in needs:
            if need in held and need not in seen:
                seen.add(need)
                needed_by[need] = release
                staying.append(need)
    return ReleasesToClear(
        [item for item in releases if item.kind in kinds and item.release not in needed_by],
        [KeptRelease(item.release, needed_by[item.release]) for item in releases if item.release in needed_by],
    )


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


def clear_releases(store: LocalRegistry, *, kinds: Collection[ReleaseKind]) -> ClearedReleases:
    """Delete selected releases unless remaining releases need them.

    Each dataset folder that this leaves empty goes too, and other files in the store stay. A release whose
    provenance record changes during the call stays.

    Args:
        store: The store.
        kinds: The kinds of release to delete: ``"download"``, ``"build"`` or both.

    Returns:
        The bytes freed, the release folders removed, and the releases kept with a release that needs each.

    Raises:
        TimeFValidationError: Before any deletion, if a dataset or release folder is a symlink and a kind stays.
        TimeFFormatError: Before any deletion, if a provenance record or a kept release's manifest is damaged.
        OSError: If a folder cannot be read or removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by releases_to_clear and _delete_and_measure)
    releases, kept = releases_to_clear(store, kinds=kinds)
    freed = 0
    removed: list[Path] = []
    for item in releases:
        # A build or a download can change the store after the check, so a changed release stays.
        if item.path.is_symlink() or not _record_unchanged(store, item):
            continue
        freed += _delete_and_measure(item.path)
        removed.append(item.path)
        _remove_empty_dataset_dirs(item.path.parent, store.root)
    return ClearedReleases(freed, removed, kept)


def _record_unchanged(store: LocalRegistry, item: StoredRelease) -> bool:
    """Check whether a release's provenance record still equals the one read before the clear.

    Returns:
        ``True`` if the record is unchanged, or ``False`` if it changed or is invalid.
    """
    try:
        return read_origin(store, item.release) == item.origin
    except TimeFFormatError:
        return False


def _cache_folders(cache_dir: Path) -> list[Path]:
    """Find existing download and control cache paths.

    Returns:
        The ``downloads`` and ``control`` paths that exist, including symlinks.
    """
    folders = (cache_dir / "downloads", cache_dir / "control")
    return [folder for folder in folders if folder.is_symlink() or folder.exists()]


def _refuse_symlinks(root: Path) -> None:
    """Reject symlinks in the store's dataset and release folders.

    Unreadable folders raise ``OSError``.

    Args:
        root: The store's root directory.

    Raises:
        TimeFValidationError: If a dataset or release folder is a symlink.
    """
    dataset_depth = 2  # <org>/<name>: the folders below it are releases

    def fail(error: OSError) -> None:
        raise error  # os.walk otherwise skips a folder that it cannot read

    for parent, dirs, _files in os.walk(root, onerror=fail):
        dirs[:] = [name for name in dirs if not name.startswith(".") and ".tmp-" not in name]
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise TimeFValidationError(
                    f"{Path(parent) / name} is a symlink, so the clear cannot see the releases behind it or the "
                    "releases that they need. Replace the link with its folder, then clear again."
                )
        if len(Path(parent).relative_to(root).parts) == dataset_depth:
            dirs.clear()  # the walk stops at the releases


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
