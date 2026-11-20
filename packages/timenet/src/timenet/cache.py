"""Inspect and clear the cache and the downloads and builds in the local registry.

Cache removal requires TimeNet's ``CACHEDIR.TAG`` marker.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

from timenet._fs import delete_and_measure, lies_in
from timenet.config import TimeNetSettings, is_cache_folder
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.registry.origin import Origin, OriginKind
from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class StoredRelease:
    """A stored release with its path, provenance record, and size on disk."""

    release: DatasetRef
    path: Path
    origin: Origin | None
    """The record of how the release reached the local registry, or ``None`` if it has no record."""

    def size(self) -> int:
        """Measure the size of the release on disk.

        Returns:
            The size in bytes, measured on each call.
        """
        return _dir_size(self.path)

    @property
    def kind(self) -> OriginKind:
        """The kind of the record, or :attr:`~OriginKind.BUILD` for a release without a record."""
        return OriginKind.BUILD if self.origin is None else self.origin.kind


@dataclass(frozen=True)
class KeptRelease:
    """A release that a clear keeps, because a release that stays needs it."""

    release: DatasetRef
    """The release that stays."""
    needed_by: DatasetRef
    """A release that stays and needs it, as a payload source or as the parent of a build."""


@dataclass(frozen=True)
class ReleasesToClear:
    """The releases that a clear deletes, and the releases of the same kinds that it keeps."""

    releases: list[StoredRelease]
    """The releases to delete, sorted by dataset ID and version."""
    kept: list[KeptRelease]
    """The releases of the cleared kinds that stay, sorted by dataset ID and version."""


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

    Args:
        store: The local registry.

    Returns:
        The releases, sorted by dataset ID and version.

    Raises:
        TimeFFormatError: If a provenance record is invalid.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by read_origin)
    return [
        StoredRelease(release, release_dir(store.root, release), store.read_origin(release))
        for release in store.releases()
    ]


def stored_releases_by_kind(store: LocalRegistry) -> dict[OriginKind, list[StoredRelease]]:
    """List stored releases with their provenance records and sizes, by kind.

    Args:
        store: The local registry.

    Returns:
        The releases of each kind, sorted by dataset ID and version, and an empty list for a kind without releases.

    Raises:
        TimeFFormatError: If a provenance record is invalid.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by read_origin)
    by_kind: dict[OriginKind, list[StoredRelease]] = {kind: [] for kind in OriginKind}
    for item in stored_releases(store):
        by_kind[item.kind].append(item)
    return by_kind


def releases_to_clear(store: LocalRegistry, *, kinds: Collection[OriginKind]) -> ReleasesToClear:
    """Find the releases of the given kinds that no release that stays needs.

    A release that stays keeps its payload sources, and a build that stays also keeps its parents. This holds
    for each release that stays, also for one that stays only because another one needs it.

    Args:
        store: The local registry.
        kinds: The kinds of release to delete.

    Returns:
        The releases to delete, and the releases of those kinds that stay with a release that needs each.

    Raises:
        TimeFValidationError: If ``kinds`` is empty, or a dataset or release folder in the local registry is a symlink.
        TimeFFormatError: If a provenance record or the manifest of a release that stays is damaged.
        OSError: If a folder of the local registry cannot be read.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by check_no_linked_folders and stored_releases)
    _check_kinds(kinds)
    store.check_no_linked_folders()
    releases = stored_releases(store)
    held = {item.release: item for item in releases}
    # Walk from each release that stays to the releases that it needs. If every kind goes, nothing stays.
    staying = [item.release for item in releases if item.kind not in kinds]
    seen = set(staying)
    needed_by: dict[DatasetRef, DatasetRef] = {}
    while staying:
        release = staying.pop()
        manifest = store.get_manifest(release.dataset_id, str(release.version))
        for need in manifest.required_releases(with_parents=held[release].kind is OriginKind.BUILD):
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


def clear_releases(store: LocalRegistry, *, kinds: Collection[OriginKind]) -> ClearedReleases:
    """Delete selected releases unless remaining releases need them.

    Each dataset folder that this leaves empty goes too, and other files in the local registry stay.

    Args:
        store: The local registry.
        kinds: The kinds of release to delete.

    Returns:
        The bytes freed, the release folders removed, and the releases kept with a release that needs each.

    Raises:
        TimeNetRegistryError: Before any deletion, if the local registry is read-only.
        TimeFValidationError: Before any deletion, if ``kinds`` is empty or a dataset or release folder is a symlink.
        TimeFFormatError: Before any deletion, if a provenance record or a kept release's manifest is damaged.
        OSError: If a folder cannot be read or removed completely.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by releases_to_clear and remove_release)
    _check_kinds(kinds)
    store.check_writable("a clear of its releases")
    found = releases_to_clear(store, kinds=kinds)
    freed = 0
    removed: list[Path] = []
    for item in found.releases:
        freed += store.remove_release(item.release)
        removed.append(item.path)
    return ClearedReleases(freed, removed, found.kept)


def _check_kinds(kinds: Collection[OriginKind]) -> None:
    """Refuse a clear that names no kind of release.

    Raises:
        TimeFValidationError: If ``kinds`` is empty.
    """
    if not kinds:
        raise TimeFValidationError("kinds names no kind of release")


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
