"""Inspect and clear the local TimeNet cache under ``~/.cache/timenet/`` (see :mod:`timenet.config`).

Datasets live in the local registry (curated locally) and the download storage; raw sources land in the
download cache. All three sit under the home directory.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from timenet.config import settings
from timenet.format.constants import MANIFEST_FILE


@dataclass(frozen=True)
class CachedDataset:
    """One cached dataset version on disk."""

    location: str  # "registry" or "storage"
    dataset_id: str
    version: str
    path: Path
    size_bytes: int


def human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        A string like ``"512 B"``, ``"1.5 KB"``, or ``"3.0 GB"``.
    """
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cached_datasets() -> list[CachedDataset]:
    """List every dataset version in the local registry and download storage, with sizes.

    Returns:
        The cached datasets, sorted by (location, id, version).
    """
    config = settings()
    found = _scan(config.registry_path, "registry") + _scan(config.storage_dir, "storage")
    return sorted(found, key=lambda dataset: (dataset.location, dataset.dataset_id, dataset.version))


def raw_cache_size() -> int:
    """Return the size in bytes of the raw download cache directory (0 if absent).

    Returns:
        The total size of ``<TIMENET_CACHE>``.
    """
    cache_dir = settings().cache_dir
    return _dir_size(cache_dir) if cache_dir.is_dir() else 0


def clear_cache(*, include_registry: bool) -> tuple[int, list[Path]]:
    """Delete cached data, returning the bytes freed and the directories removed.

    Always clears the download storage and the raw download cache (both re-fetchable). Only removes the
    local registry (locally curated datasets) when ``include_registry`` is set.

    Args:
        include_registry: Also remove the local registry.

    Returns:
        A ``(bytes_freed, removed_dirs)`` pair. Every directory in ``removed_dirs`` is gone by the time
        this returns: a tree that cannot be fully removed raises an ``OSError`` rather than letting the
        caller report space that was never freed.
    """
    config = settings()
    targets = [config.storage_dir, config.cache_dir]
    if include_registry:
        targets.append(config.registry_path)

    freed = 0
    removed: list[Path] = []
    for target in targets:
        if target.is_dir():
            freed += _delete_and_measure(target)
            removed.append(target)
    return freed, removed


def _delete_and_measure(path: Path) -> int:
    """Delete a directory tree and return the bytes it held, statting each file only once.

    Sizing then :func:`shutil.rmtree` would walk every file twice; here the single file pass both
    measures and unlinks, leaving :func:`shutil.rmtree` only the empty directory skeleton to remove.

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


def _scan(root: Path, location: str) -> list[CachedDataset]:
    """Find every ``<...>/<version>/manifest.json`` under ``root`` and measure its version directory.

    Returns:
        The cached datasets found under ``root``.
    """
    if not root.is_dir():
        return []
    datasets: list[CachedDataset] = []
    for manifest_path in root.rglob(MANIFEST_FILE):
        version_dir = manifest_path.parent
        parts = version_dir.relative_to(root).parts
        if any(part.startswith(".") or ".tmp-" in part for part in parts):
            continue
        dataset_id = "/".join(parts[:-1])
        if not dataset_id:
            continue
        datasets.append(CachedDataset(location, dataset_id, parts[-1], version_dir, _dir_size(version_dir)))
    return datasets


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of all files under a directory."""
    return sum(entry.lstat().st_size for entry in path.rglob("*") if not entry.is_dir())
