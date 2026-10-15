"""Fetch pinned physical files without traversing parent catalogs."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError
from timenet.format.checksums import file_checksum
from timenet.manifest import Manifest
from timenet.registry._paths import safe_version_path
from timenet.registry.version import DatasetVersion
from timenet.types.references import manifest_digest


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


def complete_copy(root: Path, manifest: Manifest) -> bool:
    """Return whether a cached release has its manifest and every declared file."""
    path = root / "manifest.json"
    return (
        path.is_file()
        and manifest_digest(Manifest.model_validate_json(path.read_text()).to_dict())
        == manifest_digest(manifest.to_dict())
        and all(
            (file := safe_version_path(root, part.path)).is_file() and file.stat().st_size == part.size
            for part in manifest.files.all_files()
        )
    )


def validate_owners(registry: "BaseRegistry", manifest: Manifest) -> dict[str, Manifest]:
    """Check that physical owners are available with exactly the pinned manifests.

    Returns:
        The validated owner manifests keyed by physical owner.

    Raises:
        TimeFFormatError: If an owner changed or a referenced file differs from its manifest.
    """
    owners = {}
    for key, source in manifest.payload_sources.items():
        owner = registry.get_manifest(source.source.dataset_id, str(source.source.version))
        if manifest_digest(owner.to_dict()) != source.manifest_checksum:
            raise TimeFFormatError(f"manifest changed for payload owner {source.source}")
        parts = {part.path: part for part in owner.files.all_files()}
        for part in source.files:
            if parts.get(part.path) != part:
                raise TimeFFormatError(f"invalid physical file {part.path!r} for {source.source}")
        owners[key] = owner
    return owners


def download_owners(
    registry: "BaseRegistry", manifest: Manifest, root: Path, *, progress_cb: Callable[[int], None] | None = None
) -> None:
    """Cache only referenced physical files and their manifests for offline use.

    Owner catalogs and unrelated payload files are not required. A later full download
    detects this partial cache and fetches the remaining files.
    """
    owners = validate_owners(registry, manifest)
    for key, source in manifest.payload_sources.items():
        owner = owners[key]
        directory = root / source.source.dataset_id / str(source.source.version)
        version = DatasetVersion(owner, pafs.LocalFileSystem(), str(directory), registry=registry)
        for part in source.files:
            path = safe_version_path(directory, part.path)
            if path.exists() and (path.stat().st_size != part.size or file_checksum(path) != part.checksum):
                path.unlink()
            already_present = path.exists()
            version.path(part.path)
            if progress_cb is not None and not already_present:
                progress_cb(part.size)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "manifest.json.part"
        temporary.write_text(owner.to_json())
        temporary.replace(directory / "manifest.json")
