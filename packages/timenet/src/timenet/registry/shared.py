"""Release dependencies and local cache completeness."""

from pathlib import Path
from typing import TYPE_CHECKING

from timenet.manifest import Manifest
from timenet.types.references import DatasetRef


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


def complete_copy(root: Path, manifest: Manifest) -> bool:
    """Return whether a release has its manifest and every declared file."""
    return (root / "manifest.json").is_file() and all(
        (file := root / part.path).is_file() and file.stat().st_size == part.size for part in manifest.files.all_files()
    )


def dependency_references(manifest: Manifest) -> tuple[DatasetRef, ...]:
    """Return the build inputs and physical sources required in the publishing registry.

    Returns:
        The dependencies with duplicate references removed.
    """
    references = [*(card.source for card in manifest.parent_cards.values()), *manifest.payload_sources.values()]
    return tuple({str(ref): ref for ref in references}.values())


def validate_dependencies(registry: "BaseRegistry", manifest: Manifest) -> None:
    """Check that each dependency exists in the registry before publishing a child."""
    for reference in dependency_references(manifest):
        registry.get_manifest(reference.dataset_id, str(reference.version))
