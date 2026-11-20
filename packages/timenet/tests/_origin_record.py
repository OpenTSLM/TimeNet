"""The file of a release's provenance record, for tests that damage, delete or read it."""

from pathlib import Path

from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.registry.local import _ORIGIN_FILE
from timenet.types.references import DatasetRef


def origin_file(store: LocalRegistry, release: DatasetRef) -> Path:
    """Return the file of the provenance record of a release in a local registry."""
    return release_dir(store.root, release) / _ORIGIN_FILE
