"""Resolve a direct local root with pinned parents from a registry."""

from typing import BinaryIO

from timenet.manifest import Manifest
from timenet.registry import BaseRegistry, DatasetVersion
from timenet.types import DatasetMetadata


class PathRegistry(BaseRegistry):
    """Resolve one direct root and delegate pinned dependency lookups.

    Enumeration intentionally exposes only the selected root. This viewer launch adapter
    does not enumerate the parent registry, which may be remote or not support listing.
    It is not a merged catalog of both registries.
    """

    def __init__(self, root: DatasetVersion, parents: BaseRegistry) -> None:
        self.root, self.parents = root, parents

    def _is_root(self, dataset_id: str, version: str | None) -> bool:
        return dataset_id == self.root.manifest.dataset_id and (
            version is None or version == str(self.root.manifest.metadata.dataset_version)
        )

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return the direct root's metadata."""
        return [self.root.manifest.metadata]

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return the root manifest or delegate a pinned parent lookup."""
        return (
            self.root.manifest if self._is_root(dataset_id, version) else self.parents.get_manifest(dataset_id, version)
        )

    def open_file(self, dataset_id: str, version: str, relpath: str) -> BinaryIO:
        """Return a root file stream or a parent registry stream."""
        if self._is_root(dataset_id, version):
            return self.root.filesystem.open_input_file(self.root.path(relpath))
        return self.parents.open_file(dataset_id, version, relpath)

    def open_version(self, dataset_id: str, version: str | None = None) -> DatasetVersion:
        """Return the direct root handle or a parent registry handle."""
        return self.root if self._is_root(dataset_id, version) else self.parents.open_version(dataset_id, version)
