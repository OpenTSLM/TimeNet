"""Define the writable-registry contract for a backend that build publishes datasets into.

This class extends the read-only :class:`~timenet.registry.BaseRegistry` with two write primitives,
:meth:`WritableRegistry.put_dataset` and :meth:`WritableRegistry.put_release`. Backends differ only in where a
release lands, for example a local directory, an S3 prefix, or a remote service. So these two are the
abstract methods, and :meth:`exists` is shared.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

import pyarrow.fs as pafs

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetBuildError
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest, PayloadSource, read_manifest
from timenet.provenance import TimeNetSnapshot
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_file_sizes
from timenet.registry.version import DatasetVersion
from timenet.types.references import DatasetRef
from timenet.writer import WriteProgressEvent


class WritableRegistry(BaseRegistry, ABC):
    """A registry that can also publish datasets and release directories."""

    @abstractmethod
    def put_dataset(
        self,
        dataset: TimeFDataset,
        *,
        force: bool = False,
        values_backend: str = "parquet",
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> DatasetRef:
        """Write a dataset to this registry.

        This method derives a schema if the dataset has none. It returns an existing version's
        reference unless ``force`` is set.

        Args:
            dataset: The dataset to write.
            force: Replace an existing version.
            values_backend: Storage format for values (``"parquet"`` or ``"zarr"``).
            progress_cb: Optional callback for write progress.
            pinned_by: Releases that already read this release's files, with the pins that the new files must match.

        Returns:
            A reference to the stored release.
        """

    @abstractmethod
    def put_release(self, from_directory: Path) -> DatasetRef:
        """Publish an existing release to this registry.

        The directory must contain ``manifest.json`` and every file it lists. The method publishes only
        those files and leaves the directory unchanged.

        Args:
            from_directory: The release directory.

        Returns:
            A reference to the published release.

        Raises:
            TimeFFormatError: If ``manifest.json`` is missing, unreadable, or invalid.
            TimeNetFileError: If a listed file is missing or its size differs from the manifest.
            TimeNetBuildError: If this registry refuses releases built from locally modified code, and the manifest records local changes.
        """

    @property
    def requires_clean_code_provenance(self) -> bool:
        """Whether this registry refuses releases built from locally modified TimeNet code."""
        return True

    def require_clean_code_provenance(self, manifest: Manifest) -> None:
        """Raise ``TimeNetBuildError`` if this registry refuses releases built from locally modified code, and ``manifest`` records local changes.

        Args:
            manifest: The manifest of the release to publish.

        Raises:
            TimeNetBuildError: If the manifest records local changes or no commit for a TimeNet package.
        """
        if not self.requires_clean_code_provenance:
            return
        modified = TimeNetSnapshot.from_source_map(manifest.build_env.git).packages_with_local_changes()
        if modified:
            reference = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
            changes = ", ".join(f"{package.package_name} (commit {package.source.sha[:12]})" for package in modified)
            raise TimeNetBuildError(
                f"{self} does not take releases built from locally modified TimeNet code, so it refuses {reference}: "
                f"it records local changes in {changes}. Commit the changes and build the release again, or keep it "
                "in a local registry."
            )

    def _read_release(self, directory: Path) -> Manifest:
        """Read a release manifest, check the listed file sizes, and refuse a release that this registry does not take.

        Args:
            directory: The release directory.

        Returns:
            The release manifest.

        Raises:
            TimeFFormatError: If ``manifest.json`` is missing, unreadable, or invalid.
            TimeNetFileError: If a listed file is missing or its size differs from the manifest.
            TimeNetBuildError: If this registry refuses releases built from locally modified code, and the manifest records local changes.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by read_manifest, check_file_sizes and require_clean_code_provenance
        manifest = read_manifest(directory / MANIFEST_FILE)
        self.require_clean_code_provenance(manifest)
        reference = DatasetRef(dataset_id=manifest.dataset_id, version=manifest.metadata.dataset_version)
        version = DatasetVersion(manifest, pafs.LocalFileSystem(), str(directory))
        check_file_sizes(version, manifest.files.all_files(), owner=str(reference))
        return manifest

    def exists(self, dataset_id: str, version: str) -> bool:
        """Check whether this registry holds a dataset version.

        Args:
            dataset_id: The dataset id.
            version: The version string.

        Returns:
            ``True`` if the version has a committed manifest, else ``False``.
        """
        return self.find_manifest(DatasetRef.parse(f"{dataset_id}@{version}")) is not None
