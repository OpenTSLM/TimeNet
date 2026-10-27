"""Opened dataset handles with payload-source and file-path lookups."""

from dataclasses import dataclass, field

from fsspec import FSMap
from fsspec.implementations.arrow import ArrowFSWrapper
from fsspec.implementations.local import LocalFileSystem
import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError
from timenet.format.constants import check_relative_path
from timenet.manifest import Manifest
from timenet.types.reference_keys import SourceKey


@dataclass(frozen=True)
class DatasetVersion:
    """An opened release: its manifest, filesystem, and its already-opened payload sources."""

    manifest: Manifest
    filesystem: pafs.FileSystem
    root: str
    payload_sources: dict[SourceKey, "DatasetVersion"] = field(default_factory=dict, compare=False, repr=False)
    """The opened release behind each key of ``manifest.payload_sources``. Each was checked against
    its pins when this release was opened, and its own payload sources are not opened."""

    def resolve_payload_source(self, source_key: SourceKey | None) -> "DatasetVersion":
        """Return the opened release that stores a payload file.

        Args:
            source_key: The ``source_key`` of a catalog row, a key of ``manifest.payload_sources``.
                ``None`` means this release stores the file.

        Returns:
            This release or one of its already-opened payload sources. No files are opened here.

        Raises:
            TimeFFormatError: If the catalog names a payload source the manifest does not list.
        """
        if source_key is None:
            return self
        try:
            return self.payload_sources[source_key]
        except KeyError as error:
            raise TimeFFormatError(f"unknown payload source {source_key!r} in {self.manifest.dataset_id}") from error

    def resolve_file_path(self, relative_path: str) -> str:
        """Return a file's filesystem path without checking or downloading the file.

        Args:
            relative_path: A path within this version, such as ``time_series/part-000.parquet``.
                Absolute paths and paths that escape the version directory are rejected.

        Returns:
            The path to use with :attr:`filesystem`.
        """
        check_relative_path("relative_path", relative_path)
        return f"{self.root}/{relative_path}"

    def fsspec_mapper(self, relpath: str) -> FSMap:
        """Return an fsspec mapping for a version-relative directory.

        The mapping uses the prepared filesystem, including its credentials and caching.
        Consumers choose the store's read-only policy when opening it.
        """
        path = self.resolve_file_path(relpath)
        if isinstance(self.filesystem, pafs.LocalFileSystem):
            filesystem = LocalFileSystem()
        elif isinstance(self.filesystem, pafs.PyFileSystem) and isinstance(self.filesystem.handler, pafs.FSSpecHandler):
            filesystem = self.filesystem.handler.fs
        else:
            filesystem = ArrowFSWrapper(self.filesystem)
        return filesystem.get_mapper(path)
