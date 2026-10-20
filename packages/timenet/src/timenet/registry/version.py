"""Prepared dataset handles with simple payload-source and file-path lookups."""

from dataclasses import dataclass, field

import pyarrow.fs as pafs

from timenet.errors import TimeFFormatError
from timenet.format.constants import check_relative_path
from timenet.manifest import Manifest


@dataclass(frozen=True)
class DatasetVersion:
    """A prepared release: its manifest, filesystem, and already-opened payload sources."""

    manifest: Manifest
    filesystem: pafs.FileSystem
    root: str
    payload_sources: dict[str, "DatasetVersion"] = field(default_factory=dict, compare=False, repr=False)

    def resolve_payload_source(self, key: str | None) -> "DatasetVersion":
        """Return the prepared version that stores a chunk's bytes.

        Args:
            key: The chunk's ``owner_key``, naming an entry in ``manifest.payload_sources``.
                ``None`` means the bytes belong to this version.

        Returns:
            This version or an already-prepared payload source. No files are opened here.

        Raises:
            TimeFFormatError: If the catalog refers to an unknown payload source.
        """
        if key is None:
            return self
        try:
            return self.payload_sources[key]
        except KeyError as error:
            raise TimeFFormatError(f"unknown payload source {key!r} in {self.manifest.dataset_id}") from error

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

    def store_uri(self, relpath: str) -> str:
        """Return a location for a store-oriented backend such as Zarr.

        Args:
            relpath: A path relative to the version root.

        Returns:
            The local path, or a scheme-qualified URI for an S3 filesystem.
        """
        path = self.resolve_file_path(relpath)
        return f"s3://{path}" if isinstance(self.filesystem, pafs.S3FileSystem) else path
