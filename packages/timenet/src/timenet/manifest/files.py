"""Manifest descriptors for the DuckDB control file and values-plane artifacts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilePart:
    """One data file of a dataset version. It has a path, a checksum, and a byte size in one record.

    The path, checksum, and size stay together in one record. A reader does not need to join a file
    to its digest across two structures. A consumer can verify integrity and plan a download from
    the manifest alone.
    """

    path: str
    """The version-relative POSIX path to the file."""
    checksum: str
    """The digest of the file, with a ``sha256:`` prefix."""
    size: int
    """The size of the file, in bytes."""


@dataclass(frozen=True)
class ManifestFiles:
    """Descriptors for every TimeF artifact. Readers use this data rather than a file glob."""

    control: tuple[FilePart, ...]
    """The single immutable ``control.duckdb`` file."""
    time_series: tuple[FilePart, ...] = ()
    """Parquet shards or files inside the Zarr values store."""

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all artifacts, in a stable order.

        Returns:
            The control file followed by every values-plane artifact.
        """
        return (
            *self.control,
            *self.time_series,
        )

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same order as :meth:`all_files`.

        Returns:
            The path of every file. Use this when you only need to find the files, for example to
            download them.
        """
        return tuple(part.path for part in self.all_files())
