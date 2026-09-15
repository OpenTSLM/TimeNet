"""The :class:`ManifestFiles` block. It holds the file descriptors of a dataset, relative to the version directory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilePart:
    """One data file of a dataset version. It has a path, a checksum, and a byte size in one record.

    The path, checksum, and size stay together in one record. A reader does not need to join a file
    to its digest across two structures. A consumer can check integrity and plan a download from
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
    """Descriptors for every artifact of a dataset version, grouped by kind. Readers use this data, not a glob.

    A version has two artifacts: one control database and the values-plane parts. The values plane
    is a list, so it can shard. The control database is one file, because DuckDB addresses one file.
    Each entry is a :class:`FilePart`, with its own path, checksum, and size.
    """

    control_db: FilePart | None = None
    """The version's control database: its records, series, annotations, tasks and chunk locators."""
    time_series: tuple[FilePart, ...] = ()
    """Parts of the time series data. Each part is also a shard."""

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all artifacts, in a stable order.

        Returns:
            The control database, then the parts of ``time_series``, joined into one tuple.
        """
        control = () if self.control_db is None else (self.control_db,)
        return (*control, *self.time_series)

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same order as :meth:`all_files`.

        Returns:
            The path of every file. Use this when you only need to find the files, for example to
            download them.
        """
        return tuple(part.path for part in self.all_files())
