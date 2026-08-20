"""The :class:`ManifestFiles` block: the dataset's file descriptors, relative to the version directory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilePart:
    """One data file of a dataset version: its path, checksum, and byte size in a single record.

    Path, checksum, and size travel together so a reader never joins a file to its digest across two
    structures, and so a consumer can verify integrity and plan a download from the manifest alone.
    """

    path: str
    """Version-relative POSIX path to the file."""
    checksum: str
    """The file's digest, ``sha256:`` prefixed."""
    size: int
    """The file's size in bytes."""


@dataclass(frozen=True)
class ManifestFiles:
    """Descriptors for every artifact of a dataset version, grouped by kind. Readers use this, not a glob.

    Every artifact is a list of parts, so any of them can shard later without a manifest-format change.
    Today the writer emits a single part for ``samples`` / ``annotations`` / ``time_series_index``;
    ``tasks`` and ``time_series`` already carry several. Each part is a :class:`FilePart` carrying its
    own path, checksum, and size.
    """

    samples: tuple[FilePart, ...]
    """Parts of the samples table."""
    annotations: tuple[FilePart, ...]
    """Parts of the annotations table."""
    time_series_index: tuple[FilePart, ...]
    """Parts of the time series index table."""
    tasks: tuple[FilePart, ...] = ()
    """Parts of the task tables, one per task type."""
    time_series: tuple[FilePart, ...] = ()
    """Parts (shards) of the time series data."""

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all artifacts, in a stable order.

        Returns:
            The concatenation of the ``samples``, ``annotations``, ``time_series_index``, ``tasks``,
            and ``time_series`` parts.
        """
        return (*self.samples, *self.annotations, *self.time_series_index, *self.tasks, *self.time_series)

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same stable order as :meth:`all_files`.

        Returns:
            Every file's path, for callers that only need to locate the files (e.g. a download).
        """
        return tuple(part.path for part in self.all_files())
