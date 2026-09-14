"""The :class:`ManifestFiles` block. It holds the file descriptors of a dataset, relative to the version directory."""

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
    """Descriptors for every artifact of a dataset version, grouped by kind. Readers use this data, not a glob.

    Each artifact is a list of parts. This lets any artifact shard later without a change to the
    manifest format. The writer fills ``control_db`` with the control database and ``time_series``
    with the values plane's parts; the four Parquet control-table tuples that the database replaces
    stay empty. Each part is a :class:`FilePart` object, with its own path, checksum, and size.
    """

    records: tuple[FilePart, ...] = ()
    """Parts of the records table."""
    annotations: tuple[FilePart, ...] = ()
    """Parts of the annotations table."""
    time_series_index: tuple[FilePart, ...] = ()
    """Parts of the time series index table."""
    tasks: tuple[FilePart, ...] = ()
    """Parts of the task tables. There is one table for each task type."""
    time_series: tuple[FilePart, ...] = ()
    """Parts of the time series data. Each part is also a shard."""
    control_db: FilePart | None = None
    """The DuckDB control-plane database, when the version stores its control plane that way.

    A version carries either the sharded control tables above or this single file, never both. The
    file holds every control-plane entity (records, sources, signals, tasks, annotations) in one
    embedded database. It is one part, not a tuple, because the database does its own paging: it
    never shards.
    """

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all artifacts, in a stable order.

        Returns:
            The parts of ``records``, ``annotations``, ``time_series_index``, ``tasks``,
            ``control_db``, and ``time_series``, joined into one tuple.
        """
        control = (self.control_db,) if self.control_db is not None else ()
        return (*self.records, *self.annotations, *self.time_series_index, *self.tasks, *control, *self.time_series)

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same order as :meth:`all_files`.

        Returns:
            The path of every file. Use this when you only need to find the files, for example to
            download them.
        """
        return tuple(part.path for part in self.all_files())
