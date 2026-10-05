"""Manifest descriptors for the DuckDB control file and values-plane artifacts."""

from functools import partial
from typing import Annotated

from pydantic import AfterValidator, Field

from timenet.format.constants import check_relative_path
from timenet.types._model import TimeFModel
from timenet.types._wire import Checksum, NonEmptyString, NonNegativeInt


_ManifestPath = Annotated[
    NonEmptyString,
    AfterValidator(partial(check_relative_path, "manifest file path")),
]


class FilePart(TimeFModel):
    """One data file of a dataset version. It has a path, a checksum, and a byte size in one record.

    The path, checksum, and size stay together in one record. A reader does not need to join a file
    to its digest across two structures. A consumer can verify integrity and plan a download from
    the manifest alone.
    """

    path: _ManifestPath
    """The version-relative POSIX path to the file."""
    checksum: Checksum
    """The digest of the file, with a ``sha256:`` prefix."""
    size: NonNegativeInt
    """The size of the file, in bytes."""


ControlParts = Annotated[tuple[FilePart, ...], Field(min_length=1, max_length=1)]
"""The exactly-one control-file tuple."""


class ManifestFiles(TimeFModel):
    """Descriptors for every TimeF artifact. Readers use this data rather than a file glob."""

    control: ControlParts
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
