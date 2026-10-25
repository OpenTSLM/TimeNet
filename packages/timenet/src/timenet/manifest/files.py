"""Manifest descriptors for the DuckDB control file and values-plane artifacts."""

from functools import partial
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StrictStr, model_validator

from timenet.format.constants import check_relative_path
from timenet.types._model import TimeFModel
from timenet.types.wire import Checksum, NonEmptyString, StrictNonNegativeInt, ValueEncoding
from timenet.values_backends import ValuesBackend


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
    size: StrictNonNegativeInt
    """The size of the file, in bytes."""


ControlParts = Annotated[tuple[FilePart, ...], Field(min_length=1, max_length=1)]
"""The exactly-one control-file tuple."""


class ControlFiles(TimeFModel):
    """The control plane: one DuckDB file."""

    backend: Literal["duckdb"]
    """The backend that wrote the file."""
    parts: ControlParts
    """The single immutable ``control.duckdb`` file."""


class TimeSeriesFiles(TimeFModel):
    """The values plane: the files that hold the time-series values."""

    backend: ValuesBackend
    """The backend that wrote the files."""
    encoding: dict[StrictStr, ValueEncoding]
    """``spec_type`` to the values encoding that its shards use. Empty for Zarr."""
    parts: tuple[FilePart, ...]
    """Parquet shards or files inside the Zarr values store."""

    @model_validator(mode="after")
    def _encoding_needs_parquet(self) -> Self:
        if self.encoding and self.backend != ValuesBackend.PARQUET:
            raise ValueError(f"the {self.backend} backend does not record a values encoding")
        return self


class ManifestFiles(TimeFModel):
    """Every TimeF artifact, keyed by kind. Readers use this data rather than a file glob."""

    control: ControlFiles
    """The control plane."""
    time_series: TimeSeriesFiles
    """The values plane."""

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all artifacts, in a stable order.

        Returns:
            The control file followed by every values-plane artifact.
        """
        return (
            *self.control.parts,
            *self.time_series.parts,
        )

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same order as :meth:`all_files`.

        Returns:
            The path of every file. Use this when you only need to find the files, for example to
            download them.
        """
        return tuple(part.path for part in self.all_files())
