"""Manifest descriptors for every file of a version, grouped by kind and backend."""

from dataclasses import dataclass, field
from enum import StrEnum

from timenet.errors import TimeNetInvalidManifestError
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS


CONTROL_BACKEND = "duckdb"
"""The backend of the control group. The control plane is always one DuckDB file."""


class FileKind(StrEnum):
    """What the files of a :class:`FileGroup` hold. Each member value is the on-disk ``kind`` tag."""

    CONTROL = "control"
    TIME_SERIES = "time_series"


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
class FileGroup:
    """The files of one kind, written by one backend.

    Raises:
        TimeNetInvalidManifestError: If the backend does not match the kind, or a control group does
            not have exactly one file.
    """

    kind: FileKind
    """What the files hold."""
    backend: str
    """The backend that wrote the files: ``duckdb`` for control, ``parquet`` or ``zarr`` for values."""
    parts: tuple[FilePart, ...] = ()
    """The files of the group."""
    encoding: dict[str, str] = field(default_factory=dict)
    """``spec_type`` -> the values encoding that its shards carry.

    Provenance only. Parquet records the applied encoding in the footer of each file, so a reader
    does not need this field. The field is empty for a backend with no such choice.
    """

    def __post_init__(self) -> None:
        """Validate the backend against the kind, and the file count of a control group.

        Raises:
            TimeNetInvalidManifestError: If the backend does not match the kind, or a control group
                does not have exactly one file.
        """
        if self.kind == FileKind.CONTROL:
            if self.backend != CONTROL_BACKEND:
                raise TimeNetInvalidManifestError(
                    f"control file group backend must be {CONTROL_BACKEND!r}, got {self.backend!r}"
                )
            if len(self.parts) != 1:
                raise TimeNetInvalidManifestError("TimeF manifest must declare exactly one control file")
        elif self.backend not in SUPPORTED_VALUES_BACKENDS:
            raise TimeNetInvalidManifestError(
                f"unsupported {self.kind} backend {self.backend!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
            )


@dataclass(frozen=True)
class ManifestFiles:
    """Every file of a version, grouped by kind. Readers use this data rather than a file glob.

    Raises:
        TimeNetInvalidManifestError: If a kind occurs more than once, or the control group is missing.
    """

    groups: tuple[FileGroup, ...]
    """One group per kind. The control group is required."""

    def __post_init__(self) -> None:
        """Validate that each kind occurs once and that the control group is present.

        Raises:
            TimeNetInvalidManifestError: If a kind occurs more than once, or the control group is
                missing.
        """
        kinds = [group.kind for group in self.groups]
        duplicates = sorted({kind for kind in kinds if kinds.count(kind) > 1})
        if duplicates:
            raise TimeNetInvalidManifestError(f"duplicate file group kind: {', '.join(duplicates)}")
        if FileKind.CONTROL not in kinds:
            raise TimeNetInvalidManifestError("TimeF manifest must declare a control file group")

    @property
    def control(self) -> FilePart:
        """Return the single ``control.duckdb`` file."""
        return next(group.parts[0] for group in self.groups if group.kind == FileKind.CONTROL)

    def group(self, kind: FileKind) -> FileGroup | None:
        """Return the group of one kind.

        Args:
            kind: The kind to find.

        Returns:
            The group, or ``None`` if the version has no files of this kind.
        """
        return next((group for group in self.groups if group.kind == kind), None)

    def all_files(self) -> tuple[FilePart, ...]:
        """Return every file descriptor across all groups, in a stable order.

        Returns:
            The files of each group, in the order of :attr:`groups`.
        """
        return tuple(part for group in self.groups for part in group.parts)

    def all_parts(self) -> tuple[str, ...]:
        """Return the version-relative path of every file, in the same order as :meth:`all_files`.

        Returns:
            The path of every file. Use this when you only need to find the files, for example to
            download them.
        """
        return tuple(part.path for part in self.all_files())
