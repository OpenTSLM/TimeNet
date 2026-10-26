"""The TRACE archive as published: its pinned download, its split files, and their pinned counts."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import Split


# The README's Google Drive link, as a direct download that skips the HTML confirmation page. The
# archive is 510,759,681 bytes (487 MiB) and unpacks to 1.75 GB.
SOURCE_URL = (
    "https://drive.usercontent.google.com/download?id=1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY&export=download&confirm=t"
)
# Drive has no immutable revision, so pin the archive's bytes instead.
SOURCE_SHA256 = "a5d24d75a0b9f0f846a185caa46c4f77e0ef98272c12808df76fac25a181be58"


@unique
class TraceTaskKind(StrEnum):
    """The authors structure the data in three different kinds that define what rows hold.

    ``pretrain`` and ``forecasting`` files hold windows; ``pretrain`` windows add their event label.
    ``retrieval`` files hold no windows: each row gives one window's description and event report.
    """

    PRETRAIN = "pretrain"
    FORECASTING = "forecasting"
    RETRIEVAL = "retrieval"

    @property
    def holds_windows(self) -> bool:
        """Return whether the folder's files hold windows, rather than text about windows."""
        return self is not TraceTaskKind.RETRIEVAL

    @property
    def file_extension(self) -> str:
        """Return the extension of the folder's split files: JSON windows, or Stage 2 parquet."""
        return "json" if self.holds_windows else "parquet"

    def file_path(self, split: "TraceSplit") -> str:
        """Return the path of one of the folder's split files, below the archive root."""
        return f"dataset/{self}/{split}.{self.file_extension}"


@unique
class TraceSplit(StrEnum):
    """The authors' split name, as a split file is named, with the TimeNet split it maps to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the split file's rows."""
        return _TIMENET_SPLITS[self]


_TIMENET_SPLITS: dict[TraceSplit, Split] = {
    TraceSplit.TRAIN: Split.TRAIN,
    TraceSplit.VAL: Split.VALIDATION,
    TraceSplit.TEST: Split.TEST,
}


class PinnedCounts(NamedTuple):
    """One source file's pinned counts."""

    rows: int
    tasks: int
    """Classification tasks (``pretrain``), forecasting tasks (``forecasting``), or rows with an event
    report (``retrieval``)."""


# The archive's split files and their pinned counts, keyed by (task kind, split). The JSON row counts
# match Table 8 of the paper, and the pretrain task counts match classification/*_data/labels.npy.
SOURCE_FILES = {
    (TraceTaskKind.PRETRAIN, TraceSplit.TRAIN): PinnedCounts(45339, 10110),
    (TraceTaskKind.PRETRAIN, TraceSplit.VAL): PinnedCounts(12953, 2884),
    (TraceTaskKind.PRETRAIN, TraceSplit.TEST): PinnedCounts(6484, 1452),
    (TraceTaskKind.FORECASTING, TraceSplit.TRAIN): PinnedCounts(6690, 6627),
    (TraceTaskKind.FORECASTING, TraceSplit.VAL): PinnedCounts(1914, 1900),
    (TraceTaskKind.FORECASTING, TraceSplit.TEST): PinnedCounts(957, 947),
    (TraceTaskKind.RETRIEVAL, TraceSplit.TRAIN): PinnedCounts(42180, 13787),
    (TraceTaskKind.RETRIEVAL, TraceSplit.TEST): PinnedCounts(2006, 659),
}
# Distinct measurement windows among the 74,337 windows of the full archive, one Record each.
EXPECTED_DISTINCT_WINDOWS = 63366


@dataclass(frozen=True)
class TraceFileRef:
    """One extracted split file of the archive."""

    path: Path
    kind: TraceTaskKind
    split: TraceSplit
    expected: PinnedCounts | None = None
    """Pinned counts to check the file against; ``None`` skips the checks, as local fixtures do."""

    @property
    def archive_path(self) -> str:
        """Return the file's path inside the archive."""
        return self.kind.file_path(self.split)


@dataclass(frozen=True)
class TraceArchive:
    """The archive's eight split files, each once: the window files and the retrieval files."""

    files: tuple[TraceFileRef, ...]

    @classmethod
    def from_refs(cls, refs: Iterable[TraceFileRef]) -> "TraceArchive":
        """Check that the references are the archive's split files, each once.

        Returns:
            The archive.

        Raises:
            TimeFValidationError: If a split file is missing, repeated, or unknown.
        """
        files = tuple(refs)
        if sorted((file.kind, file.split) for file in files) != sorted(SOURCE_FILES):
            names = [kind.file_path(split) for kind, split in SOURCE_FILES]
            raise TimeFValidationError(f"TRACE requires each of the split files {names} once")
        return cls(files)

    @property
    def window_files(self) -> tuple[TraceFileRef, ...]:
        """Return the ``pretrain`` and ``forecasting`` files, which hold the windows."""
        return tuple(file for file in self.files if file.kind.holds_windows)

    @property
    def retrieval_files(self) -> tuple[TraceFileRef, ...]:
        """Return the ``retrieval`` files, which hold the descriptions and event reports."""
        return tuple(file for file in self.files if not file.kind.holds_windows)

    def check_distinct_measurements(self, found: int) -> None:
        """Compare the number of distinct measurements the windows hold with the pinned archive's.

        Fixtures carry no pinned counts, so only the pinned archive checks the count.

        Raises:
            TimeFValidationError: If the pinned archive's windows hold another number of distinct measurements.
        """
        pinned = all(file.expected is not None for file in self.files)
        if pinned and found != EXPECTED_DISTINCT_WINDOWS:
            raise TimeFValidationError(
                f"expected {EXPECTED_DISTINCT_WINDOWS} distinct measurement windows, found {found}"
            )


def check_pinned_counts(file: TraceFileRef, rows: int, tasks: int) -> None:
    """Compare one file's counts with its pinned counts, when the reference carries one.

    A reference without one, such as a test fixture, skips the check.

    Raises:
        TimeFValidationError: If the counts differ from the pinned archive's.
    """
    if file.expected is not None and (rows, tasks) != file.expected:
        raise TimeFValidationError(
            f"{file.path}: expected {file.expected.rows} rows and {file.expected.tasks} tasks or events, "
            f"found {rows} and {tasks}"
        )
