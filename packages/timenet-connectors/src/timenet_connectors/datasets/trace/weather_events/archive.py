"""Where TRACE's files come from, and the eight split files the archive holds.

The authors publish three folders. ``pretrain`` and ``forecasting`` hold JSON window files, and
``retrieval`` holds parquet files of text about windows. Each folder is split as the authors split it.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from timenet.types import Split


# The README's Google Drive link, as a direct download that skips the HTML confirmation page. The
# archive is 510,759,681 bytes (487 MiB) and unpacks to 1.75 GB. Drive has no immutable revision, so
# the archive's bytes are pinned by sha256 instead.
SOURCE_URL = (
    "https://drive.usercontent.google.com/download?id=1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY&export=download&confirm=t"
)
SOURCE_SHA256 = "a5d24d75a0b9f0f846a185caa46c4f77e0ef98272c12808df76fac25a181be58"


@unique
class Kind(StrEnum):
    """The authors' three folders: what their files hold.

    ``pretrain`` and ``forecasting`` files hold windows, and ``pretrain`` windows add an event label.
    ``retrieval`` files hold no windows: each row gives one window's description and event report.
    """

    PRETRAIN = "pretrain"
    FORECASTING = "forecasting"
    RETRIEVAL = "retrieval"

    @property
    def holds_windows(self) -> bool:
        """Whether the folder's files hold windows, not text about windows."""
        return self is not Kind.RETRIEVAL

    @property
    def holds_forecasts(self) -> bool:
        """Whether the folder's windows are the authors' forecasting windows."""
        return self is Kind.FORECASTING


@unique
class SplitName(StrEnum):
    """The authors' split name, as a split file is named."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def timef_split(self) -> Split:
        """The TimeF split that the authors' split becomes."""
        return _SPLITS[self]


_SPLITS = {SplitName.TRAIN: Split.TRAIN, SplitName.VAL: Split.VALIDATION, SplitName.TEST: Split.TEST}

FILES = (
    (Kind.PRETRAIN, SplitName.TRAIN),
    (Kind.PRETRAIN, SplitName.VAL),
    (Kind.PRETRAIN, SplitName.TEST),
    (Kind.FORECASTING, SplitName.TRAIN),
    (Kind.FORECASTING, SplitName.VAL),
    (Kind.FORECASTING, SplitName.TEST),
    (Kind.RETRIEVAL, SplitName.TRAIN),
    (Kind.RETRIEVAL, SplitName.TEST),
)
"""The archive's split files, in the order the connector reads them."""


def archive_path(kind: Kind, split: SplitName) -> str:
    """Return a split file's path below the archive root: JSON windows, or parquet retrieval rows.

    Returns:
        The path, such as ``dataset/pretrain/train.json``.
    """
    return f"dataset/{kind}/{split}.{'json' if kind.holds_windows else 'parquet'}"


@dataclass(frozen=True)
class TraceFile:
    """One extracted split file of the archive."""

    path: Path
    kind: Kind
    split: SplitName

    @property
    def archive_path(self) -> str:
        """The file's path inside the archive."""
        return archive_path(self.kind, self.split)
