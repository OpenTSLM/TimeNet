"""The TSAQA release as published: its Hub pin, its three split files, and their pinned inventory."""

from collections import Counter
from collections.abc import Mapping
from enum import StrEnum, unique
from pathlib import Path

from timenet.errors import TimeFValidationError
from timenet.types import Split
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import QuestionFormat, QuestionType, SourceTask


HF_REPO = "TSAQA/TSAQA-Benchmark"
SOURCE_REVISION = "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd"


@unique
class SourceSplit(StrEnum):
    """One of the release's split files, named as the authors name it, with the TimeNet split it maps to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def source_file(self) -> str:
        """Return the file's name in the Hub repository, such as ``val.parquet``."""
        return f"{self.value}.parquet"

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the file's rows."""
        return _TIMENET_SPLITS[self]


_TIMENET_SPLITS: dict[SourceSplit, Split] = {
    SourceSplit.TRAIN: Split.TRAIN,
    SourceSplit.VAL: Split.VALIDATION,
    SourceSplit.TEST: Split.TEST,
}


InventoryKey = tuple[SourceTask, QuestionType, QuestionFormat]
"""A row's published ``task`` and ``question_type``, and the format its question has."""


def check_inventory(path: Path, found: Counter[InventoryKey], pinned: Mapping[InventoryKey, int] | None) -> None:
    """Compare one file's rows, counted by task, published type and format, with its pinned inventory.

    Args:
        path: The file, for the error message.
        found: The file's rows, counted.
        pinned: The pinned counts, or ``None`` to skip the check, as test fixtures do.

    Raises:
        TimeFValidationError: If the counts differ from the pinned release's.
    """
    if pinned is not None and dict(found) != dict(pinned):
        differs = sorted(
            f"{'/'.join(key)}: expected {pinned.get(key, 0)}, found {found.get(key, 0)}"
            for key in set(found) | set(pinned)
            if found.get(key, 0) != pinned.get(key, 0)
        )
        raise TimeFValidationError(f"{path}: the rows differ from the pinned release: {differs}")


_T, _QT, _QF = SourceTask, QuestionType, QuestionFormat

# The rows at SOURCE_REVISION by task, published question type and format, then train, val and test
# counts. The totals are the paper's 70/10/20 split of 210,000 rows, and the published question types
# sum to its Section 3.4 counts: 84,632 TF, 85,479 MC and 39,889 PZ. The "ordering" rows include 5,049
# true-or-false and 4,951 multiple-choice questions.
_PINNED_ROWS: tuple[tuple[SourceTask, QuestionType, QuestionFormat, int, int, int], ...] = (
    (_T.ANOMALY_DETECTION, _QT.TRUE_OR_FALSE, _QF.TRUE_OR_FALSE, 21000, 3000, 6000),
    (_T.CLASSIFICATION, _QT.MULTIPLE_CHOICES, _QF.MULTIPLE_CHOICE, 21000, 3000, 6000),
    (_T.CHARACTERIZATION, _QT.TRUE_OR_FALSE, _QF.TRUE_OR_FALSE, 10424, 1503, 3003),
    (_T.CHARACTERIZATION, _QT.MULTIPLE_CHOICES, _QF.MULTIPLE_CHOICE, 10576, 1497, 2997),
    (_T.COMPARISON, _QT.TRUE_OR_FALSE, _QF.TRUE_OR_FALSE, 10375, 1462, 2945),
    (_T.COMPARISON, _QT.MULTIPLE_CHOICES, _QF.MULTIPLE_CHOICE, 10625, 1538, 3055),
    (_T.DATA_TRANSFORMATION, _QT.TRUE_OR_FALSE, _QF.TRUE_OR_FALSE, 10518, 1466, 3002),
    (_T.DATA_TRANSFORMATION, _QT.MULTIPLE_CHOICES, _QF.MULTIPLE_CHOICE, 10482, 1534, 2998),
    (_T.TEMPORAL_RELATIONSHIP, _QT.TRUE_OR_FALSE, _QF.TRUE_OR_FALSE, 6915, 963, 2056),
    (_T.TEMPORAL_RELATIONSHIP, _QT.MULTIPLE_CHOICES, _QF.MULTIPLE_CHOICE, 7153, 1058, 1966),
    (_T.TEMPORAL_RELATIONSHIP, _QT.ORDERING, _QF.TRUE_OR_FALSE, 3525, 537, 987),
    (_T.TEMPORAL_RELATIONSHIP, _QT.ORDERING, _QF.MULTIPLE_CHOICE, 3475, 463, 1013),
    (_T.TEMPORAL_RELATIONSHIP, _QT.ORDERING, _QF.PUZZLING, 20932, 2979, 5978),
)
PINNED_INVENTORY: Mapping[SourceSplit, Mapping[InventoryKey, int]] = {
    split: {(task, question_type, fmt): counts[column] for task, question_type, fmt, *counts in _PINNED_ROWS}
    for column, split in enumerate(SourceSplit)
}
"""Each file's rows at :data:`SOURCE_REVISION`, by task, published question type and format."""
