"""The EngineMT-QA release as published: its Hub pin, its three files, their shapes, and their pinned inventory."""

from collections.abc import Mapping
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import Split


HF_REPO = "pandalin98/EngineMT-QA"
SOURCE_REVISION = "ea2051a15512212976c22c1a075ec549c7d48eef"
"""The revision the authors' audit (``docs/ISSUE_23_CHANNEL_AUDIT.md`` at ITFormer-ICML25 ``b80142f``) checked."""


@unique
class Part(StrEnum):
    """The release's three files: the signals, and the train and test QA."""

    SIGNALS = "signals"
    TRAIN = "train"
    TEST = "test"

    @property
    def source_file(self) -> str:
        """Return the file's name in the Hub repository, such as ``train_qa.jsonl``."""
        return "time_series_data.h5" if self is Part.SIGNALS else f"{self.value}_qa.jsonl"


QA_PARTS = (Part.TRAIN, Part.TEST)
"""The QA files, in the order the loader reads them."""


def split_of(part: Part) -> Split:
    """Return the TimeNet split of a QA file: the authors train on ``train_qa`` and test on ``test_qa``.

    Raises:
        TimeFValidationError: If the part is not a QA file.
    """
    if part not in QA_PARTS:
        raise TimeFValidationError(f"{part} is not a QA file")
    return Split(part.value)


SEQ_DATA = "seq_data"
"""The HDF5 dataset of every cycle's values, shaped ``(rows, steps, channels)``."""
DATA_ID = "data_ID"
"""The HDF5 dataset of each row's id, which a QA's ``id`` names. It is ``1, 2, ...``: row ``i`` has id ``i + 1``."""

N_ROWS = 118_921
N_STEPS = 600
N_CHANNELS = 33
"""The released channel count. The paper says 32; the authors' audit confirms 33 and names none of them."""

SAMPLE_RATE_HZ = 1
"""The paper's "10-minute (600 points) time-series segments" (Section 5.2). The HDF5 states no rate."""

MULTI_CYCLE_ROWS = 10
"""How many rows a multi-cycle QA names."""


class PinnedInventory(NamedTuple):
    """One QA file's pinned inventory at :data:`SOURCE_REVISION`."""

    lines: int
    qa_per_stage: Mapping[int, int]
    """The published QA pairs per stage, empty answers included."""
    empty_answers: Mapping[str, int]
    """The closed QA pairs with an empty answer, by closed question, which the connector skips."""

    def check(self, path: Path, found: "PinnedInventory") -> None:
        """Compare a file's counted inventory with this one.

        Raises:
            TimeFValidationError: If the file isn't the pinned revision's.
        """
        if (found.lines, dict(found.qa_per_stage), dict(found.empty_answers)) != (
            self.lines,
            dict(self.qa_per_stage),
            dict(self.empty_answers),
        ):
            raise TimeFValidationError(f"{path}: EngineMT-QA holds {found}, the pinned file holds {self}")


class PinnedRelease(NamedTuple):
    """What the pinned release holds, which every build checks."""

    rows: int
    """The HDF5 rows."""
    referenced_rows: int
    """The distinct rows the QA files name, which the default build holds."""
    qa: Mapping[Part, PinnedInventory]
    """Each QA file's inventory."""

    def check_referenced_rows(self, found: int) -> None:
        """Compare the distinct rows the QA files name with the pinned count.

        Raises:
            TimeFValidationError: If the QA files aren't the pinned revision's.
        """
        if found != self.referenced_rows:
            raise TimeFValidationError(f"the QA files name {found} rows, the pinned files {self.referenced_rows}")


# The QA counts are the authors' README's, for test, and their audit's, for train. The empty answers are
# missing labels: an empty fault cause doesn't follow from the five components' health.
PINNED_RELEASE = PinnedRelease(
    rows=N_ROWS,
    referenced_rows=7_755,
    qa={
        Part.TRAIN: PinnedInventory(
            lines=13_291,
            qa_per_stage={1: 41_128, 2: 35_987, 3: 16_300, 4: 8_150},
            empty_answers={"fault_cause": 1_054, "health_state": 215},
        ),
        Part.TEST: PinnedInventory(
            lines=5_539,
            qa_per_stage={1: 17_240, 2: 15_085, 3: 6_768, 4: 3_384},
            empty_answers={"fault_cause": 422, "health_state": 78},
        ),
    },
)
