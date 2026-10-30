"""The VeriTime release as published: its Hub pin, its four files, their tasks, and their pinned inventory."""

from collections.abc import Mapping
from enum import StrEnum, unique
from pathlib import Path
from typing import NamedTuple

from timenet.errors import TimeFValidationError
from timenet.types import Split


HF_REPO = "HayleyZhou1113/VeriTime"
SOURCE_REVISION = "fc88cb1305a96a208093e13b69d766b19c84c99b"


@unique
class Family(StrEnum):
    """The paper's two task categories, one folder of the release each."""

    SCENARIO = "scenario"
    """Questions the authors selected from the ChatTS SFT data, on its synthetic series."""
    KNOWLEDGE = "knowledge"
    """Classification questions on four real-world datasets, with their original splits."""


@unique
class Part(StrEnum):
    """One file of the release: a family's train or test rows."""

    SCENARIO_TRAIN = "scenario_train"
    SCENARIO_TEST = "scenario_test"
    KNOWLEDGE_TRAIN = "knowledge_train"
    KNOWLEDGE_TEST = "knowledge_test"

    @property
    def family(self) -> Family:
        """Return the family whose rows the file holds."""
        return Family(self.value.split("_")[0])

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the file's rows: the authors' split."""
        return Split(self.value.split("_")[1])

    @property
    def source_file(self) -> str:
        """Return the file's path in the Hub repository, with the release's spelling."""
        return _SOURCE_FILES[self]


_SOURCE_FILES = {
    Part.SCENARIO_TRAIN: "Dataset/Scenario/scenario_shuffled_train.jsonl",
    Part.SCENARIO_TEST: "Dataset/Scenario/scenario_shuffled_test.jsonl",
    # "knowldege" is the release's own spelling.
    Part.KNOWLEDGE_TRAIN: "Dataset/Knowledge/knowldege_shuffled_train.jsonl",
    Part.KNOWLEDGE_TEST: "Dataset/Knowledge/knowledge_test.jsonl",
}


@unique
class SourceTask(StrEnum):
    """A row's ``task``, as the release spells it: the paper's seven tasks."""

    ANOMALY_DETECTION = "Anomaly detection"
    SCENARIO_ATTRIBUTION = "Scenario attribution"
    INFERENTIAL_CALCULATION = "Inferential calculation"
    CTU = "CTU"
    """Computer type usage: desktop or laptop, from a household's 24-hour electricity consumption."""
    ECG = "ECG"
    """Heart rhythm from a single-lead AliveCor ECG."""
    EMG = "EMG"
    """Neuromuscular condition from a needle EMG of the tibialis anterior."""
    RCW = "RCW"
    """Right whale call detection in a two-second hydrophone recording."""

    @property
    def family(self) -> Family:
        """Return the family the task belongs to."""
        return Family.SCENARIO if self in _SCENARIO_TASKS else Family.KNOWLEDGE

    @property
    def slug(self) -> str:
        """Return the task's name in annotation keys, such as ``anomaly_detection`` or ``ecg``."""
        return self.value.lower().replace(" ", "_")


_SCENARIO_TASKS = frozenset(
    {
        SourceTask.ANOMALY_DETECTION,
        SourceTask.SCENARIO_ATTRIBUTION,
        SourceTask.INFERENTIAL_CALCULATION,
    }
)


class PinnedInventory(NamedTuple):
    """The release's pinned inventory at :data:`SOURCE_REVISION`."""

    rows: Mapping[Part, Mapping[SourceTask, int]]
    """Each file's rows per task."""
    series: Mapping[Family, int]
    """Each family's distinct series over both its files. Scenario questions share series."""

    def check_rows(self, path: Path, part: Part, rows: Mapping[SourceTask, int]) -> None:
        """Compare a file's counted rows per task with this inventory.

        Raises:
            TimeFValidationError: If the file isn't the pinned revision's.
        """
        if dict(rows) != dict(self.rows[part]):
            raise TimeFValidationError(
                f"{path}: rows per task {dict(rows)} differ from the pinned {dict(self.rows[part])}"
            )

    def check_series(self, family: Family, series: int) -> None:
        """Compare a family's counted distinct series with this inventory.

        Raises:
            TimeFValidationError: If the files aren't the pinned revision's.
        """
        if series != self.series[family]:
            raise TimeFValidationError(
                f"{family}: {series} distinct series differ from the pinned {self.series[family]}"
            )


PINNED_INVENTORY = PinnedInventory(
    rows={
        Part.SCENARIO_TRAIN: {
            SourceTask.ANOMALY_DETECTION: 1016,
            SourceTask.SCENARIO_ATTRIBUTION: 855,
            SourceTask.INFERENTIAL_CALCULATION: 188,
        },
        Part.SCENARIO_TEST: {
            SourceTask.ANOMALY_DETECTION: 180,
            SourceTask.SCENARIO_ATTRIBUTION: 176,
            SourceTask.INFERENTIAL_CALCULATION: 105,
        },
        Part.KNOWLEDGE_TRAIN: {SourceTask.CTU: 131, SourceTask.ECG: 584, SourceTask.EMG: 300, SourceTask.RCW: 127},
        Part.KNOWLEDGE_TEST: {SourceTask.CTU: 144, SourceTask.ECG: 200, SourceTask.EMG: 150, SourceTask.RCW: 200},
    },
    # 2,520 scenario rows ask about 1,259 series; no two knowledge rows share one.
    series={Family.SCENARIO: 1259, Family.KNOWLEDGE: 1836},
)
