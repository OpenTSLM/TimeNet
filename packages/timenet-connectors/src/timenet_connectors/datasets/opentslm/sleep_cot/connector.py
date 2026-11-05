"""The SleepEDF-CoT connector: EEG windows with a sleep stage and a GPT-4o rationale.

Each CSV row of the OpenTSLM release is one 30-second window of the Fpz-Cz EEG from a Sleep-EDF
recording, at 50 Hz, so each window has 1500 values. Each row becomes one Record and one
ClassificationTask. The target is the sleep stage. The rationale is the GPT-4o text that leads to
the stage.

GPT-4o prompts used to generate the rationales are attached to the tasks as annotations.
"""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeInt, ValidationError

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


SLEEP_COT_URL = "https://polybox.ethz.ch/index.php/s/ZryWSdCFJZ9JR3R/download"
SLEEP_COT_SHA256 = "8c3ecd9b42179ab50d14e30edae87f9c5a0b7acfd4d15598198dfeaaedae01e3"
COLUMNS = ("id", "time_series", "label", "prompt", "rationale")
SAMPLES_PER_WINDOW = 1500
"""30 seconds at 50 Hz."""
SPLITS_PATH = Path(__file__).with_name("splits.json")
"""The split of OpenTSLM's loader for each row of the release, as generate_splits.py wrote it."""

# The release writes the stage codes of the Rechtschaffen and Kales manual. The prompts and the
# rationales write the names in the descriptions.
SLEEP_STAGE = LabelSet(
    name="sleep_stage",
    labels=(
        Label(name="W", description="Wake"),
        Label(name="N1", description="Non-REM stage 1"),
        Label(name="N2", description="Non-REM stage 2"),
        Label(name="N3", description="Non-REM stage 3"),
        Label(name="N4", description="Non-REM stage 4"),
        Label(name="REM", description="REM sleep"),
    ),
)
# OpenTSLM converted the microvolts of Sleep-EDF to volts and kept every second sample of 100 Hz.
EEG = TimeSeriesSpec(spec_type="eeg", name="EEG", unit_value=ureg.volt, dtype="float64")
SAMPLED_AT_50_HZ = RegularAxis.from_rate_hz(50)

# A Sleep-EDF recording name, in lower case: the study, two subject digits, the night, a letter,
# and a zero. The subject id has the form of the physionet/sleep-edfx dataset.
_RECORDING_NAME = re.compile(r"(?P<study>sc4|st7)(?P<subject>[0-9]{2})[0-9][a-z]0")
_STUDIES = {"sc4": "sleep-cassette", "st7": "sleep-telemetry"}


def _parse_window(row: Mapping[str, str], *, location: str) -> np.ndarray:
    """Parse the time_series cell of a row, one list of numbers inside a list, into float64 values.

    The cells are Python list literals of floats. JSON reads them to the same values, and faster
    than ``ast.literal_eval``.

    Returns:
        The values of the window.

    Raises:
        TimeNetConnectorError: If the cell does not hold one list of 1500 finite numbers.
    """
    message = f"{location}: time_series must hold one list of {SAMPLES_PER_WINDOW} finite numbers"
    try:
        values = np.array(json.loads(row["time_series"]), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimeNetConnectorError(message) from exc
    # NumPy turns a JSON null into NaN, so the finite check also rejects a missing value.
    if values.shape != (1, SAMPLES_PER_WINDOW) or not np.isfinite(values).all():
        raise TimeNetConnectorError(message)
    return values[0]


def _parse_subject_id(row: Mapping[str, str], *, location: str) -> str:
    """Get the subject ID from a row's Sleep-EDF recording name.

    Returns:
        The subject ID, such as ``sleep-cassette-10``.

    Raises:
        TimeNetConnectorError: If the ID cell is not a Sleep-EDF recording name.
    """
    match = _RECORDING_NAME.fullmatch(row["id"])
    if match is None:
        raise TimeNetConnectorError(f"{location}: id {row['id']!r} is not a Sleep-EDF recording name")
    return f"{_STUDIES[match['study']]}-{match['subject']}"


@dataclass(frozen=True)
class SleepCotRow:
    """A recording, EEG values, sleep stage, prompt, and rationale for one window."""

    recording_id: str
    subject_id: str
    eeg: np.ndarray
    label: str
    generation_prompt: str
    """The prompt used to generate the rationale, including the correct sleep stage."""
    rationale: str

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str], *, location: str) -> Self:
        """Parse one data row of the release.

        Returns:
            The parsed row.
        """
        return cls(
            recording_id=row["id"],
            subject_id=_parse_subject_id(row, location=location),
            eeg=_parse_window(row, location=location),
            label=row["label"],
            generation_prompt=row["prompt"],
            rationale=row["rationale"],
        )

    def create_record(self) -> Record:
        """Create a record for the window with its recording name and subject.

        Returns:
            The new record.
        """
        signal = Signal.from_values(self.eeg, spec=EEG, name="EEG Fpz-Cz", time_axis=SAMPLED_AT_50_HZ)
        return Record(
            sources=(Source(name="EEG", signals=(signal,)),),
            annotations=(Annotation(key="subject_ids", value=[self.subject_id]),),
            metadata={"recording_id": self.recording_id},
        )

    def create_classification_task(self, record: Record, *, split: Split) -> ClassificationTask:
        """Create a sleep stage classification task for the window.

        Returns:
            The new task.
        """
        generation_prompt = Annotation(
            key="rationale_generation_prompt",
            value=self.generation_prompt,
            description="The GPT-4o prompt that produced the rationale. It names the correct stage.",
        )
        return ClassificationTask(
            inputs=(record,),
            targets=(self.label,),
            label_set=SLEEP_STAGE,
            rationale=self.rationale,
            split=split,
            annotations=(generation_prompt,),
        )


def read_rows(path: Path) -> list[SleepCotRow]:
    """Read every data row of the release, in file order.

    Returns:
        The parsed rows.

    Raises:
        TimeNetConnectorError: If the header is not the expected one.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise TimeNetConnectorError(f"{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}")
        return [
            SleepCotRow.from_csv_row(row, location=f"{path.name} data row {number}")
            for number, row in enumerate(reader, start=1)
        ]


class FrozenSplits(BaseModel):
    """Validation and test row positions, starting at zero in file order.

    The connector reads the split from splits.json.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    """SHA-256 digest of the release CSV used to create the split."""
    validation: tuple[NonNegativeInt, ...]
    test: tuple[NonNegativeInt, ...]

    @classmethod
    def read_from_file(cls, path: Path) -> Self:
        """Read the split from a splits.json file.

        Returns:
            The split.

        Raises:
            TimeNetConnectorError: If the file does not hold a valid split.
        """
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise TimeNetConnectorError(f"{path.name} does not hold a valid split: {exc}") from exc

    def assign_split_to_each_row(self, n_rows: int) -> list[Split]:
        """Assign a split to each row of the pinned release, in file order.

        Returns:
            The split for each row; rows absent from the file belong to the train split.

        Raises:
            TimeNetConnectorError: If the CSV digest differs or a row position is invalid or repeated.
        """
        if self.source_sha256 != SLEEP_COT_SHA256:
            raise TimeNetConnectorError(
                f"the split was made from the CSV {self.source_sha256}, not from the pinned {SLEEP_COT_SHA256}"
            )
        splits = [Split.TRAIN] * n_rows
        for split, positions in ((Split.VALIDATION, self.validation), (Split.TEST, self.test)):
            for position in positions:
                if position >= n_rows or splits[position] is not Split.TRAIN:
                    raise TimeNetConnectorError(f"the split names row {position} twice or outside the {n_rows} rows")
                splits[position] = split
        return splits


@dataclass(frozen=True)
class SleepCotRelease:
    """Paths to the release CSV and its saved row split."""

    csv_path: Path
    splits_path: Path


class SleepCotConnector(BaseConnector[SleepCotRelease]):
    """Connector for the SleepEDF-CoT release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[SleepCotRelease]:  # ruff: ignore[no-self-use]
        """Download the release CSV.

        Args:
            cache_dir: Directory for the downloaded file.

        Returns:
            The release CSV and saved split paths.
        """
        artifact = Artifact(SLEEP_COT_URL, cache_dir / "sleep_cot.csv", sha256=SLEEP_COT_SHA256)
        (csv_path,) = await download_files([artifact])
        return [SleepCotRelease(csv_path=csv_path, splits_path=SPLITS_PATH)]

    def convert(self, raw_refs: list[SleepCotRelease]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each window.

        Args:
            raw_refs: The release files from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (release,) = raw_refs
        rows = read_rows(release.csv_path)
        splits = FrozenSplits.read_from_file(release.splits_path).assign_split_to_each_row(len(rows))
        dataset = TimeFDataset(metadata=self.metadata())
        for row, split in zip(rows, splits, strict=True):
            record = dataset.add_record(record=row.create_record())
            dataset.add_task(task=row.create_classification_task(record, split=split))
        return dataset


CONNECTOR = SleepCotConnector
