"""The SleepEDF-CoT connector: EEG windows with a sleep stage and a GPT-4o rationale.

Each CSV row of the OpenTSLM release is one 30-second window of the Fpz-Cz EEG from a Sleep-EDF
recording, at 50 Hz, so each window has 1500 values. Each row becomes one Record and one
ClassificationTask. The target is the sleep stage. The rationale is the GPT-4o text that leads to
the stage.

The release has no split column. OpenTSLM splits the rows when it loads them, and this connector
gives each row the same split.

The task also keeps the GPT-4o prompt that produced the rationale, as a task annotation. That
prompt names the correct stage, so it is not an input annotation and a model does not see it.
"""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Self

import numpy as np
from sklearn.model_selection import train_test_split

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

# The settings of OpenTSLM's load_sleepedf_cot_splits.
_TEST_FRACTION = 0.1
_VALIDATION_FRACTION = 0.1
_TEST_SEED = 42
_VALIDATION_SEED = 43

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
    """Give the subject id that the recording name of a row states, such as ``sleep-cassette-10``.

    Returns:
        The subject id.

    Raises:
        TimeNetConnectorError: If the id cell is not a Sleep-EDF recording name.
    """
    match = _RECORDING_NAME.fullmatch(row["id"])
    if match is None:
        raise TimeNetConnectorError(f"{location}: id {row['id']!r} is not a Sleep-EDF recording name")
    return f"{_STUDIES[match['study']]}-{match['subject']}"


@dataclass(frozen=True)
class SleepCotRow:
    """One window of the release: the recording, the EEG values, the stage, and the two GPT-4o texts."""

    recording_id: str
    subject_id: str
    eeg: np.ndarray
    label: str
    generation_prompt: str
    """The prompt that GPT-4o got to write the rationale. It names the correct stage."""
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
        """Create the Record of the window, with its recording name and its subject.

        Returns:
            The Record, not yet added to a dataset.
        """
        signal = Signal.from_values(self.eeg, spec=EEG, name="EEG Fpz-Cz", time_axis=SAMPLED_AT_50_HZ)
        return Record(
            sources=(Source(name="EEG", signals=(signal,)),),
            annotations=(Annotation(key="subject_ids", value=[self.subject_id]),),
            metadata={"recording_id": self.recording_id},
        )

    def create_classification_task(self, record: Record, *, split: Split) -> ClassificationTask:
        """Create the task that asks for the sleep stage of the window.

        Returns:
            The task, not yet added to a dataset.
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


def reproduce_opentslm_splits(labels: Sequence[str]) -> list[Split]:
    """Give each row the split that OpenTSLM's ``load_sleepedf_cot_splits`` gives it.

    OpenTSLM splits the rows with two stratified ``train_test_split`` calls: 10 % of all rows for
    test, then 10 % of all rows for validation. Splitting the row positions with the same
    settings selects the same rows.

    Args:
        labels: The label of each row, in file order.

    Returns:
        The split of each row, in file order.
    """
    stratify = np.asarray(labels)
    positions = np.arange(len(stratify))
    train_and_validation, test = train_test_split(
        positions, test_size=_TEST_FRACTION, random_state=_TEST_SEED, stratify=stratify
    )
    _, validation = train_test_split(
        train_and_validation,
        test_size=_VALIDATION_FRACTION / (1.0 - _TEST_FRACTION),
        random_state=_VALIDATION_SEED,
        stratify=stratify[train_and_validation],
    )
    splits = [Split.TRAIN] * len(stratify)
    for position in validation:
        splits[position] = Split.VALIDATION
    for position in test:
        splits[position] = Split.TEST
    return splits


class SleepCotConnector(BaseConnector[Path]):
    """Connector for the SleepEDF-CoT release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use]
        """Download the release CSV once.

        Args:
            cache_dir: The directory that holds the downloaded file.

        Returns:
            The path of the CSV file.
        """
        return await download_files([Artifact(SLEEP_COT_URL, cache_dir / "sleep_cot.csv", sha256=SLEEP_COT_SHA256)])

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each window.

        Args:
            raw_refs: The CSV file from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (csv_path,) = raw_refs
        rows = read_rows(csv_path)
        splits = reproduce_opentslm_splits([row.label for row in rows])
        dataset = TimeFDataset(metadata=self.metadata())
        for row, split in zip(rows, splits, strict=True):
            record = dataset.add_record(record=row.create_record())
            dataset.add_task(task=row.create_classification_task(record, split=split))
        return dataset


CONNECTOR = SleepCotConnector
