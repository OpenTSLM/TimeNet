"""The SleepEDF-CoT connector: EEG windows with a sleep stage and a GPT-4o rationale.

Each CSV row of the OpenTSLM release is one 30-second window of the Fpz-Cz EEG from a Sleep-EDF
recording, at 50 Hz, so each window has 1500 values. Each row becomes one Record and one
ClassificationTask. The target is the sleep stage. The rationale is the GPT-4o text that leads to
the stage.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Self

import numpy as np
from pydantic import NonNegativeInt

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files
from timenet_connectors.sources.readers import CsvDataReader, DataRow
from timenet_connectors.splitting import PrecomputedSplits


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


def _parse_window(row: dict[str, str]) -> np.ndarray:
    """Parse the time_series cell of a row, one list of numbers inside a list, into float64 values.

    The cells are Python list literals of floats. JSON reads them to the same values, and faster
    than ``ast.literal_eval``.

    Returns:
        The values of the window.

    Raises:
        TimeNetConnectorError: If the cell does not hold one list of 1500 finite numbers.
    """
    message = f"time_series must hold one list of {SAMPLES_PER_WINDOW} finite numbers"
    try:
        values = np.array(json.loads(row["time_series"]), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimeNetConnectorError(message) from exc
    # NumPy turns a JSON null into NaN, so the finite check also rejects a missing value.
    if values.shape != (1, SAMPLES_PER_WINDOW) or not np.isfinite(values).all():
        raise TimeNetConnectorError(message)
    return values[0]


def _parse_subject_id(row: dict[str, str]) -> str:
    """Get the subject ID from a row's Sleep-EDF recording name.

    Returns:
        The subject ID, such as ``sleep-cassette-10``.

    Raises:
        TimeNetConnectorError: If the ID cell is not a Sleep-EDF recording name.
    """
    match = _RECORDING_NAME.fullmatch(row["id"])
    if match is None:
        raise TimeNetConnectorError(f"id {row['id']!r} is not a Sleep-EDF recording name")
    return f"{_STUDIES[match['study']]}-{match['subject']}"


class _SourceRow(DataRow):
    """A source CSV row before parsing its signal."""

    id: str
    time_series: str
    label: str
    prompt: str
    rationale: str


@dataclass(frozen=True)
class SleepCotRow:
    """A recording, EEG values, sleep stage, and rationale for one window."""

    recording_id: str
    subject_id: str
    eeg: np.ndarray
    label: str
    rationale: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Self:
        """Parse one data row of the release.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the label is not a stage of the label set.
        """
        if row["label"] not in SLEEP_STAGE.names:
            raise TimeNetConnectorError(f"label {row['label']!r} is not one of {SLEEP_STAGE.names}")
        return cls(
            recording_id=row["id"],
            subject_id=_parse_subject_id(row),
            eeg=_parse_window(row),
            label=row["label"],
            rationale=row["rationale"],
        )

    def create_record(self) -> Record:
        """Create a record for the window with its recording name and subject.

        Returns:
            The new record.
        """
        signal = Signal.from_values(self.eeg, spec=EEG, name="EEG Fpz-Cz", time_axis=SAMPLED_AT_50_HZ)
        source = Source(name="EEG", signals=(signal,))
        subject = Annotation(key="subject_ids", value=[self.subject_id])
        return Record(sources=(source,), annotations=(subject,), metadata={"recording_id": self.recording_id})

    def create_classification_task(self, record: Record, *, split: Split) -> ClassificationTask:
        """Create a sleep stage classification task for the window.

        Returns:
            The new task.
        """
        return ClassificationTask(
            inputs=(record,),
            targets=(self.label,),
            label_set=SLEEP_STAGE,
            rationale=self.rationale,
            split=split,
        )


@dataclass(frozen=True)
class SleepCotReader:
    """Reader for the release CSV and the precomputed split of its rows."""

    csv_path: Path
    splits_path: Path

    def read_rows(self) -> list[SleepCotRow]:
        """Read every data row of the release, in file order.

        Returns:
            The parsed rows.

        Raises:
            TimeNetConnectorError: If the header is not the expected one, or a data row is malformed.
                The message names the file and the data row.
        """
        rows = []
        for number, source in enumerate(
            CsvDataReader(self.csv_path, _SourceRow, expected_columns=COLUMNS).iter_rows(), start=1
        ):
            try:
                rows.append(SleepCotRow.from_csv_row(source.model_dump()))
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{self.csv_path.name} data row {number}: {exc}") from exc
        return rows

    def read_splits(self, n_rows: int) -> list[Split]:
        """Read the split of each row from the splits file.

        Returns:
            The split for each of the n_rows rows, in file order.
        """
        splits = PrecomputedSplits[NonNegativeInt].read_from_file(self.splits_path)
        return splits.assign_splits(range(n_rows), source_sha256=SLEEP_COT_SHA256)


class SleepCotConnector(BaseConnector[SleepCotReader]):
    """Connector for the SleepEDF-CoT release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[SleepCotReader]:  # ruff: ignore[no-self-use]
        """Download the release CSV.

        Args:
            cache_dir: Directory for the downloaded file.

        Returns:
            A reader for the release CSV and the precomputed split of the connector.
        """
        artifact = Artifact(SLEEP_COT_URL, cache_dir / "sleep_cot.csv", sha256=SLEEP_COT_SHA256)
        (csv_path,) = await download_files([artifact])
        return [SleepCotReader(csv_path=csv_path, splits_path=SPLITS_PATH)]

    def convert(self, raw_refs: list[SleepCotReader]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each window.

        Args:
            raw_refs: The reader from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (reader,) = raw_refs
        rows = reader.read_rows()
        splits = reader.read_splits(len(rows))
        dataset = TimeFDataset(metadata=self.metadata())
        for row, split in zip(rows, splits, strict=True):
            record = dataset.add_record(record=row.create_record())
            dataset.add_task(task=row.create_classification_task(record, split=split))
        return dataset


CONNECTOR = SleepCotConnector
