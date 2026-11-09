"""The HAR-CoT connector: accelerometer windows with an activity label and a GPT-4o rationale.

Each CSV row of the OpenTSLM release is one 2.56-second window of a triaxial accelerometer at
50 Hz, so each axis has 128 values. Each row becomes one Record and one ClassificationTask. The
target is the activity label. The rationale is the GPT-4o text that leads to the label.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Self

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask, Label, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive
from timenet_connectors.sources.readers import CsvDataReader, DataRow


HAR_COT_URL = "https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download"
HAR_COT_SHA256 = "0ec23e2ca7bff022148db8c0e669a6b973fc445e97c0178c1217ad635d03beae"
SPLIT_FILES: Mapping[Split, str] = {
    Split.TRAIN: "har_cot_train_cot.csv",
    Split.VALIDATION: "har_cot_val_cot.csv",
    Split.TEST: "har_cot_test_cot.csv",
}
COLUMNS = ("x_axis", "y_axis", "z_axis", "label", "prompt", "rationale")
AxisColumn = Literal["x_axis", "y_axis", "z_axis"]
SAMPLES_PER_WINDOW = 128
"""2.56 seconds at 50 Hz."""

# The label order is the order of the OpenTSLM training prompt.
ACTIVITY = LabelSet(
    name="activity",
    labels=(
        Label(name="biking"),
        Label(name="lying"),
        Label(name="running"),
        Label(name="sitting"),
        Label(name="standing"),
        Label(name="walking"),
        Label(name="walking_down", description="Walking down stairs."),
        Label(name="walking_up", description="Walking up stairs."),
    ),
)
# The release states no unit. A resting window has a magnitude of about 9.8, which is gravity in m/s².
ACCELERATION = TimeSeriesSpec(
    spec_type="accelerometer",
    name="Accelerometer",
    unit_value=ureg.meter / ureg.second**2,
    dtype="float64",
)
SAMPLED_AT_50_HZ = RegularAxis.from_rate_hz(50)


def _parse_axis(row: dict[str, str], column: AxisColumn) -> np.ndarray:
    """Parse one axis cell of a row into float64 values.

    The cells are Python list literals of floats. JSON reads them to the same values, and faster
    than ``ast.literal_eval``.

    Returns:
        The values of the axis.

    Raises:
        TimeNetConnectorError: If the cell does not hold 128 finite numbers.
    """
    message = f"{column} must hold {SAMPLES_PER_WINDOW} finite numbers"
    try:
        values = np.array(json.loads(row[column]), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimeNetConnectorError(message) from exc
    # NumPy turns a JSON null into NaN, so the finite check also rejects a missing value.
    if values.shape != (SAMPLES_PER_WINDOW,) or not np.isfinite(values).all():
        raise TimeNetConnectorError(message)
    return values


def _is_repeated_header(row: dict[str, str]) -> bool:
    """Return whether a row repeats the header line, with each cell equal to its column name."""
    return all(row[column] == column for column in COLUMNS)


class _SourceRow(DataRow):
    """A source CSV row before parsing its sensor axes."""

    x_axis: str
    y_axis: str
    z_axis: str
    label: str
    prompt: str
    rationale: str


@dataclass(frozen=True)
class HarCotRow:
    """One window of the release: three accelerometer axes, the label, and the GPT-4o rationale."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    label: str
    rationale: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Self:
        """Parse one data row of a split file.

        Returns:
            The parsed row.

        Raises:
            TimeNetConnectorError: If the label is not an activity of the label set.
        """
        if row["label"] not in ACTIVITY.names:
            raise TimeNetConnectorError(f"label {row['label']!r} is not one of {ACTIVITY.names}")
        return cls(
            x=_parse_axis(row, "x_axis"),
            y=_parse_axis(row, "y_axis"),
            z=_parse_axis(row, "z_axis"),
            label=row["label"],
            rationale=row["rationale"],
        )

    def create_record(self) -> Record:
        """Create the Record of the window, with one Signal for each axis.

        Returns:
            The Record, not yet added to a dataset.
        """
        signals = (
            Signal.from_values(self.x, spec=ACCELERATION, name="accelerometer_x", time_axis=SAMPLED_AT_50_HZ),
            Signal.from_values(self.y, spec=ACCELERATION, name="accelerometer_y", time_axis=SAMPLED_AT_50_HZ),
            Signal.from_values(self.z, spec=ACCELERATION, name="accelerometer_z", time_axis=SAMPLED_AT_50_HZ),
        )
        return Record(sources=(Source(name="Accelerometer", signals=signals),))

    def create_classification_task(self, record: Record, *, split: Split) -> ClassificationTask:
        """Create the task that asks for the activity of the window.

        Returns:
            The task, not yet added to a dataset.
        """
        return ClassificationTask(
            inputs=(record,),
            targets=(self.label,),
            label_set=ACTIVITY,
            rationale=self.rationale,
            split=split,
        )


@dataclass(frozen=True)
class HarCotSplitFile:
    """One CSV file of the release and the split that it holds."""

    split: Split
    path: Path

    def read_rows(self) -> Iterator[HarCotRow]:
        """Read the data rows of the file, in file order.

        The validation and test files repeat the header line before each data row after the first.
        This method skips each exact repeat of the header.

        Yields:
            Each parsed data row.

        Raises:
            TimeNetConnectorError: If the header is not the expected one, or a data row is malformed.
                The message names the file and the data row.
        """
        rows = (row.model_dump() for row in CsvDataReader(self.path, _SourceRow, columns=COLUMNS).iter_rows())
        data_rows = (row for row in rows if not _is_repeated_header(row))
        for number, row in enumerate(data_rows, start=1):
            try:
                parsed = HarCotRow.from_csv_row(row)
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{self.path.name} data row {number}: {exc}") from exc
            yield parsed


class HarCotConnector(BaseConnector[HarCotSplitFile]):
    """Connector for the HAR-CoT release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[HarCotSplitFile]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            The train, validation, and test files.
        """
        root = await ensure_archive(HAR_COT_URL, cache_dir, filename="har_cot.zip", sha256=HAR_COT_SHA256)
        return [HarCotSplitFile(split=split, path=root / name) for split, name in SPLIT_FILES.items()]

    def convert(self, raw_refs: list[HarCotSplitFile]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each window.

        Args:
            raw_refs: The split files from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for split_file in raw_refs:
            for row in split_file.read_rows():
                record = dataset.add_record(record=row.create_record())
                dataset.add_task(task=row.create_classification_task(record, split=split_file.split))
        return dataset


CONNECTOR = HarCotConnector
