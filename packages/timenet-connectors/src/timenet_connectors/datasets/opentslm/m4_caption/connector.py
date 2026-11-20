"""The M4-Caption connector: M4 competition series, each with a caption that GPT-4o wrote.

The OpenTSLM release holds the train part of all 100,000 M4 series, one CSV of series and one CSV of
captions for each of the six M4 frequencies. Each series becomes one Record and one AnswerTask. The
target is the caption. The connector leaves out the 706 series whose caption is an error message. The release archive stores no split. generate_splits.py made an 80/10/10 split,
stratified by frequency, and wrote it to splits.json.xz. The connector reads that file.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
from typing import Annotated

import numpy as np
from pydantic import AllowInfNan, Field, Json, StrictFloat

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.download import ensure_archive
from timenet_connectors.sources.readers import CsvDataReader, DataRow
from timenet_connectors.splitting import DatasetSplits, PrecomputedSplitter


M4_CAPTION_URL = "https://polybox.ethz.ch/index.php/s/MT3y9WdEebT8wfj/download/M4TimeSeriesCaptionDatasetV02.zip"
M4_CAPTION_SHA256 = "4acb748d94d7287fe58dbb5c3cf6314ea259010b7720f42506739269dec54edf"
RELEASE_FOLDER = "M4TimeSeriesCaptionDataset"
"""The folder in the archive that holds the CSV files."""
SPLITS_PATH = Path(__file__).with_name("splits.json.xz")
"""The stratified split of the release, as generate_splits.py wrote it."""

CAPTION_PROMPT = "Please generate a detailed caption for this time-series, describing it as accurately as possible."
"""The prompt of OpenTSLM's loader, the same for every series: the post-prompt of ``M4QADataset``."""

CAPTION_ERROR_PREFIX = "Error generating caption: "
"""The start of an error message of the caption API, which the release has instead of a caption for
706 series. The dataset leaves out these series."""

# The release states no unit, and the M4 series come from many domains.
M4_SERIES = TimeSeriesSpec(spec_type="m4_series", name="M4 series", unit_value=None, dtype="float32")

_SERIES_ID = re.compile(r"series-(?P<m4_id>(?P<letter>[A-Z])[1-9][0-9]*)")
"""A release id: the M4 id, such as M1, after a ``series-`` prefix that OpenTSLM added."""

_SeriesValue = Annotated[StrictFloat, AllowInfNan(False)]
"""A finite JSON number. Strict, so that a Boolean or a string in the list is an error, not a number."""


class M4Frequency(StrEnum):
    """An M4 frequency. The members are in the order that OpenTSLM's loader reads them."""

    DAILY = "Daily"
    HOURLY = "Hourly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    WEEKLY = "Weekly"
    YEARLY = "Yearly"

    @property
    def id_letter(self) -> str:
        """The letter that starts the M4 id of each series of this frequency, such as M in M1."""
        return _M4_ID_LETTERS[self]

    def parse_m4_id(self, series_id: str) -> str:
        """Get the M4 id from a release id, and check that it belongs to this frequency.

        Returns:
            The M4 id, such as ``M1``.

        Raises:
            TimeNetConnectorError: If the id is not an M4 series id of this frequency.
        """
        match = _SERIES_ID.fullmatch(series_id)
        if match is None or match["letter"] != self.id_letter:
            raise TimeNetConnectorError(f"{series_id!r} is not a {self} M4 series id")
        return match["m4_id"]


_M4_ID_LETTERS = {
    M4Frequency.DAILY: "D",
    M4Frequency.HOURLY: "H",
    M4Frequency.MONTHLY: "M",
    M4Frequency.QUARTERLY: "Q",
    M4Frequency.WEEKLY: "W",
    M4Frequency.YEARLY: "Y",
}


class _SeriesRow(DataRow):
    """A row of a series file: the release id and the values as a JSON list."""

    id: str
    series: Json[Annotated[list[_SeriesValue], Field(min_length=1)]]


class _CaptionRow(DataRow):
    """A row of a caption file: the release id and the caption."""

    id: str
    caption: str


@dataclass(frozen=True)
class M4CaptionSeries:
    """One M4 series of the release, with its frequency and caption."""

    m4_id: str
    frequency: M4Frequency
    values: np.ndarray
    caption: str

    def create_record_and_caption_task(self, *, split: Split) -> tuple[Record, AnswerTask]:
        """Create the Record of the series and the task that asks for its caption.

        The Record holds the values, the M4 id and the frequency. The task names the Record as its
        input. Add the Record to the dataset before the task.

        Returns:
            The Record and the task, not yet added to a dataset.
        """
        signal = Signal.from_values(self.values, spec=M4_SERIES, name="value", time_axis=OrdinalAxis())
        frequency = Annotation(key="frequency", value=self.frequency.value)
        record = Record(
            sources=(Source(name="M4", signals=(signal,)),),
            annotations=(frequency,),
            metadata={"m4_id": self.m4_id},
        )
        task = AnswerTask(inputs=(record,), prompt=CAPTION_PROMPT, targets=(self.caption,), split=split)
        return record, task


@dataclass(frozen=True)
class M4CaptionReader:
    """Read the series, the captions and the split of the extracted release."""

    directory: Path
    splits_path: Path

    def read_series(self) -> list[M4CaptionSeries]:
        """Read every series of the release with its caption, in the order of OpenTSLM's loader.

        The loader reads the files of each frequency, and skips ``m4_series.csv`` and ``m4_captions.csv``.
        The method leaves out each series whose caption is an error message of the caption API. A
        malformed file raises a :class:`~timenet.errors.TimeNetConnectorError` that names the file and
        its row.

        Returns:
            Each series with its caption.
        """
        series: list[M4CaptionSeries] = []
        for frequency in M4Frequency:
            series.extend(self._read_frequency(frequency))
        return series

    def split_series(self, series: Sequence[M4CaptionSeries]) -> DatasetSplits[M4CaptionSeries]:
        """Put each series in the split that the split file names for its M4 id.

        Returns:
            The series of each split, in the order of ``series``.
        """
        splitter = PrecomputedSplitter[M4CaptionSeries].from_file(
            self.splits_path, source_sha256=M4_CAPTION_SHA256, key=lambda one_series: one_series.m4_id
        )
        return splitter.split(series)

    def _read_frequency(self, frequency: M4Frequency) -> Iterator[M4CaptionSeries]:
        """Read the series of one frequency, paired with the caption in the same row of its caption file.

        Yields:
            Each series with its caption, except a series whose caption is an error message.

        Raises:
            TimeNetConnectorError: If the files do not pair one to one, or a series row is malformed.
        """
        series_path = self.directory / f"m4_series_{frequency}.csv"
        captions_path = self.directory / f"m4_captions_{frequency}.csv"
        series_rows = list(CsvDataReader(series_path, _SeriesRow, expected_columns=("id", "series")).iter_rows())
        caption_rows = list(CsvDataReader(captions_path, _CaptionRow, expected_columns=("id", "caption")).iter_rows())
        if len(series_rows) != len(caption_rows):
            raise TimeNetConnectorError(
                f"{series_path.name} has {len(series_rows)} series rows but {len(caption_rows)} caption rows"
            )
        for number, (series_row, caption_row) in enumerate(zip(series_rows, caption_rows, strict=True), start=1):
            try:
                m4_id = frequency.parse_m4_id(series_row.id)
                if series_row.id != caption_row.id:
                    raise TimeNetConnectorError(
                        f"series id {series_row.id!r} does not match caption id {caption_row.id!r}"
                    )
                if caption_row.caption.startswith(CAPTION_ERROR_PREFIX):
                    continue
                yield M4CaptionSeries(
                    m4_id=m4_id,
                    frequency=frequency,
                    values=np.array(series_row.series, dtype=np.float32),
                    caption=caption_row.caption,
                )
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{series_path.name} data row {number}: {exc}") from exc


class M4CaptionConnector(BaseConnector[M4CaptionReader]):
    """Connector for the M4-Caption release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[M4CaptionReader]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            A reader of the extracted release and of the split file of the connector.
        """
        root = await ensure_archive(M4_CAPTION_URL, cache_dir, filename="m4_caption.zip", sha256=M4_CAPTION_SHA256)
        return [M4CaptionReader(directory=root / RELEASE_FOLDER, splits_path=SPLITS_PATH)]

    def convert(self, raw_refs: list[M4CaptionReader]) -> TimeFDataset:
        """Create one Record and one AnswerTask for each series.

        Args:
            raw_refs: The reader from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (reader,) = raw_refs
        dataset = TimeFDataset(metadata=self.metadata())
        for split, series_in_split in reader.split_series(reader.read_series()).items():
            for series in series_in_split:
                record, task = series.create_record_and_caption_task(split=split)
                dataset.add_record(record=record)
                dataset.add_task(task=task)
        return dataset


CONNECTOR = M4CaptionConnector
