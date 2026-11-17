"""The M4-Caption connector: M4 competition series, each with a caption that GPT-4o wrote.

The OpenTSLM release holds the train part of all 100,000 M4 series, one CSV of series and one CSV of
captions for each of the six M4 frequencies. Each series becomes one Record and one AnswerTask. The
target is the caption. The release has no split, so the connector reads the split of OpenTSLM's
loader from splits.json.xz, which generate_splits.py wrote.
"""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
import lzma
from pathlib import Path
import re
from typing import Annotated, Self

import numpy as np
from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, Json, StrictFloat, ValidationError, model_validator

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.download import ensure_archive
from timenet_connectors.sources.readers import CsvDataReader, DataRow


M4_CAPTION_URL = "https://polybox.ethz.ch/index.php/s/MT3y9WdEebT8wfj/download/M4TimeSeriesCaptionDatasetV02.zip"
M4_CAPTION_SHA256 = "4acb748d94d7287fe58dbb5c3cf6314ea259010b7720f42506739269dec54edf"
RELEASE_FOLDER = "M4TimeSeriesCaptionDataset"
"""The folder in the archive that holds the CSV files."""
MAX_SERIES_CHARS = 1_000_000
"""The largest series cell that the connector reads, in characters. The largest cell of the release
holds a daily series of 9,919 values in 169,126 characters."""
SPLITS_PATH = Path(__file__).with_name("splits.json.xz")
"""The split of OpenTSLM's loader for the release, as generate_splits.py wrote it."""

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

    @property
    def series_file(self) -> str:
        """The name of the release file that holds the series of this frequency."""
        return f"m4_series_{self}.csv"

    @property
    def captions_file(self) -> str:
        """The name of the release file that holds the captions of this frequency."""
        return f"m4_captions_{self}.csv"

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


def _as_float32(values: Sequence[float]) -> np.ndarray:
    """Convert the values to float32, and check that the conversion keeps every value exactly.

    The release holds M4 values that OpenTSLM rounded to float32, so float32 is lossless for it.

    Returns:
        The values as float32.

    Raises:
        TimeNetConnectorError: If a value is not exactly a float32 value.
    """
    exact = np.array(values, dtype=np.float64)
    rounded = exact.astype(np.float32)
    if not np.array_equal(rounded.astype(np.float64), exact):
        raise TimeNetConnectorError("the series holds a value that is not exactly float32")
    return rounded


@dataclass(frozen=True)
class M4CaptionSeries:
    """One M4 series of the release, with its frequency and caption."""

    m4_id: str
    frequency: M4Frequency
    values: np.ndarray
    caption: str

    def create_record(self) -> Record:
        """Create the Record of the series, with its M4 id and frequency.

        Returns:
            The Record, not yet added to a dataset.
        """
        signal = Signal.from_values(self.values, spec=M4_SERIES, name="value", time_axis=OrdinalAxis())
        frequency = Annotation(key="frequency", value=self.frequency.value)
        return Record(
            sources=(Source(name="M4", signals=(signal,)),),
            annotations=(frequency,),
            metadata={"m4_id": self.m4_id},
        )

    def create_caption_task(self, record: Record, *, split: Split) -> AnswerTask:
        """Create the task that asks for the caption of the series.

        Returns:
            The task, not yet added to a dataset.
        """
        return AnswerTask(inputs=(record,), targets=(self.caption,), split=split)


class M4CaptionSplits(BaseModel):
    """The validation and test series of OpenTSLM's split, by M4 id.

    The release has no split. generate_splits.py ran the calls of OpenTSLM's loader once and wrote this
    split to splits.json.xz. Every series that the file does not name is in the train split.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    """SHA-256 digest of the release archive that the split was made from."""
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @model_validator(mode="after")
    def _each_series_in_one_split(self) -> Self:
        repeated = sorted(m4_id for m4_id, count in Counter(self.m4_ids).items() if count > 1)
        if repeated:
            raise ValueError(f"series {repeated} are named more than once")
        return self

    @property
    def m4_ids(self) -> tuple[str, ...]:
        """The validation and test series together."""
        return (*self.validation, *self.test)

    @classmethod
    def read_from_file(cls, path: Path) -> Self:
        """Read the split from an xz-compressed JSON file.

        Returns:
            The split.

        Raises:
            TimeNetConnectorError: If the file does not hold a valid split.
        """
        try:
            return cls.model_validate_json(lzma.decompress(path.read_bytes()))
        except (lzma.LZMAError, ValidationError) as exc:
            raise TimeNetConnectorError(f"{path.name} does not hold a valid split: {exc}") from exc

    def assign_split_to_each_series(self, m4_ids: Sequence[str]) -> list[Split]:
        """Assign a split to each series of the pinned release.

        Returns:
            The split of each series, in the order of ``m4_ids``.

        Raises:
            TimeNetConnectorError: If the split was made from another release, or names a series that
                the release does not have.
        """
        if self.source_sha256 != M4_CAPTION_SHA256:
            raise TimeNetConnectorError(
                f"the split was made from the release {self.source_sha256}, not from the pinned {M4_CAPTION_SHA256}"
            )
        unknown = sorted(set(self.m4_ids) - set(m4_ids))
        if unknown:
            raise TimeNetConnectorError(f"the split names series {unknown} that the release does not have")
        split_of = dict.fromkeys(self.validation, Split.VALIDATION) | dict.fromkeys(self.test, Split.TEST)
        return [split_of.get(m4_id, Split.TRAIN) for m4_id in m4_ids]


@dataclass(frozen=True)
class M4CaptionRelease:
    """The extracted release and the file that holds the split of its series."""

    directory: Path
    splits_path: Path

    def read_series(self) -> Iterator[M4CaptionSeries]:
        """Read every series of the release, in the order of OpenTSLM's loader.

        The loader reads the files of each frequency, and skips ``m4_series.csv`` and ``m4_captions.csv``.
        A malformed file raises a :class:`~timenet.errors.TimeNetConnectorError` that names the file and
        its row.

        Yields:
            Each series with its caption.
        """
        for frequency in M4Frequency:
            yield from self._read_frequency(frequency)

    def read_splits(self, m4_ids: Sequence[str]) -> list[Split]:
        """Read the split of each series from the split file.

        Returns:
            The split of each series, in the order of ``m4_ids``.
        """
        return M4CaptionSplits.read_from_file(self.splits_path).assign_split_to_each_series(m4_ids)

    def _read_frequency(self, frequency: M4Frequency) -> Iterator[M4CaptionSeries]:
        """Read the series of one frequency, paired with the caption in the same row of its caption file.

        Yields:
            Each series with its caption.

        Raises:
            TimeNetConnectorError: If the files do not pair one to one, or a series row is malformed.
        """
        series_path = self.directory / frequency.series_file
        captions_path = self.directory / frequency.captions_file
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
                yield M4CaptionSeries(
                    m4_id=m4_id,
                    frequency=frequency,
                    values=_as_float32(series_row.series),
                    caption=caption_row.caption,
                )
            except TimeNetConnectorError as exc:
                raise TimeNetConnectorError(f"{series_path.name} data row {number}: {exc}") from exc


class M4CaptionConnector(BaseConnector[M4CaptionRelease]):
    """Connector for the M4-Caption release of OpenTSLM."""

    async def download_async(self, cache_dir: Path) -> list[M4CaptionRelease]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            The extracted release, with the frozen split of the connector.
        """
        root = await ensure_archive(M4_CAPTION_URL, cache_dir, filename="m4_caption.zip", sha256=M4_CAPTION_SHA256)
        return [M4CaptionRelease(directory=root / RELEASE_FOLDER, splits_path=SPLITS_PATH)]

    def convert(self, raw_refs: list[M4CaptionRelease]) -> TimeFDataset:
        """Create one Record and one AnswerTask for each series.

        Args:
            raw_refs: The release from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (release,) = raw_refs
        # A daily series of the release does not fit in the default cell limit of the csv module.
        CsvDataReader.set_field_size_limit(MAX_SERIES_CHARS)
        all_series = list(release.read_series())
        splits = release.read_splits([series.m4_id for series in all_series])
        dataset = TimeFDataset(metadata=self.metadata())
        for series, split in zip(all_series, splits, strict=True):
            record = dataset.add_record(record=series.create_record())
            dataset.add_task(task=series.create_caption_task(record, split=split))
        return dataset


CONNECTOR = M4CaptionConnector
