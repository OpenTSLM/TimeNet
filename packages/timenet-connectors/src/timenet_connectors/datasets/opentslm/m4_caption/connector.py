"""The M4-Caption connector: M4 competition series, each with a caption that GPT-4o wrote.

The OpenTSLM release holds the train part of all 100,000 M4 series, one CSV of series and one CSV of
captions for each of the six M4 frequencies. Each series becomes one Record and one AnswerTask. The
target is the caption. The release has no split, so the connector computes the split of OpenTSLM's
loader and checks it against a pinned digest.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
from math import ceil
from pathlib import Path
import re

import numpy as np
from pydantic import FiniteFloat, Json

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
FREQUENCIES = ("Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly")
"""The M4 frequencies, in the order that OpenTSLM's loader reads them. The split depends on this order."""
MAX_SERIES_CHARS = 1_000_000
"""The largest series cell that the connector reads, in characters. The largest cell of the release
holds a daily series of 9,919 values in 169,126 characters."""
OPENTSLM_SPLIT_SHA256 = "370cc2580bcdb912e13a546c1a84f3d52d7e2098db07cfe2873ffde8270f5256"
"""The digest of the split that OpenTSLM's loader gives the release, as generate_splits.py computed it."""

# The release states no unit, and the M4 series come from many domains.
M4_SERIES = TimeSeriesSpec(spec_type="m4_series", name="M4 series", unit_value=None, dtype="float32")

_SERIES_ID = re.compile(r"series-(?P<m4_id>(?P<letter>[DHMQWY])[1-9][0-9]*)")
"""A release id: the M4 id, such as M1, after a ``series-`` prefix that OpenTSLM added."""


class _SeriesRow(DataRow):
    """A row of a series file: the release id and the values as a JSON list."""

    id: str
    series: Json[list[FiniteFloat]]


class _CaptionRow(DataRow):
    """A row of a caption file: the release id and the caption."""

    id: str
    caption: str


def _parse_m4_id(series_id: str, frequency: str) -> str:
    """Get the M4 id from a release id, and check that it belongs to the frequency of its file.

    Returns:
        The M4 id, such as ``M1``.

    Raises:
        TimeNetConnectorError: If the id is not an M4 series id of the frequency.
    """
    match = _SERIES_ID.fullmatch(series_id)
    if match is None or match["letter"] != frequency[0]:
        raise TimeNetConnectorError(f"{series_id!r} is not a {frequency} M4 series id")
    return match["m4_id"]


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
    frequency: str
    values: np.ndarray
    caption: str

    def create_record(self) -> Record:
        """Create the Record of the series, with its M4 id and frequency.

        Returns:
            The Record, not yet added to a dataset.
        """
        signal = Signal.from_values(self.values, spec=M4_SERIES, name="value", time_axis=OrdinalAxis())
        frequency = Annotation(key="frequency", value=self.frequency)
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


@dataclass(frozen=True)
class OpenTSLMSplit:
    """The 80/10/10 split of ``create_combined_dataset`` in OpenTSLM's M4 loader.

    The loader calls ``train_test_split`` of the datasets library twice, without stratification. That
    function shuffles the rows with ``numpy.random.default_rng(seed).permutation``: the first rows of the
    permutation are the test part and the rest are the train part. NumPy does not promise the same
    permutation in every version, so the split is checked against a digest.
    """

    sha256: str
    """The digest that :meth:`digest` must give for the split."""

    @staticmethod
    def digest(m4_ids: Sequence[str], splits: Sequence[Split]) -> str:
        """Compute the digest of a split.

        The digest is the SHA-256 of one line per series, in order. Each line is the M4 id, a tab, the
        split, and a newline.

        Returns:
            The digest in hexadecimal.
        """
        lines = "".join(f"{m4_id}\t{split}\n" for m4_id, split in zip(m4_ids, splits, strict=True))
        return hashlib.sha256(lines.encode()).hexdigest()

    def assign_splits(self, m4_ids: Sequence[str]) -> list[Split]:
        """Assign the split of OpenTSLM's loader to each series, in the order of the loader.

        Returns:
            The split of each series.

        Raises:
            TimeNetConnectorError: If the split does not match the pinned digest.
        """
        splits = [Split.TRAIN] * len(m4_ids)
        all_positions = np.arange(len(m4_ids))
        train_and_validation = self._split_off(all_positions, splits, Split.TEST, seed=42, size=0.1)
        # The loader keeps the share of the validation part at 10 percent of all series.
        self._split_off(train_and_validation, splits, Split.VALIDATION, seed=43, size=0.1 / (1.0 - 0.1))
        if self.digest(m4_ids, splits) != self.sha256:
            raise TimeNetConnectorError(
                f"the OpenTSLM split of {len(m4_ids)} series does not match the pinned digest {self.sha256}"
            )
        return splits

    @staticmethod
    def _split_off(positions: np.ndarray, splits: list[Split], split: Split, *, seed: int, size: float) -> np.ndarray:
        """Do one ``train_test_split`` call of the datasets library on the series at ``positions``.

        The series of the part that the call names "test" get ``split``.

        Returns:
            The positions of the part that the call names "train", in the order of the permutation.
        """
        permutation = np.random.default_rng(seed).permutation(len(positions))
        n_split = ceil(size * len(positions))
        for position in positions[permutation[:n_split]]:
            splits[position] = split
        return positions[permutation[n_split:]]


@dataclass(frozen=True)
class M4CaptionRelease:
    """The extracted release and the split that its series must get."""

    directory: Path
    split: OpenTSLMSplit

    def read_series(self) -> Iterator[M4CaptionSeries]:
        """Read every series of the release, in the order of OpenTSLM's loader.

        The loader reads the files of each frequency, and skips ``m4_series.csv`` and ``m4_captions.csv``.
        A malformed file raises a :class:`~timenet.errors.TimeNetConnectorError` that names the file and
        its row.

        Yields:
            Each series with its caption.
        """
        for frequency in FREQUENCIES:
            yield from self._read_frequency(frequency)

    def _read_frequency(self, frequency: str) -> Iterator[M4CaptionSeries]:
        """Read the series of one frequency, paired with the caption in the same row of its caption file.

        Yields:
            Each series with its caption.

        Raises:
            TimeNetConnectorError: If the files do not pair one to one, or a series row is malformed.
        """
        series_path = self.directory / f"m4_series_{frequency}.csv"
        captions_path = self.directory / f"m4_captions_{frequency}.csv"
        CsvDataReader.set_field_size_limit(MAX_SERIES_CHARS)
        series_rows = list(CsvDataReader(series_path, _SeriesRow, expected_columns=("id", "series")).iter_rows())
        caption_rows = list(CsvDataReader(captions_path, _CaptionRow, expected_columns=("id", "caption")).iter_rows())
        if len(series_rows) != len(caption_rows):
            raise TimeNetConnectorError(
                f"{series_path.name} has {len(series_rows)} series rows but {len(caption_rows)} caption rows"
            )
        for number, (series_row, caption_row) in enumerate(zip(series_rows, caption_rows, strict=True), start=1):
            try:
                m4_id = _parse_m4_id(series_row.id, frequency)
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
            The extracted release, with the pinned split.
        """
        root = await ensure_archive(M4_CAPTION_URL, cache_dir, filename="m4_caption.zip", sha256=M4_CAPTION_SHA256)
        return [M4CaptionRelease(directory=root / RELEASE_FOLDER, split=OpenTSLMSplit(sha256=OPENTSLM_SPLIT_SHA256))]

    def convert(self, raw_refs: list[M4CaptionRelease]) -> TimeFDataset:
        """Create one Record and one AnswerTask for each series.

        Args:
            raw_refs: The release from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (release,) = raw_refs
        all_series = list(release.read_series())
        splits = release.split.assign_splits([series.m4_id for series in all_series])
        dataset = TimeFDataset(metadata=self.metadata())
        for series, split in zip(all_series, splits, strict=True):
            record = dataset.add_record(record=series.create_record())
            dataset.add_task(task=series.create_caption_task(record, split=split))
        return dataset


CONNECTOR = M4CaptionConnector
