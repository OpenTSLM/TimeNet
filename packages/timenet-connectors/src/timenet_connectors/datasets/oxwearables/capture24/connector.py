"""The CAPTURE-24 connector: wrist accelerometry of 151 participants over about a day each.

One record is one participant file, ``P<nnn>.csv.gz``. Its rows hold a local timestamp, the
three acceleration axes in g at 100 Hz, and the activity annotation of that row. The release
resamples every file to 100 Hz, so the timestamps step by exactly 10 ms. Eight files jump forward
once: six where the recording stops for hours, and P016 and P017 where the local clock moves an
hour ahead. A segment is a stretch of rows 10 ms apart. Each one becomes a child source with a
regular axis that starts at the time offset its first timestamp states, and nothing fills a gap.
The connector keeps the first timestamp of the file. The rest of the time column holds nothing
more.

The annotation column holds one text per row. It is blank where nobody annotated the
participant. Each run of equal non-blank rows inside one segment becomes one span annotation on
the record and one ``ClassificationTask`` scoped to that run. Blank rows get neither.

The target of each task is the annotation text of the release, such as
``7030 sleeping;MET 0.95``. ``annotation-label-dictionary.csv`` maps each text to six coarser
label schemes, such as ``label:Walmsley2020``. The dataset keeps that table as one annotation,
so whoever trains picks the scheme.
"""

from collections.abc import Callable
import csv
from dataclasses import dataclass
import gzip
from itertools import pairwise
import json
from pathlib import Path
import threading

import numpy as np
import pyarrow as pa
from pyarrow import csv as pa_csv
import pyarrow.compute as pc

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive, find_dir_containing


# The release on the Oxford University Research Archive, as one zip.
CAPTURE24_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
CAPTURE24_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"

METADATA_FILE = "metadata.csv"
DICTIONARY_FILE = "annotation-label-dictionary.csv"
PARTICIPANT_COUNT = 151

PARTICIPANT_COLUMNS = ("time", "x", "y", "z", "annotation")
METADATA_COLUMNS = ("pid", "age", "sex")
AXES = ("x", "y", "z")

# The key of the dictionary column that holds the annotation text.
_ANNOTATION_COLUMN = "annotation"

SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD_US = 10_000

ACCELERATION = TimeSeriesSpec(
    spec_type="accelerometer",
    name="Accelerometer",
    unit_value=ureg.standard_gravity,
    dtype="float64",
)

ANNOTATION_KEY = "annotation"


@dataclass(frozen=True)
class Capture24Source:
    """The extracted release: the participant files and the two tables beside them."""

    root: Path


@dataclass(frozen=True)
class ParticipantScoring:
    """What one participant file states besides the acceleration.

    ``segments`` holds one ``(first_row, end_row, offset_us)`` entry for each stretch of rows
    10 ms apart. ``runs`` holds one ``(label, start_us, end_us)`` entry for each run of equal
    non-blank annotations inside one segment. Every end is exclusive, and every time is from the
    first timestamp of the file.
    """

    n_values: int
    start_local: str
    segments: tuple[tuple[int, int, int], ...]
    runs: tuple[tuple[str, int, int], ...]


def read_label_dictionary(path: Path) -> dict[str, dict[str, str]]:
    """Read the table that maps each annotation text to its label in each scheme.

    Returns:
        The label of each scheme, keyed by annotation text, in file order.

    Raises:
        TimeFFormatError: If the table has no scheme column, a blank cell, or an annotation twice.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        schemes = fields[1:]
        if fields[:1] != (_ANNOTATION_COLUMN,) or not schemes or not all(s.startswith("label:") for s in schemes):
            raise TimeFFormatError(f"{path.name}: expected an annotation column and label:* columns, got {fields}")
        dictionary: dict[str, dict[str, str]] = {}
        for row in reader:
            text = row[_ANNOTATION_COLUMN]
            if not text or any(not row[scheme] for scheme in schemes):
                raise TimeFFormatError(f"{path.name}: the row for {text!r} has a blank cell")
            if text in dictionary:
                raise TimeFFormatError(f"{path.name}: states {text!r} twice")
            dictionary[text] = {scheme: row[scheme] for scheme in schemes}
    if not dictionary:
        raise TimeFFormatError(f"{path.name}: holds no annotation")
    return dictionary


def read_participants(path: Path) -> dict[str, tuple[str, str]]:
    """Read the age group and the sex of each participant.

    Returns:
        ``(age, sex)`` keyed by participant id.

    Raises:
        TimeFFormatError: If the header differs, a cell is blank, or a participant appears twice.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METADATA_COLUMNS:
            raise TimeFFormatError(f"{path.name}: expected the columns {METADATA_COLUMNS}, got {reader.fieldnames}")
        participants: dict[str, tuple[str, str]] = {}
        for row in reader:
            pid, age, sex = (row[column] for column in METADATA_COLUMNS)
            if not pid or not age or not sex:
                raise TimeFFormatError(f"{path.name}: the row for {pid!r} has a blank cell")
            if pid in participants:
                raise TimeFFormatError(f"{path.name}: states {pid!r} twice")
            participants[pid] = (age, sex)
    return participants


def _check_header(path: Path) -> None:
    """Check that a participant file has the columns of the release, in order.

    Raises:
        TimeFFormatError: If the header differs.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        header = tuple(next(csv.reader(handle), ()))
    if header != PARTICIPANT_COLUMNS:
        raise TimeFFormatError(f"{path.name}: expected the columns {PARTICIPANT_COLUMNS}, got {header}")


def _read_columns(path: Path, columns: dict[str, pa.DataType]) -> pa.Table:
    """Read some columns of a participant file. Only a blank cell reads as null.

    Returns:
        The columns, in the types given.

    Raises:
        TimeFFormatError: If a cell does not parse as its type.
    """
    try:
        return pa_csv.read_csv(
            path,
            convert_options=pa_csv.ConvertOptions(
                include_columns=list(columns),
                column_types=columns,
                null_values=[""],
                strings_can_be_null=True,
            ),
        )
    except pa.ArrowInvalid as exc:
        raise TimeFFormatError(f"{path.name}: {exc}") from exc


def _read_offsets(path: Path, text: pa.ChunkedArray) -> np.ndarray:
    """Give the time offset of each row from the first, and check that rows are 10 ms or more apart.

    Returns:
        The offsets in microseconds.

    Raises:
        TimeFFormatError: If a timestamp is blank or does not parse, or two are less than 10 ms apart.
    """
    try:
        stamps = pc.cast(text, pa.timestamp("us")).to_numpy().astype(np.int64)
    except (pa.ArrowInvalid, ValueError) as exc:
        raise TimeFFormatError(f"{path.name}: a timestamp is blank or does not parse") from exc
    offsets = stamps - stamps[0]
    short = np.flatnonzero(np.diff(offsets) < SAMPLE_PERIOD_US)
    if short.size:
        row = int(short[0]) + 1
        raise TimeFFormatError(
            f"{path.name}: row {row} is at {text[row]}, less than {SAMPLE_PERIOD_US} us after {text[row - 1]}"
        )
    return offsets


def _read_runs(path: Path, annotation: pa.ChunkedArray, labels: LabelSet) -> tuple[list[str], np.ndarray]:
    """Give the annotation texts of a file and the code of each row, which is -1 where it is blank.

    Returns:
        The texts, and one index into them for each row.

    Raises:
        TimeFFormatError: If an annotation is not in the label dictionary.
    """
    encoded = annotation.combine_chunks().dictionary_encode()
    texts = encoded.dictionary.to_pylist()
    unknown = sorted(set(texts) - set(labels.names))
    if unknown:
        raise TimeFFormatError(f"{path.name}: annotations not in {DICTIONARY_FILE}: {unknown}")
    return texts, pc.fill_null(encoded.indices, -1).to_numpy()


def read_scoring(path: Path, labels: LabelSet) -> ParticipantScoring:
    """Read the time and annotation columns of a participant file. The axes stay unread.

    Args:
        path: The ``P<nnn>.csv.gz`` file.
        labels: The annotation texts the release defines.

    Returns:
        The number of rows, the first timestamp, the segments, and the annotated runs.

    Raises:
        TimeFFormatError: If the file has no row.
    """
    _check_header(path)
    table = _read_columns(path, {"time": pa.string(), "annotation": pa.string()})
    n_values = table.num_rows
    if n_values == 0:
        raise TimeFFormatError(f"{path.name}: holds no row")
    offsets = _read_offsets(path, table.column("time"))
    texts, codes = _read_runs(path, table.column("annotation"), labels)

    breaks = np.flatnonzero(np.diff(offsets) > SAMPLE_PERIOD_US) + 1
    segments = tuple(
        (start, end, int(offsets[start])) for start, end in pairwise(np.concatenate(([0], breaks, [n_values])).tolist())
    )
    # A run ends where the annotation changes or a segment does.
    bounds = np.union1d(np.flatnonzero(np.diff(codes)) + 1, breaks)
    runs = tuple(
        (texts[codes[start]], int(offsets[start]), int(offsets[end - 1]) + SAMPLE_PERIOD_US)
        for start, end in pairwise(np.concatenate(([0], bounds, [n_values])).tolist())
        if codes[start] >= 0
    )
    return ParticipantScoring(n_values=n_values, start_local=str(table.column("time")[0]), segments=segments, runs=runs)


class _AxisReader:
    """Read the three axes of one participant file, and keep the last file read.

    The writer reads ``x``, ``y`` and ``z`` of a record one after the other. One parse of the
    file serves all three.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._table: pa.Table | None = None

    def loader(self, path: Path, axis: str, n_values: int, rows: tuple[int, int]) -> Callable[[], pa.Array]:
        """Give a loader for one axis of one segment of a participant file.

        Args:
            path: The participant file.
            axis: ``x``, ``y`` or ``z``.
            n_values: The number of rows of the whole file.
            rows: The first row of the segment and the row after its last.

        Returns:
            A callable that reads the axis of the segment as float64.
        """
        first, end = rows
        return lambda: self._read(path, n_values).column(axis).slice(first, end - first).combine_chunks()

    def _read(self, path: Path, n_values: int) -> pa.Table:
        """Parse the axes of a file, or give the parse already held.

        Returns:
            The three axes.

        Raises:
            TimeFFormatError: If the row count changed since ``convert``, or a value is blank or
                not finite.
        """
        with self._lock:
            if self._path != path or self._table is None:
                table = _read_columns(path, dict.fromkeys(AXES, pa.float64()))
                if table.num_rows != n_values:
                    raise TimeFFormatError(f"{path.name}: holds {table.num_rows} rows, {n_values} when converted")
                for axis in AXES:
                    values = table.column(axis)
                    if values.null_count or not np.isfinite(values.to_numpy()).all():
                        raise TimeFFormatError(f"{path.name}: axis {axis} holds a blank or non-finite value")
                self._path, self._table = path, table
            return self._table


class Capture24Connector(BaseConnector[Capture24Source]):
    """Connector for CAPTURE-24 (``oxwearables/capture24``)."""

    async def download_async(self, cache_dir: Path) -> list[Capture24Source]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory for the archive and its extracted contents.

        Returns:
            One handle to the extracted release.

        Raises:
            TimeNetDownloadError: If the release does not hold the two tables and 151 participant files.
        """
        root = find_dir_containing(
            await ensure_archive(CAPTURE24_URL, cache_dir, filename="capture24.zip", sha256=CAPTURE24_SHA256),
            METADATA_FILE,
        )
        participants = sorted(root.glob("P*.csv.gz"))
        if not (root / DICTIONARY_FILE).is_file() or len(participants) != PARTICIPANT_COUNT:
            raise TimeNetDownloadError(
                f"CAPTURE-24 at {root}: expected {DICTIONARY_FILE} and {PARTICIPANT_COUNT} participant files, "
                f"found {len(participants)}"
            )
        return [Capture24Source(root=root)]

    def convert(self, raw_refs: list[Capture24Source]) -> TimeFDataset:
        """Build one record for each participant file, and one task for each annotated run.

        Args:
            raw_refs: The one handle from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeFFormatError: If a participant file has no row in ``metadata.csv``.
        """
        root = raw_refs[0].root
        dictionary = read_label_dictionary(root / DICTIONARY_FILE)
        participants = read_participants(root / METADATA_FILE)
        labels = LabelSet(name=ANNOTATION_KEY, labels=tuple(Label(name=text) for text in dictionary))
        axes = _AxisReader()

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(
            Annotation(
                key="annotation_label_dictionary",
                value=json.dumps(dictionary, ensure_ascii=False),
                description=f"{DICTIONARY_FILE} as JSON: the label of each scheme, keyed by annotation text.",
            )
        )
        for path in sorted(root.glob("P*.csv.gz")):
            participant_id = path.name.removesuffix(".csv.gz")
            if participant_id not in participants:
                raise TimeFFormatError(f"{path.name}: {participant_id} has no row in {METADATA_FILE}")
            age, sex = participants[participant_id]
            scoring = read_scoring(path, labels)

            segments = tuple(
                Source(
                    name=f"segment {number}",
                    signals=tuple(
                        Signal.from_loader(
                            spec=ACCELERATION,
                            name=axis,
                            time_axis=RegularAxis.from_rate_hz(SAMPLE_RATE_HZ, offset_us=offset_us),
                            loader=axes.loader(path, axis, scoring.n_values, (first, end)),
                            n_values=end - first,
                            null_count=0,
                        )
                        for axis in AXES
                    ),
                )
                for number, (first, end, offset_us) in enumerate(scoring.segments, start=1)
            )
            record = dataset.add_record(
                record=Record(
                    sources=(Source(name="Axivity AX3", sources=segments),),
                    metadata={"participant_id": participant_id},
                )
            )
            record.add_annotations(
                (
                    Annotation(key="subject_ids", value=[participant_id]),
                    Annotation(key="age", value=age, description="The age group in years."),
                    Annotation(key="sex", value=sex),
                    Annotation(
                        key="recording_start_local",
                        value=scoring.start_local,
                        description="The first timestamp of the file, a local wall clock with no zone.",
                    ),
                )
            )
            for label, start_us, end_us in scoring.runs:
                span = TimeInterval.micros(start_us, end_us)
                record.annotate(Annotation(key=ANNOTATION_KEY, value=label, span=span))
                dataset.add_task(
                    task=ClassificationTask(inputs=(record,), targets=(label,), label_set=labels, scope=span)
                )
        return dataset


CONNECTOR = Capture24Connector
