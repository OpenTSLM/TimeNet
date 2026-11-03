"""Convert the pinned CAPTURE-24 zip into lazy Records and one classification task per window and scheme.

CAPTURE-24 holds about a day of 100 Hz wrist acceleration from each of 151 participants, labelled from
body-camera images and sleep diaries. A participant becomes one Record with one Source per gap-free
segment, each with lazy ``x``, ``y`` and ``z`` signals. Each 10 s window that the authors' benchmark code
keeps becomes one task per label scheme the benchmark scores. README.md explains how this follows the
authors.

The build has two passes. The first reads every recording's ``time`` and ``annotation`` columns and keeps
only small results: the segments, and each window's start and class. The second builds the Records and
streams the tasks. The acceleration values stay in the files until they are written.
"""

from collections.abc import Callable, Iterator
import csv
from dataclasses import dataclass
from functools import partial
import gzip
from itertools import pairwise
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import US_PER_S, Annotation, ClassificationTask, Split, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive, find_dir_containing


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The one zip of the Oxford Research Archive record, 6.9 GB."""
_ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ARCHIVE_ROOT = "capture24"
"""The archive's one top-level directory, which holds every file below."""

_METADATA_FILE = "metadata.csv"
_METADATA_COLUMNS = ("pid", "age", "sex")
_DICTIONARY_FILE = "annotation-label-dictionary.csv"
_DICTIONARY_LABEL_COLUMN = "annotation"
_RECORDING_FILE = re.compile(r"(P[0-9]{3})\.csv\.gz")
_RECORDING_COLUMNS = ["time", "x", "y", "z", "annotation"]
_AXES = ("x", "y", "z")

_DEVICE = "Axivity AX3"
_SAMPLE_RATE_HZ = 100
_SAMPLE_PERIOD_US = US_PER_S // _SAMPLE_RATE_HZ
# The source writes eight significant digits, more than float32 keeps.
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float64"
)

# What the authors' prepare_data.py (``--winsec 10``) and benchmark.py fix.
_WINDOW_US = 10 * US_PER_S
_WINDOW_SAMPLES = _WINDOW_US // _SAMPLE_PERIOD_US
_SCHEMES = {
    "WillettsSpecific2018": "willetts_specific_2018_vocabulary",
    "Walmsley2020": "walmsley_2020_vocabulary",
}
"""The label schemes the authors' benchmark.py scores, each with the dataset annotation that holds its
classes. Each task names that annotation as its ``target_schema``. The dictionary has four more schemes,
which have no tasks."""
_FIRST_TEST_PARTICIPANT = 101
"""The paper's test set is P101 to P151, and benchmark.py tests ``range(101, 152)``."""
_NO_LABEL = -1
"""The code of an unlabelled sample, which the source writes as an empty field."""


@dataclass(frozen=True)
class _Labels:
    """The label dictionary, as the windowing needs it."""

    code_of_raw: dict[str, int]
    """Each raw label's index, in file order."""
    vocabularies: dict[str, tuple[str, ...]]
    """Each scheme's distinct classes, sorted. A class's index breaks the authors' ties: ``Series.mode``
    sorts the modes it finds, and their ``make_windows`` takes the first."""
    class_of_raw: dict[str, np.ndarray]
    """Each scheme's class index per raw label, plus a last entry for an unlabelled sample, so that
    indexing with ``_NO_LABEL`` finds it."""


@dataclass(frozen=True)
class _Segment:
    """A run of samples exactly one period apart, between gaps."""

    first_row: int
    n_values: int
    offset_us: int
    """The first sample's time from the recording's first sample."""


@dataclass(frozen=True)
class _Participant:
    """One participant after the first pass: what the Record and the tasks need, not the samples."""

    pid: str
    age: str
    sex: str
    path: Path
    n_rows: int
    """The rows the recording held when it was read, which a reread must match."""
    start_local: str
    """The first sample's timestamp, formatted by datetime.isoformat(): no zone, and a randomized date."""
    segments: tuple[_Segment, ...]
    window_starts_us: np.ndarray
    """Each kept window's first sample's time from the recording's first sample."""
    window_classes: dict[str, np.ndarray]
    """Each kept window's class per scheme, as an index into that scheme's vocabulary."""

    @property
    def split(self) -> Split:
        """Return the participant's split in the authors' benchmark.py."""
        return Split.TEST if int(self.pid.removeprefix("P")) >= _FIRST_TEST_PARTICIPANT else Split.TRAIN


def _require_values(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    """Check that every row has a value in each of columns.

    A short row reads as None and a blank field as "", and neither is a value.

    Raises:
        TimeFFormatError: If a row has no value in a column.
    """
    for line, row in enumerate(rows, start=2):
        for column in columns:
            if not row[column]:
                raise TimeFFormatError(f"{path}:{line}: no value for {column!r}")


def _read_labels(path: Path) -> _Labels:
    """Read the label dictionary: each raw label and its class in each scheme.

    Returns:
        The dictionary, as the windowing needs it.

    Raises:
        TimeFFormatError: If a column is missing, a row has no value in a needed column, or a raw label is
            listed twice.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    columns = [_DICTIONARY_LABEL_COLUMN, *(f"label:{scheme}" for scheme in _SCHEMES)]
    if not set(columns) <= set(header):
        raise TimeFFormatError(f"{path}: expected the columns {columns}, got {header}")
    _require_values(path, rows, columns)
    raw_labels = [row[_DICTIONARY_LABEL_COLUMN] for row in rows]
    if len(set(raw_labels)) != len(raw_labels):
        raise TimeFFormatError(f"{path}: a raw label is listed twice")
    vocabularies = {}
    class_of_raw = {}
    for scheme in _SCHEMES:
        classes = [row[f"label:{scheme}"] for row in rows]
        vocabularies[scheme] = tuple(sorted(set(classes)))
        index = {name: code for code, name in enumerate(vocabularies[scheme])}
        class_of_raw[scheme] = np.array([index[name] for name in classes] + [len(index)], dtype=np.int16)
    return _Labels(
        code_of_raw={label: code for code, label in enumerate(raw_labels)},
        vocabularies=vocabularies,
        class_of_raw=class_of_raw,
    )


def _read_people(root: Path) -> list[tuple[str, str, str, Path]]:
    """Read ``metadata.csv`` and pair each row with its recording.

    Returns:
        One ``(pid, age, sex, recording path)`` per participant, in id order.

    Raises:
        TimeFFormatError: If a column is missing, a row has no value in a needed column, a pid repeats, or
            the table and the recordings don't name the same participants.
    """
    path = root / _METADATA_FILE
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    if not set(_METADATA_COLUMNS) <= set(header):
        raise TimeFFormatError(f"{path}: expected the columns {list(_METADATA_COLUMNS)}, got {header}")
    _require_values(path, rows, list(_METADATA_COLUMNS))
    pids = [row["pid"] for row in rows]
    if len(set(pids)) != len(pids):
        raise TimeFFormatError(f"{path}: a pid is listed twice")
    recordings = {
        match[1]: file for file in root.iterdir() if (match := _RECORDING_FILE.fullmatch(file.name)) is not None
    }
    if set(recordings) != set(pids):
        raise TimeFFormatError(
            f"{root}: recordings missing for {sorted(set(pids) - set(recordings))}, "
            f"unlisted recordings {sorted(set(recordings) - set(pids))}"
        )
    return [
        (row["pid"], row["age"], row["sex"], recordings[row["pid"]]) for row in sorted(rows, key=lambda r: r["pid"])
    ]


def _read_participant(pid: str, age: str, sex: str, path: Path, labels: _Labels) -> _Participant:
    """Read a recording's ``time`` and ``annotation`` columns: its segments and its kept windows.

    An empty ``annotation`` field is an unlabelled sample. No other value is.

    Returns:
        The participant, without the samples.

    Raises:
        TimeFFormatError: If the file can't be read, the header is wrong, the file has no rows, a timestamp is
            missing or unreadable, a step is shorter than one period, a raw label isn't in the dictionary, or
            a kept window would cross a gap.
    """
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if header != _RECORDING_COLUMNS:
        raise TimeFFormatError(f"{path}: expected the header {','.join(_RECORDING_COLUMNS)}, got {header}")
    try:
        table = pa_csv.read_csv(
            path,
            convert_options=pa_csv.ConvertOptions(
                include_columns=["time", "annotation"],
                column_types={"time": pa.timestamp("us"), "annotation": pa.dictionary(pa.int32(), pa.string())},
                strings_can_be_null=True,
                null_values=[""],
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows == 0:
        raise TimeFFormatError(f"{path}: holds no samples")
    times = table["time"].combine_chunks()
    if times.null_count:
        raise TimeFFormatError(f"{path}: {times.null_count} samples have no timestamp")
    stamps = times.cast(pa.int64()).to_numpy()
    steps = np.diff(stamps)
    # Reject any step shorter than one period: it is neither a regular step nor a gap.
    if (steps < _SAMPLE_PERIOD_US).any():
        row = int(np.flatnonzero(steps < _SAMPLE_PERIOD_US)[0]) + 1
        raise TimeFFormatError(f"{path}: the timestamp of row {row} doesn't follow the one before it by a full period")
    offsets = stamps - stamps[0]
    raw_codes = _raw_codes(path, table["annotation"].combine_chunks(), labels)
    # A gap is any step but one period. Each starts a new segment.
    bounds = [0, *(np.flatnonzero(steps != _SAMPLE_PERIOD_US) + 1).tolist(), len(stamps)]
    segments = tuple(
        _Segment(first_row=start, n_values=end - start, offset_us=int(offsets[start]))
        for start, end in pairwise(bounds)
    )
    first_rows, classes = _windows(offsets, raw_codes, labels)
    # A step between one and two periods is a gap too, and a bin can still hold 1000 samples across it.
    first_row_of_segment = np.array([segment.first_row for segment in segments])
    last_rows = first_rows + _WINDOW_SAMPLES - 1
    crossing = np.searchsorted(first_row_of_segment, first_rows, side="right") != np.searchsorted(
        first_row_of_segment, last_rows, side="right"
    )
    if crossing.any():
        row = int(first_rows[crossing][0])
        raise TimeFFormatError(f"{path}: the window at row {row} holds 1000 samples across a gap")
    return _Participant(
        pid=pid,
        age=age,
        sex=sex,
        path=path,
        n_rows=len(stamps),
        start_local=times[0].as_py().isoformat(),
        segments=segments,
        window_starts_us=offsets[first_rows],
        window_classes=classes,
    )


def _raw_codes(path: Path, column: pa.DictionaryArray, labels: _Labels) -> np.ndarray:
    """Return each sample's raw label as an index into the dictionary, or ``_NO_LABEL``.

    Raises:
        TimeFFormatError: If the recording writes a raw label the dictionary doesn't list.
    """
    written = column.dictionary.to_pylist()
    unknown = sorted(label for label in written if label not in labels.code_of_raw)
    if unknown:
        raise TimeFFormatError(f"{path}: raw labels missing from the label dictionary: {unknown}")
    # One extra slot maps a null's filled index to _NO_LABEL.
    codes = np.array([labels.code_of_raw[label] for label in written] + [_NO_LABEL], dtype=np.int16)
    return codes[column.indices.fill_null(len(written)).to_numpy()]


def _windows(
    offsets_us: np.ndarray, raw_codes: np.ndarray, labels: _Labels
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the windows the authors' ``prepare_data.make_windows`` keeps, with each one's classes.

    ``make_windows`` groups the samples into 10 s bins anchored at the first sample. After a gap the bins
    stay on the first segment's grid. It keeps a bin with exactly 1000 samples and at least one labelled
    sample. A bin that a gap cuts usually has fewer samples. The caller rejects the rare window that holds
    1000 samples across a gap, so no kept window crosses one.

    A window's class in a scheme is the most common class among its labelled samples. A tie goes to the
    first class in string order.

    It also drops a window with a NaN acceleration value. This function never meets one: a NaN fails the
    build when its signal is read.

    Returns:
        Each kept window's first row, and its class index per scheme.
    """
    bins = offsets_us // _WINDOW_US
    bin_starts = np.flatnonzero(np.diff(bins, prepend=bins[0] - 1))
    full = bin_starts[np.diff(bin_starts, append=len(bins)) == _WINDOW_SAMPLES]
    window_codes = raw_codes[full[:, None] + np.arange(_WINDOW_SAMPLES)]
    labelled = (window_codes != _NO_LABEL).any(axis=1)
    kept, window_codes = full[labelled], window_codes[labelled]
    classes = {}
    for scheme, vocabulary in labels.vocabularies.items():
        n_classes = len(vocabulary)
        # Unlabelled samples land in a last bin, which the count then drops.
        per_sample = labels.class_of_raw[scheme][window_codes]
        row_offsets = np.arange(len(kept))[:, None] * (n_classes + 1)
        counts = np.bincount((row_offsets + per_sample).ravel(), minlength=len(kept) * (n_classes + 1))
        counts = counts.reshape(len(kept), n_classes + 1)[:, :n_classes]
        # argmax takes the first of equal counts: the class first in string order.
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return kept, classes


def _read_acceleration(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    """Read a recording's x, y and z columns as float64.

    Returns:
        One array per acceleration column.

    Raises:
        TimeFFormatError: If a value is missing, unreadable or not finite, or the row count changed.
    """
    try:
        table = pa_csv.read_csv(
            path,
            # Only an empty field is missing, so a written nan reaches the finite check below.
            convert_options=pa_csv.ConvertOptions(
                include_columns=list(_AXES), column_types=dict.fromkeys(_AXES, pa.float64()), null_values=[""]
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows != n_rows:
        raise TimeFFormatError(f"{path}: holds {table.num_rows} rows, {n_rows} when it was read; the file changed")
    values = {}
    for column in _AXES:
        array = table[column].combine_chunks()
        if array.null_count:
            raise TimeFFormatError(f"{path}: {array.null_count} samples have no {column} value")
        values[column] = array.to_numpy()
        if not np.isfinite(values[column]).all():
            raise TimeFFormatError(f"{path}: a {column} value is not finite")
    return values


class _OneRecording:
    """Read a recording's acceleration once for all its signals, and hold only the latest recording.

    A Record's signals load one after another, so one recording suffices. lru_cache(maxsize=1) would
    read the next recording before it dropped the last, holding two at once.
    """

    def __init__(self) -> None:
        self._key: tuple[Path, int] | None = None
        self._values: dict[str, np.ndarray] = {}

    def __call__(self, path: Path, n_rows: int) -> dict[str, np.ndarray]:
        """Return the recording's x, y and z, reading it unless it was the last one read.

        Returns:
            One array per acceleration column.
        """
        if self._key != (path, n_rows):
            # Release the previous recording before reading the next.
            self._key, self._values = None, {}
            self._values = _read_acceleration(path, n_rows)
            self._key = (path, n_rows)
        return self._values


def _load_segment(
    read: Callable[[Path, int], dict[str, np.ndarray]], participant: _Participant, segment: _Segment, column: str
) -> pa.Array:
    """Return one segment's values of one acceleration column.

    Returns:
        The segment's float64 values.
    """
    values = read(participant.path, participant.n_rows)[column]
    return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=pa.float64())


def _record(participant: _Participant, read: Callable[[Path, int], dict[str, np.ndarray]]) -> Record:
    """Build a participant's Record: one Source of lazy x, y and z signals per segment.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    sources = []
    for index, segment in enumerate(participant.segments):
        time_axis = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = tuple(
            Signal.from_loader(
                name=column,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(_load_segment, read, participant, segment, column),
            )
            for column in _AXES
        )
        sources.append(Source(name=f"{_DEVICE} segment {index}", signals=signals))
    record = Record(sources=tuple(sources), metadata={"subject_id": participant.pid, "file": participant.path.name})
    record.add_annotations(
        (
            Annotation(key="age", value=participant.age),
            Annotation(key="sex", value=participant.sex),
            Annotation(key="recording_start_local", value=participant.start_local),
        )
    )
    return record


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from the pinned zip of its Oxford Research Archive record."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301 - the connector interface
        """Fetch the pinned archive, check its digest, and extract it once.

        Returns:
            The extracted ``capture24`` directory.
        """
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=ARCHIVE_SHA256)
        return [find_dir_containing(root, f"{_ARCHIVE_ROOT}/{_METADATA_FILE}")]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each recording into one lazy Record, and stream one task per window and scheme.

        Every recording is read once up front to check it and keep its segments and windows. The
        acceleration values stay in the files until written, so the files must stay available until then.

        Args:
            raw_refs: The one extracted ``capture24`` directory.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If ``raw_refs`` isn't one directory.
            TimeFFormatError: If a table or recording is malformed.
        """  # noqa: DOC502 - raised by the read helpers
        if len(raw_refs) != 1:
            raise TimeFValidationError(f"CAPTURE-24 loads from one directory, got {raw_refs}")
        (root,) = raw_refs
        labels = _read_labels(root / _DICTIONARY_FILE)
        # One participant's samples are held at a time: the first pass keeps only small results.
        participants = [_read_participant(pid, age, sex, path, labels) for pid, age, sex, path in _read_people(root)]
        read = _OneRecording()
        records = [_record(participant, read) for participant in participants]

        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_url", ARCHIVE_URL), ("source_sha256", ARCHIVE_SHA256)):
            dataset.annotate(Annotation(key=key, value=value))
        for record in records:
            dataset.add_record(record=record)
        for scheme, key in _SCHEMES.items():
            dataset.annotate(Annotation(key=key, value=list(labels.vocabularies[scheme])))

        def tasks() -> Iterator[ClassificationTask]:
            for participant, record in zip(participants, records, strict=True):
                for window, start_us in enumerate(participant.window_starts_us.tolist()):
                    for scheme, key in _SCHEMES.items():
                        yield ClassificationTask(
                            inputs=(record,),
                            scope=TimeInterval.micros(start_us, start_us + _WINDOW_US),
                            targets=(labels.vocabularies[scheme][participant.window_classes[scheme][window]],),
                            target_schema=key,
                            split=participant.split,
                        )

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = Capture24Connector
