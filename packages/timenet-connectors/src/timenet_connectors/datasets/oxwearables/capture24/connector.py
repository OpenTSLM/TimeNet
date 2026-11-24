"""Convert the pinned CAPTURE-24 archive to records and activity classification tasks.

Each participant has one record with wrist acceleration and activity labels.
Tasks cover 10-second windows in the two benchmark label schemes.
"""

from collections.abc import Callable, Iterator
import csv
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import partial
import gzip
import hashlib
from itertools import pairwise
from pathlib import Path
import re
from typing import NamedTuple, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    LabelSet,
    Split,
    TimeInterval,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The URL of the 6.9 GB CAPTURE-24 archive."""
_ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ARCHIVE_ROOT = "capture24"
"""The top-level directory inside the archive."""

_METADATA_FILE = "metadata.csv"


@unique
class _MetadataColumn(StrEnum):
    """A metadata table column."""

    PID = "pid"
    AGE = "age"
    SEX = "sex"


_DICTIONARY_FILE = "annotation-label-dictionary.csv"
_DICTIONARY_LABEL_COLUMN = "annotation"
# A recording's file name, such as P001.csv.gz. The group is the participant id.
_RECORDING_FILE = re.compile(r"(P[0-9]{3})\.csv\.gz")
_TIME_COLUMN = "time"
_LABEL_COLUMN = "annotation"


class _Acceleration(NamedTuple):
    """A recording's x, y, and z acceleration arrays with one float32 value per sample."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


_AXES = _Acceleration._fields
"""The acceleration column names and signal names."""
_RECORDING_COLUMNS = [_TIME_COLUMN, *_AXES, _LABEL_COLUMN]
_FIRST_DATA_LINE = 2
"""The first data row's line number after the CSV header."""
_RECORD_FILE_KEY = "file"
"""The metadata key for a recording's file name."""

# The paper's Methods: an Axivity AX3 on the dominant wrist, at 100 Hz with a range of ±8 g. The
# authors resampled every recording to exactly 100 Hz before release.
_DEVICE = "Axivity AX3"
_SAMPLE_RATE_HZ = 100
_SAMPLE_PERIOD_US = US_PER_S // _SAMPLE_RATE_HZ
# The source writes eight significant digits. float32 keeps about seven, a step far below the sensor's
# resolution, and halves the stored size. The authors' own loader reads x, y and z as float32 too.
_AXIS_TYPE = pa.float32()
_COLUMN_TYPES = {
    _TIME_COLUMN: pa.timestamp("us"),
    **dict.fromkeys(_AXES, _AXIS_TYPE),
    _LABEL_COLUMN: pa.dictionary(pa.int32(), pa.string()),
}
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float32"
)
_SCHEME_COLUMN_PREFIX = "label:"


@unique
class _Key(StrEnum):
    """An annotation key for CAPTURE-24 records or sources."""

    SUBJECT_IDS = "subject_ids"
    AGE_BAND = "age_band"
    """The participant's age range in years."""
    SEX = "sex"
    RECORDING_START_LOCAL = "recording_start_local"
    """The first sample's published timestamp with a randomized date and shifted time."""
    ACTIVITY = "activity"
    """An activity annotation for consecutive samples with the same raw label."""


# TODO(#624): use `Sex` from `timenet_connectors.demographics` after PR #624 reaches the base branch.
@unique
class _Sex(StrEnum):
    """A participant's recorded sex."""

    FEMALE = "female"
    MALE = "male"


_SEX_OF_LETTER = {"F": _Sex.FEMALE, "M": _Sex.MALE}
"""The sex value for each metadata letter."""


@unique
class _AgeBand(StrEnum):
    """A participant's age range from the metadata table."""

    AGE_18_29 = "18-29"
    AGE_30_37 = "30-37"
    AGE_38_52 = "38-52"
    AGE_53_PLUS = "53+"


# What the authors' prepare_data.py (``--winsec 10``) and benchmark.py fix.
_WINDOW_US = 10 * US_PER_S
_WINDOW_SAMPLES = _WINDOW_US // _SAMPLE_PERIOD_US
_FIRST_TEST_PARTICIPANT = 101
"""The first test participant's number in the P101 to P151 test set."""
_NO_LABEL = -1
"""The code for an empty activity label."""
_AGE_BAND_DESCRIPTION = "The participant's age band in years, as metadata.csv writes it. The authors give no exact age."
_RECORDING_START_DESCRIPTION = (
    "The first sample's timestamp as the authors published it, with no zone. For anonymity they randomized "
    "the date and shifted the time by a small random amount. The time of day is as published, not an "
    "exact local time."
)


# The label sets of the two schemes that benchmark.py scores, with the classes as the label dictionary
# names them. Each set is in string order, because a tie between classes goes to the first in string
# order.
WILLETTS_SPECIFIC_2018 = LabelSet(
    [
        "bicycling",
        "household-chores",
        "manual-work",
        "mixed-activity",
        "sitting",
        "sleep",
        "sports",
        "standing",
        "vehicle",
        "walking",
    ],
    annotations=[Annotation("name", value="willetts_specific_2018")],
)
WALMSLEY_2020 = LabelSet(
    ["light", "moderate-vigorous", "sedentary", "sleep"],
    annotations=[Annotation("name", value="walmsley_2020")],
)


@unique
class _Scheme(StrEnum):
    """A label scheme in dictionary column order."""

    WILLETTS_SPECIFIC_2018 = "WillettsSpecific2018"
    WILLETTS_MET_2018 = "WillettsMET2018"
    DOHERTY_SPECIFIC_2018 = "DohertySpecific2018"
    WILLETTS_2018 = "Willetts2018"
    DOHERTY_2018 = "Doherty2018"
    WALMSLEY_2020 = "Walmsley2020"

    @property
    def column(self) -> str:
        """The dictionary column for this scheme's activity classes."""
        return f"{_SCHEME_COLUMN_PREFIX}{self.value}"


_LABEL_SETS = {
    _Scheme.WILLETTS_SPECIFIC_2018: WILLETTS_SPECIFIC_2018,
    _Scheme.WALMSLEY_2020: WALMSLEY_2020,
}
"""The label sets for the two benchmark schemes."""


@dataclass(frozen=True)
class _LabelDictionary:
    """Raw activity labels and their class mappings."""

    raw_labels: tuple[str, ...]
    """The raw labels in dictionary row order."""
    code_of_raw: dict[str, int]
    """The dictionary row index of each raw label."""
    scheme_classes_of_raw: tuple[dict[_Scheme, str], ...]
    """The class of each raw label in all six schemes."""
    class_of_raw: dict[_Scheme, np.ndarray]
    """The class indices for each benchmark scheme in raw label order.

    The final entry represents unlabeled samples and is selected by ``_NO_LABEL``.
    On a tie in class counts, the lower index wins, as in the authors' ``make_windows``.
    """


@dataclass(frozen=True)
class _Segment:
    """A sequence of samples spaced exactly one sample period apart."""

    first_row: int
    n_values: int
    offset_us: int
    """The segment's start offset in microseconds from the recording's first sample."""


@dataclass(frozen=True)
class _LabelRuns:
    """A segment's consecutive sample runs with the same nonempty raw label."""

    starts_us: np.ndarray
    """Each run's start offset in microseconds from the recording's first sample."""
    ends_us: np.ndarray
    """Each run's exclusive end offset in microseconds from the recording's first sample.

    The end falls one sample period after the last sample.
    """
    codes: np.ndarray
    """The dictionary index of each run's raw label."""


@dataclass(frozen=True)
class _Person:
    """A participant's metadata and recording path."""

    pid: str
    age_band: _AgeBand
    sex: _Sex
    path: Path

    @property
    def split(self) -> Split:
        """The participant's training or test split."""
        return Split.TEST if int(self.pid.removeprefix("P")) >= _FIRST_TEST_PARTICIPANT else Split.TRAIN


@dataclass(frozen=True)
class _Windows:
    """A recording's selected windows and activity classes."""

    first_rows: np.ndarray
    """The first row index of each selected window."""
    classes: dict[_Scheme, np.ndarray]
    """Each selected window's class index in each benchmark scheme's label set."""


@dataclass(frozen=True)
class _Participant:
    """A participant's recording summary without acceleration samples."""

    person: _Person
    n_rows: int
    """The expected recording row count."""
    acceleration_digest: str
    """The expected digest of the x, y, and z acceleration values."""
    start_local: str
    """The first sample's published ISO 8601 timestamp with microseconds and no time zone.

    The authors randomized the date and shifted the time of day.
    """
    segments: tuple[_Segment, ...]
    label_runs: tuple[_LabelRuns, ...]
    """The raw label runs for each segment in segment order."""
    window_starts_us: np.ndarray
    """Each selected window's start offset in microseconds from the recording's first sample."""
    window_classes: dict[_Scheme, np.ndarray]
    """The class index of each selected window in each benchmark scheme's label set."""


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """Read a CSV table with the specified columns in any order.

    Returns:
        A list of rows keyed by column name.

    Raises:
        TimeNetConnectorError: If reading fails or the header or row fields are invalid.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            header = reader.fieldnames or []
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if len(header) != len(columns) or set(header) != set(columns):
        raise TimeNetConnectorError(f"{path}: expected the columns {columns}, got {header}")
    for line, row in enumerate(rows, start=_FIRST_DATA_LINE):
        if None in row:
            raise TimeNetConnectorError(f"{path}:{line}: more fields than the header names")
        for column in columns:
            if not row[column]:
                raise TimeNetConnectorError(f"{path}:{line}: no value for {column!r}")
    return rows


def _read_labels(path: Path) -> _LabelDictionary:
    """Read raw activity labels and their classes in all six schemes.

    Returns:
        The raw labels and class mappings.

    Raises:
        TimeNetConnectorError: If reading fails or columns, fields, raw labels, or benchmark classes are invalid.
    """
    rows = _read_rows(path, [_DICTIONARY_LABEL_COLUMN, *(scheme.column for scheme in _Scheme)])
    raw_labels = [row[_DICTIONARY_LABEL_COLUMN] for row in rows]
    if len(set(raw_labels)) != len(raw_labels):
        raise TimeNetConnectorError(f"{path}: a raw label is listed twice")
    class_of_raw = {scheme: _class_codes(path, [row[scheme.column] for row in rows], scheme) for scheme in _LABEL_SETS}
    return _LabelDictionary(
        raw_labels=tuple(raw_labels),
        code_of_raw={label: code for code, label in enumerate(raw_labels)},
        scheme_classes_of_raw=tuple(_scheme_classes(row) for row in rows),
        class_of_raw=class_of_raw,
    )


def _class_codes(path: Path, classes: list[str], scheme: _Scheme) -> np.ndarray:
    """Map activity classes to indices in the label set of a benchmark scheme.

    Returns:
        Class indices in input order followed by ``len(label_set.labels)`` for unlabeled samples.

    Raises:
        TimeNetConnectorError: If a class is absent from the label set.
    """
    label_set = _LABEL_SETS[scheme]
    unknown = sorted(set(classes) - set(label_set.names))
    if unknown:
        raise TimeNetConnectorError(f"{path}: classes missing from the label set of {scheme}: {unknown}")
    index_of_class = {name: index for index, name in enumerate(label_set.names)}
    codes = [index_of_class[name] for name in classes]
    unlabeled = len(label_set.labels)
    return np.array([*codes, unlabeled], dtype=np.int16)


def _scheme_classes(row: dict[str, str]) -> dict[_Scheme, str]:
    """Return the activity class for each scheme in a dictionary row."""
    return {scheme: row[scheme.column] for scheme in _Scheme}


_Parsed = TypeVar("_Parsed")


def _parse_person_value(path: Path, line: int, column: str, value: str, parse: Callable[[str], _Parsed]) -> _Parsed:
    """Parse a metadata cell with the given parser.

    Returns:
        The parsed cell value.

    Raises:
        TimeNetConnectorError: If the parser rejects the value.
    """
    try:
        return parse(value)
    except (KeyError, ValueError) as exc:
        raise TimeNetConnectorError(f"{path}:{line}: unknown {column} {value!r}") from exc


def _read_people(root: Path) -> list[_Person]:
    """Read participant metadata and match each participant to a recording.

    Returns:
        Participants sorted by ID.

    Raises:
        TimeNetConnectorError: If reading fails or metadata is invalid, empty, or inconsistent with recording files.
    """
    path = root / _METADATA_FILE
    rows = _read_rows(path, [column.value for column in _MetadataColumn])
    if not rows:
        raise TimeNetConnectorError(f"{path}: lists no participants")
    pids = [row[_MetadataColumn.PID] for row in rows]
    if len(set(pids)) != len(pids):
        raise TimeNetConnectorError(f"{path}: a pid is listed twice")
    recordings = _recordings_by_pid(root)
    if set(recordings) != set(pids):
        raise TimeNetConnectorError(
            f"{root}: recordings missing for {sorted(set(pids) - set(recordings))}, "
            f"unlisted recordings {sorted(set(recordings) - set(pids))}"
        )
    people = [_person_of_row(path, line, row, recordings) for line, row in enumerate(rows, start=_FIRST_DATA_LINE)]
    return sorted(people, key=lambda person: person.pid)


def _person_of_row(path: Path, line: int, row: dict[str, str], recordings: dict[str, Path]) -> _Person:
    """Build a participant from a metadata row and its recording path.

    Returns:
        The participant's ID, age band, sex, and recording path.

    Raises:
        TimeNetConnectorError: If the age band or sex is unknown.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _parse_person_value
    pid = row[_MetadataColumn.PID]
    age_band = _parse_person_value(path, line, _MetadataColumn.AGE, row[_MetadataColumn.AGE], _AgeBand)
    sex_letter = row[_MetadataColumn.SEX]
    sex = _parse_person_value(path, line, _MetadataColumn.SEX, sex_letter, _SEX_OF_LETTER.__getitem__)
    return _Person(pid=pid, age_band=age_band, sex=sex, path=recordings[pid])


def _recordings_by_pid(root: Path) -> dict[str, Path]:
    """Find recordings by participant ID in the release directory.

    Returns:
        Recording paths keyed by the participant IDs in their file names.
    """
    recordings = {}
    for file in root.iterdir():
        match = _RECORDING_FILE.fullmatch(file.name)
        if match is not None:
            recordings[match[1]] = file
    return recordings


def _read_participant(person: _Person, labels: _LabelDictionary) -> _Participant:
    """Summarize a recording's segments, labels, windows, and acceleration digest.

    An empty annotation field marks an unlabeled sample.
    All other annotation values must appear in the label dictionary.

    Returns:
        The participant's recording summary without acceleration samples.

    Raises:
        TimeNetConnectorError: If reading fails or the header, timestamps, labels, windows, or acceleration are invalid.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
    path = person.path
    table = _read_table(path)
    stamps = _read_stamps(path, table)
    offsets = stamps - stamps[0]
    raw_codes = _raw_codes(path, table[_LABEL_COLUMN].combine_chunks(), labels)
    segments = _make_segments(stamps)
    windows = _make_windows(path, offsets, raw_codes, segments, labels)
    return _Participant(
        person=person,
        n_rows=len(stamps),
        acceleration_digest=_digest(_axis_values(path, table)),
        start_local=table[_TIME_COLUMN][0].as_py().isoformat(timespec="microseconds"),
        segments=segments,
        label_runs=tuple(_label_runs(raw_codes, segment) for segment in segments),
        window_starts_us=offsets[windows.first_rows],
        window_classes=windows.classes,
    )


def _convert_options(columns: list[str]) -> pa_csv.ConvertOptions:
    """Configure CSV parsing for the selected recording columns.

    Only empty fields count as missing values.

    Returns:
        Parsing options with the recording's column types.
    """
    return pa_csv.ConvertOptions(
        include_columns=columns,
        column_types={column: _COLUMN_TYPES[column] for column in columns},
        strings_can_be_null=True,
        null_values=[""],
    )


def _read_table(path: Path) -> pa.Table:
    """Read all recording columns into an Arrow table.

    Returns:
        The recording table with dictionary-encoded activity labels.

    Raises:
        TimeNetConnectorError: If reading fails, the header is invalid, or the recording is empty.
    """
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if header != _RECORDING_COLUMNS:
        raise TimeNetConnectorError(f"{path}: expected the header {','.join(_RECORDING_COLUMNS)}, got {header}")
    try:
        table = pa_csv.read_csv(path, convert_options=_convert_options(_RECORDING_COLUMNS))
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if table.num_rows == 0:
        raise TimeNetConnectorError(f"{path}: holds no samples")
    return table


def _read_stamps(path: Path, table: pa.Table) -> np.ndarray:
    """Read sample timestamps in microseconds.

    Returns:
        Sample timestamps in recording order.

    Raises:
        TimeNetConnectorError: If a timestamp is missing or a time step is shorter than one sample period.
    """
    times = table[_TIME_COLUMN].combine_chunks()
    if times.null_count:
        raise TimeNetConnectorError(f"{path}: {times.null_count} samples have no timestamp")
    stamps = times.cast(pa.int64()).to_numpy()
    steps = np.diff(stamps)
    # Reject any step shorter than one period: it is neither a regular step nor a gap.
    if (steps < _SAMPLE_PERIOD_US).any():
        row = int(np.flatnonzero(steps < _SAMPLE_PERIOD_US)[0]) + 1
        raise TimeNetConnectorError(
            f"{path}: the timestamp of row {row} doesn't follow the one before it by a full period"
        )
    return stamps


def _make_segments(stamps: np.ndarray) -> tuple[_Segment, ...]:
    """Split samples at timestamp steps that differ from one sample period.

    Returns:
        Sample segments in time order.
    """
    bounds = [0, *(np.flatnonzero(np.diff(stamps) != _SAMPLE_PERIOD_US) + 1).tolist(), len(stamps)]
    return tuple(
        _Segment(first_row=start, n_values=end - start, offset_us=int(stamps[start] - stamps[0]))
        for start, end in pairwise(bounds)
    )


def _label_runs(raw_codes: np.ndarray, segment: _Segment) -> _LabelRuns:
    """Find consecutive samples with the same nonempty raw label within a segment.

    Returns:
        Label runs in time order.
    """
    codes = raw_codes[segment.first_row : segment.first_row + segment.n_values]
    changes = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(codes)]))
    labeled = codes[starts] != _NO_LABEL
    starts, ends = starts[labeled], ends[labeled]
    return _LabelRuns(
        starts_us=segment.offset_us + starts * _SAMPLE_PERIOD_US,
        ends_us=segment.offset_us + ends * _SAMPLE_PERIOD_US,
        codes=codes[starts],
    )


def _axis_values(path: Path, table: pa.Table) -> _Acceleration:
    """Read the x, y, and z acceleration columns from a recording table.

    Returns:
        One float32 array per acceleration axis.

    Raises:
        TimeNetConnectorError: If an acceleration value is missing or nonfinite.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _finite_column
    return _Acceleration._make(_finite_column(path, table, column) for column in _AXES)


def _finite_column(path: Path, table: pa.Table, column: str) -> np.ndarray:
    """Read an acceleration column from a recording table.

    Returns:
        The column values as a float32 array.

    Raises:
        TimeNetConnectorError: If a column value is missing or nonfinite.
    """
    array = table[column].combine_chunks()
    if array.null_count:
        raise TimeNetConnectorError(f"{path}: {array.null_count} samples have no {column} value")
    values = array.to_numpy()
    if not np.isfinite(values).all():
        raise TimeNetConnectorError(f"{path}: a value of {column} is not finite")
    return values


def _digest(values: _Acceleration) -> str:
    """Compute a digest of acceleration values in x, y, and z order.

    Returns:
        The hexadecimal BLAKE2b digest.
    """
    digest = hashlib.blake2b()
    for axis in values:
        digest.update(axis)
    return digest.hexdigest()


def _raw_codes(path: Path, column: pa.DictionaryArray, labels: _LabelDictionary) -> np.ndarray:
    """Map sample labels to raw label indices.

    Returns:
        Raw label indices with ``_NO_LABEL`` for unlabeled samples.

    Raises:
        TimeNetConnectorError: If a raw label is absent from the dictionary.
    """
    written = column.dictionary.to_pylist()
    unknown = sorted(label for label in written if label not in labels.code_of_raw)
    if unknown:
        raise TimeNetConnectorError(f"{path}: raw labels missing from the label dictionary: {unknown}")
    # Replace each null index with the index of an extra entry whose value is _NO_LABEL.
    codes = np.array([labels.code_of_raw[label] for label in written] + [_NO_LABEL], dtype=np.int16)
    return codes[column.indices.fill_null(len(written)).to_numpy()]


def _make_windows(
    path: Path, offsets_us: np.ndarray, raw_codes: np.ndarray, segments: tuple[_Segment, ...], labels: _LabelDictionary
) -> _Windows:
    """Select 10-second windows and assign their activity classes.

    Bins start at the first sample and require 1000 samples with at least one label.
    Each scheme uses the most common labeled class and breaks ties by class name order.

    Returns:
        The selected windows and their class indices for each benchmark scheme.

    Raises:
        TimeNetConnectorError: If a selected window crosses a gap.
    """
    # Each sample's 10 s bin on the first sample's grid, and the first row of each bin that has samples.
    bins = offsets_us // _WINDOW_US
    bin_starts = np.flatnonzero(np.diff(bins, prepend=bins[0] - 1))
    # Keep the bins that hold exactly 1000 samples: the rows up to the next bin's first row.
    full = bin_starts[np.diff(bin_starts, append=len(bins)) == _WINDOW_SAMPLES]
    # One row of 1000 raw label codes per full bin. Keep the bins with at least one labeled sample.
    window_codes = raw_codes[full[:, None] + np.arange(_WINDOW_SAMPLES)]
    labeled = (window_codes != _NO_LABEL).any(axis=1)
    first_rows, window_codes = full[labeled], window_codes[labeled]
    # A window crosses a gap if its first and last rows are in different segments. searchsorted finds
    # each row's segment from the segments' first rows, with no array as long as the recording.
    segment_first_rows = np.array([segment.first_row for segment in segments])
    first_segment = np.searchsorted(segment_first_rows, first_rows, side="right")
    last_segment = np.searchsorted(segment_first_rows, first_rows + _WINDOW_SAMPLES - 1, side="right")
    crossing = first_segment != last_segment
    if crossing.any():
        row = int(first_rows[crossing][0])
        raise TimeNetConnectorError(f"{path}: the window at row {row} holds {_WINDOW_SAMPLES} samples across a gap")
    classes = {}
    for scheme, class_of_raw in labels.class_of_raw.items():
        per_sample = class_of_raw[window_codes]
        # Count each class per window. Unlabeled samples use an index beyond the classes, so the counts exclude them.
        counts = np.stack(
            [(per_sample == index).sum(axis=1) for index in range(len(_LABEL_SETS[scheme].labels))], axis=1
        )
        # argmax takes the first of equal counts: the class first in string order.
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return _Windows(first_rows=first_rows, classes=classes)


def _read_acceleration(participant: _Participant) -> _Acceleration:
    """Read and validate a recording's acceleration against its saved summary.

    Returns:
        One float32 array per acceleration axis.

    Raises:
        TimeNetConnectorError: If reading fails, a value is missing or nonfinite, or the row count or values changed.
    """
    path = participant.person.path
    try:
        table = pa_csv.read_csv(path, convert_options=_convert_options(list(_AXES)))
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if table.num_rows != participant.n_rows:
        raise TimeNetConnectorError(
            f"{path}: holds {table.num_rows} rows, {participant.n_rows} when it was read; the file changed"
        )
    values = _axis_values(path, table)
    if _digest(values) != participant.acceleration_digest:
        raise TimeNetConnectorError(f"{path}: the acceleration values differ from when it was read; the file changed")
    return values


_ReadAcceleration = Callable[[_Participant], _Acceleration]


class _OneRecording:
    """Cache one recording's acceleration values at a time.

    The cache releases the previous recording before reading a different one.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._values: _Acceleration | None = None

    def __call__(self, participant: _Participant) -> _Acceleration:
        """Return cached acceleration values or read the requested recording.

        Returns:
            One float32 array per acceleration axis.
        """
        if self._values is None or self._path != participant.person.path:
            # Release the previous recording before reading the next.
            self._path, self._values = None, None
            self._values = _read_acceleration(participant)
            self._path = participant.person.path
        return self._values


def _load_segment(acceleration: _ReadAcceleration, participant: _Participant, segment: _Segment, axis: int) -> pa.Array:
    """Extract one segment's acceleration values for the given axis index.

    Returns:
        The segment's float32 values as an Arrow array.
    """
    values = acceleration(participant)[axis]
    return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=_AXIS_TYPE)


def _activity_annotations(runs: _LabelRuns, labels: _LabelDictionary) -> Iterator[Annotation]:
    """Convert raw label runs to activity annotations.

    Yields:
        An annotation with a raw label, its time span, and its class in each of the six schemes as metadata.
    """
    for start_us, end_us, code in zip(runs.starts_us.tolist(), runs.ends_us.tolist(), runs.codes.tolist(), strict=True):
        yield Annotation(
            _Key.ACTIVITY,
            value=labels.raw_labels[code],
            span=TimeInterval.micros(start_us, end_us),
            metadata={scheme.value: name for scheme, name in labels.scheme_classes_of_raw[code].items()},
        )


def _record(participant: _Participant, labels: _LabelDictionary, acceleration: _ReadAcceleration) -> Record:
    """Build a participant record with one source per segment.

    Each source has x, y, and z signals and raw activity annotations.
    Acceleration values load from the recording file on demand.

    Returns:
        The participant record with signals and annotations.
    """
    sources = []
    for index, (segment, runs) in enumerate(zip(participant.segments, participant.label_runs, strict=True)):
        time_axis = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = [
            Signal.from_loader(
                name=column,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(_load_segment, acceleration, participant, segment, axis),
            )
            for axis, column in enumerate(_AXES)
        ]
        # Pass all runs to the Source constructor at once. Adding one run at a time copies the
        # annotation tuple for every run.
        sources.append(
            Source(
                f"{_DEVICE} segment {index}",
                signals=signals,
                annotations=list(_activity_annotations(runs, labels)),
            )
        )
    person = participant.person
    return Record(
        sources=sources,
        metadata={_RECORD_FILE_KEY: person.path.name},
        annotations=[
            Annotation(_Key.SUBJECT_IDS, value=[person.pid]),
            Annotation(_Key.AGE_BAND, value=person.age_band.value, description=_AGE_BAND_DESCRIPTION),
            Annotation(_Key.SEX, value=person.sex.value),
            Annotation(
                _Key.RECORDING_START_LOCAL,
                value=participant.start_local,
                description=_RECORDING_START_DESCRIPTION,
            ),
        ],
    )


def _scheme_tasks(scheme: _Scheme, participant: _Participant, record: Record) -> Iterator[ClassificationTask]:
    """Yield classification tasks for a participant's selected windows in one scheme.

    Yields:
        A task with the window's time span and class in time order.
    """
    label_set = _LABEL_SETS[scheme]
    starts_us = participant.window_starts_us.tolist()
    classes = participant.window_classes[scheme].tolist()
    for start_us, class_index in zip(starts_us, classes, strict=True):
        yield ClassificationTask(
            inputs=[record],
            scope=TimeInterval.micros(start_us, start_us + _WINDOW_US),
            targets=label_set.names[class_index],
            label_set=label_set,
            split=participant.person.split,
        )


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from its pinned Oxford Research Archive zip file."""

    async def download_async(self, cache_dir: Path) -> Path:  # ruff: ignore[no-self-use] - the connector interface
        """Download and extract the pinned CAPTURE-24 archive.

        The archive must match ``ARCHIVE_SHA256`` before extraction.
        Later calls reuse the extracted directory.

        Returns:
            The extracted ``capture24`` directory.
        """
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=ARCHIVE_SHA256)
        return find_dir_containing(root, f"{_ARCHIVE_ROOT}/{_METADATA_FILE}")

    def convert(self, root: Path, /) -> TimeFDataset:
        """Convert CAPTURE-24 recordings to records and window classification tasks.

        Acceleration values load from the files on demand.
        Keep the files available and unchanged until all signals have loaded.

        Args:
            root: The extracted ``capture24`` directory.

        Returns:
            A dataset with participant records and streamed tasks for both benchmark schemes.

        Raises:
            TimeFValidationError: If ``root`` is not a directory.
            TimeNetConnectorError: If a metadata table or recording is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
        if not root.is_dir():
            raise TimeFValidationError(f"CAPTURE-24 loads from one directory, got {root}")
        labels = _read_labels(root / _DICTIONARY_FILE)
        # One participant's samples are held at a time: the first pass keeps only small results.
        participants = [_read_participant(person, labels) for person in _read_people(root)]
        acceleration = _OneRecording()
        records = [_record(participant, labels, acceleration) for participant in participants]

        dataset = TimeFDataset(metadata=self.metadata())
        for record in records:
            dataset.add_record(record)

        def tasks() -> Iterator[ClassificationTask]:
            # As the authors' benchmark.py: one benchmark per scheme, over every participant's windows.
            for scheme in _LABEL_SETS:
                for participant, record in zip(participants, records, strict=True):
                    yield from _scheme_tasks(scheme, participant, record)

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = Capture24Connector
