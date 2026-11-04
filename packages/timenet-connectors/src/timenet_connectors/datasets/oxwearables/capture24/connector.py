"""Convert the pinned CAPTURE-24 zip into lazy Records and one classification task per window and scheme.

CAPTURE-24 holds about a day of 100 Hz wrist acceleration from each of 151 participants, labeled from
body-camera images and sleep diaries. A participant becomes one Record with one Source per gap-free
segment, each with lazy ``x``, ``y`` and ``z`` signals and one ``activity`` annotation per run of a raw
label. Each 10 s window that the authors' benchmark code keeps becomes one task per label scheme the
benchmark scores. README.md explains how this follows the authors.

The build has two passes. The first reads every recording, one at a time, and keeps only small results:
the segments, each window's start and class, and a digest of the acceleration values. It fails on a
missing or non-finite value. The second builds the Records and streams the tasks. The acceleration values
stay in the files until they are written, and then match the digest or fail the build.
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
    Split,
    TimeInterval,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The one zip of the Oxford Research Archive record, 6.9 GB."""
_ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ARCHIVE_ROOT = "capture24"
"""The archive's one top-level directory, which holds every file below."""

_METADATA_FILE = "metadata.csv"


@unique
class _MetadataColumn(StrEnum):
    """A column of ``metadata.csv``."""

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
    """A recording's x, y and z values: one float32 array each, with one value per row."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


_AXES = _Acceleration._fields
"""The acceleration columns of a recording, which are also its signal names."""
_RECORDING_COLUMNS = [_TIME_COLUMN, *_AXES, _LABEL_COLUMN]
_FIRST_DATA_LINE = 2
"""The file line of a table's first row, after the header line."""
_RECORD_FILE_KEY = "file"
"""The Record metadata key that holds the recording's file name."""

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
    """The key of an annotation that this connector writes."""

    SUBJECT_IDS = "subject_ids"
    AGE_BAND = "age_band"
    """The participant's age band. Other connectors use ``age`` for an exact age in years."""
    SEX = "sex"
    RECORDING_START_LOCAL = "recording_start_local"
    """The first sample's published timestamp. The authors randomized its date and shifted its time."""
    ACTIVITY = "activity"
    """A Source annotation that holds one run of a raw label."""


@unique
class _Sex(StrEnum):
    """A participant's sex, as the connector stores it.

    The values match ``Sex`` in ``timenet_connectors.demographics``, which PR #624 adds. Once it is on
    the base, the connector uses that enum instead.
    """

    FEMALE = "female"
    MALE = "male"


_SEX_OF_LETTER = {"F": _Sex.FEMALE, "M": _Sex.MALE}
"""The letter that ``metadata.csv`` writes for each sex."""


@unique
class _AgeBand(StrEnum):
    """A participant's age band, as ``metadata.csv`` writes it. The authors give no exact age."""

    AGE_18_29 = "18-29"
    AGE_30_37 = "30-37"
    AGE_38_52 = "38-52"
    AGE_53_PLUS = "53+"


# What the authors' prepare_data.py (``--winsec 10``) and benchmark.py fix.
_WINDOW_US = 10 * US_PER_S
_WINDOW_SAMPLES = _WINDOW_US // _SAMPLE_PERIOD_US
_FIRST_TEST_PARTICIPANT = 101
"""The paper's test set is P101 to P151, and benchmark.py tests ``range(101, 152)``."""
_NO_LABEL = -1
"""The code of an unlabeled sample, which the source writes as an empty field."""
_AGE_BAND_DESCRIPTION = "The participant's age band in years, as metadata.csv writes it. The authors give no exact age."
_RECORDING_START_DESCRIPTION = (
    "The first sample's timestamp as the authors published it, with no zone. For anonymity they randomized "
    "the date and shifted the time by a small random amount. The time of day is as published, not an "
    "exact local time."
)


@dataclass(frozen=True)
class _Vocabulary:
    """The closed list of a benchmarked scheme's classes, and the dataset annotation that stores it."""

    name: str
    """The key of the dataset annotation that holds the classes, which each task's target_schema names."""
    labels: tuple[str, ...]
    """The scheme's classes, in string order."""


# The classes of the two schemes that benchmark.py scores, as the label dictionary names them. Each
# list is in string order, because a tie between classes goes to the first in string order.
WILLETTS_SPECIFIC_2018_VOCABULARY = _Vocabulary(
    name="willetts_specific_2018_vocabulary",
    labels=(
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
    ),
)
WALMSLEY_2020_VOCABULARY = _Vocabulary(
    name="walmsley_2020_vocabulary", labels=("light", "moderate-vigorous", "sedentary", "sleep")
)


@unique
class _Scheme(StrEnum):
    """A label scheme of the label dictionary, in the file's column order."""

    WILLETTS_SPECIFIC_2018 = "WillettsSpecific2018"
    WILLETTS_MET_2018 = "WillettsMET2018"
    DOHERTY_SPECIFIC_2018 = "DohertySpecific2018"
    WILLETTS_2018 = "Willetts2018"
    DOHERTY_2018 = "Doherty2018"
    WALMSLEY_2020 = "Walmsley2020"

    @property
    def column(self) -> str:
        """The dictionary column that holds each raw label's class in this scheme."""
        return f"{_SCHEME_COLUMN_PREFIX}{self.value}"


_VOCABULARIES = {
    _Scheme.WILLETTS_SPECIFIC_2018: WILLETTS_SPECIFIC_2018_VOCABULARY,
    _Scheme.WALMSLEY_2020: WALMSLEY_2020_VOCABULARY,
}
"""The vocabulary of each scheme that the authors' benchmark.py scores. Only these schemes have tasks."""


@dataclass(frozen=True)
class _LabelDictionary:
    """The label dictionary, as the windowing and the activity annotations need it."""

    raw_labels: tuple[str, ...]
    """Each raw label, in file order."""
    code_of_raw: dict[str, int]
    """Each raw label's index, in file order."""
    scheme_classes_of_raw: tuple[dict[_Scheme, str], ...]
    """Each raw label's class in all six schemes, as the dictionary names it."""
    class_of_raw: dict[_Scheme, np.ndarray]
    """For each scheme, this array maps raw labels to class indices in the vocabulary.

    The final entry represents an unlabeled sample. Indexing with ``_NO_LABEL`` selects that entry.
    A class's index breaks ties. ``Series.mode`` sorts the modes, and the authors' ``make_windows`` takes
    the first."""


@dataclass(frozen=True)
class _Segment:
    """A run of samples exactly one period apart, between gaps."""

    first_row: int
    n_values: int
    offset_us: int
    """The first sample's time from the recording's first sample."""


@dataclass(frozen=True)
class _LabelRuns:
    """A segment's runs of one raw label: consecutive samples with the same non-empty label."""

    starts_us: np.ndarray
    """Each run's first sample's time from the recording's first sample."""
    ends_us: np.ndarray
    """The time one period after each run's last sample, so that a run is ``[start, end)``."""
    codes: np.ndarray
    """Each run's raw label, as an index into the dictionary."""


@dataclass(frozen=True)
class _Person:
    """A participant listed in ``metadata.csv``, with the recording that holds the samples."""

    pid: str
    age_band: _AgeBand
    sex: _Sex
    path: Path

    @property
    def split(self) -> Split:
        """The participant's split in the authors' benchmark.py."""
        return Split.TEST if int(self.pid.removeprefix("P")) >= _FIRST_TEST_PARTICIPANT else Split.TRAIN


@dataclass(frozen=True)
class _Windows:
    """The windows a recording keeps."""

    first_rows: np.ndarray
    """Each kept window's first row."""
    classes: dict[_Scheme, np.ndarray]
    """Each kept window's class per benchmarked scheme, as an index into that scheme's vocabulary."""


@dataclass(frozen=True)
class _Participant:
    """One participant after the first pass: what the Record and the tasks need, not the samples."""

    person: _Person
    n_rows: int
    """The rows the recording held when it was read, which a reread must match."""
    acceleration_digest: str
    """The digest of the recording's x, y and z values when it was read, which a reread must match."""
    start_local: str
    """The first sample's published timestamp in ISO 8601 with microseconds and no zone. The date is
    randomized, and the authors shifted the time of day by a small random amount."""
    segments: tuple[_Segment, ...]
    label_runs: tuple[_LabelRuns, ...]
    """Each segment's runs of a raw label, in segment order."""
    window_starts_us: np.ndarray
    """Each kept window's first sample's time from the recording's first sample."""
    window_classes: dict[_Scheme, np.ndarray]
    """Each kept window's class per scheme, as an index into that scheme's vocabulary."""


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """Read a small CSV table whose columns must be exactly ``columns``, in any order.

    A short row reads as None and a blank field as "", and neither is a value. ``csv.DictReader`` puts the
    fields of a long row under the key None.

    Returns:
        One dict per row, keyed by column.

    Raises:
        TimeNetConnectorError: If the file is unreadable or its columns differ from the expected columns.
            Also raised if a field is missing or a row has more fields than the header.
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
    """Read the label dictionary: each raw label and its class in each scheme.

    Returns:
        The dictionary, as the windowing and the activity annotations need it.

    Raises:
        TimeNetConnectorError: If the file is unreadable or its columns are not the raw label and the six schemes.
            Also raised if a row has missing or extra fields, or a raw label appears twice.
            Every class must belong to its scheme's vocabulary.
    """
    rows = _read_rows(path, [_DICTIONARY_LABEL_COLUMN, *(scheme.column for scheme in _Scheme)])
    raw_labels = [row[_DICTIONARY_LABEL_COLUMN] for row in rows]
    if len(set(raw_labels)) != len(raw_labels):
        raise TimeNetConnectorError(f"{path}: a raw label is listed twice")
    class_of_raw = {
        scheme: _class_codes(path, [row[scheme.column] for row in rows], vocabulary)
        for scheme, vocabulary in _VOCABULARIES.items()
    }
    return _LabelDictionary(
        raw_labels=tuple(raw_labels),
        code_of_raw={label: code for code, label in enumerate(raw_labels)},
        scheme_classes_of_raw=tuple(_scheme_classes(row) for row in rows),
        class_of_raw=class_of_raw,
    )


def _class_codes(path: Path, classes: list[str], vocabulary: _Vocabulary) -> np.ndarray:
    """Map each raw label's class in one scheme to its index in the scheme's vocabulary.

    Returns:
        One index per raw label, in file order, then one more entry for an unlabeled sample. That
        entry is one past the vocabulary's last class, so that indexing with ``_NO_LABEL`` finds it
        and no class count takes it.

    Raises:
        TimeNetConnectorError: If a class is not in the vocabulary.
    """
    unknown = sorted(set(classes) - set(vocabulary.labels))
    if unknown:
        raise TimeNetConnectorError(f"{path}: classes missing from vocabulary {vocabulary.name!r}: {unknown}")
    index_of_class = {name: index for index, name in enumerate(vocabulary.labels)}
    codes = [index_of_class[name] for name in classes]
    unlabeled = len(vocabulary.labels)
    return np.array([*codes, unlabeled], dtype=np.int16)


def _scheme_classes(row: dict[str, str]) -> dict[_Scheme, str]:
    """Return a dictionary row's class in each of the six schemes."""
    return {scheme: row[scheme.column] for scheme in _Scheme}


_Parsed = TypeVar("_Parsed")


def _parse_person_value(path: Path, line: int, column: str, value: str, parse: Callable[[str], _Parsed]) -> _Parsed:
    """Parse one cell of ``metadata.csv``.

    Returns:
        The parsed value.

    Raises:
        TimeNetConnectorError: If the value is not one that the release writes.
    """
    try:
        return parse(value)
    except (KeyError, ValueError) as exc:
        raise TimeNetConnectorError(f"{path}:{line}: unknown {column} {value!r}") from exc


def _read_people(root: Path) -> list[_Person]:
    """Read ``metadata.csv`` and pair each row with its recording.

    Returns:
        One person per participant, in id order.

    Raises:
        TimeNetConnectorError: If the file is unreadable, its columns are not pid, age and sex, or it lists
            no participants. Also raised if a row has missing or extra fields, an age band or sex is unknown,
            or a pid repeats.
            The table and the recordings must name the same participants.
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
    people = [
        _Person(
            pid=row[_MetadataColumn.PID],
            age_band=_parse_person_value(path, line, _MetadataColumn.AGE, row[_MetadataColumn.AGE], _AgeBand),
            sex=_parse_person_value(
                path, line, _MetadataColumn.SEX, row[_MetadataColumn.SEX], _SEX_OF_LETTER.__getitem__
            ),
            path=recordings[row[_MetadataColumn.PID]],
        )
        for line, row in enumerate(rows, start=_FIRST_DATA_LINE)
    ]
    return sorted(people, key=lambda person: person.pid)


def _recordings_by_pid(root: Path) -> dict[str, Path]:
    """Find the recording files in the release directory.

    Returns:
        Each recording's path, keyed by the participant id in its file name.
    """
    recordings = {}
    for file in root.iterdir():
        match = _RECORDING_FILE.fullmatch(file.name)
        if match is not None:
            recordings[match[1]] = file
    return recordings


def _read_participant(person: _Person, labels: _LabelDictionary) -> _Participant:
    """Read a recording: its segments, its raw label runs, its kept windows, and a digest of its acceleration values.

    An empty ``annotation`` field is an unlabeled sample. No other value is.

    Returns:
        The participant, without the samples.

    Raises:
        TimeNetConnectorError: If the file is unreadable, its header is wrong, or it has no rows.
            Also raised if a timestamp is missing or unreadable, or a step is shorter than one period.
            Raw labels must belong to the dictionary, and kept windows must not cross gaps.
            Acceleration values must be present and finite.
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
    """Return the options that read ``columns`` of a recording, so that both passes parse them alike.

    Only an empty field is missing, so a written ``nan`` reaches the finite check in ``_axis_values``.
    """
    return pa_csv.ConvertOptions(
        include_columns=columns,
        column_types={column: _COLUMN_TYPES[column] for column in columns},
        strings_can_be_null=True,
        null_values=[""],
    )


def _read_table(path: Path) -> pa.Table:
    """Read every column of a recording. The ``annotation`` column is dictionary-encoded.

    Returns:
        The recording's table.

    Raises:
        TimeNetConnectorError: If the file cannot be read, the header is wrong, or the file has no rows.
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
    """Return each sample's time in microseconds.

    Raises:
        TimeNetConnectorError: If a timestamp is missing, or a step is shorter than one period.
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
    """Split the samples at each gap, which is any step but one period.

    Returns:
        The segments, in time order.
    """
    bounds = [0, *(np.flatnonzero(np.diff(stamps) != _SAMPLE_PERIOD_US) + 1).tolist(), len(stamps)]
    return tuple(
        _Segment(first_row=start, n_values=end - start, offset_us=int(stamps[start] - stamps[0]))
        for start, end in pairwise(bounds)
    )


def _label_runs(raw_codes: np.ndarray, segment: _Segment) -> _LabelRuns:
    """Find a segment's runs of one raw label. Unlabeled samples belong to no run.

    A run never crosses a gap, because each segment is cut on its own. Within a segment every step is
    one period, so a row's time is its index times the period after the segment's first sample.

    Returns:
        The segment's runs, in time order.
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
    """Return the x, y and z columns of a recording's table.

    Returns:
        The float32 values of each axis.

    Raises:
        TimeNetConnectorError: If a value is missing or not finite.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _finite_column
    return _Acceleration._make(_finite_column(path, table, column) for column in _AXES)


def _finite_column(path: Path, table: pa.Table, column: str) -> np.ndarray:
    """Return one acceleration column of a recording's table.

    Returns:
        The column's float32 values.

    Raises:
        TimeNetConnectorError: If a value is missing or not finite.
    """
    array = table[column].combine_chunks()
    if array.null_count:
        raise TimeNetConnectorError(f"{path}: {array.null_count} samples have no {column} value")
    values = array.to_numpy()
    if not np.isfinite(values).all():
        raise TimeNetConnectorError(f"{path}: a {column} value is not finite")
    return values


def _digest(values: _Acceleration) -> str:
    """Return a digest of the x, y and z values, in that order, to spot a file that changed."""
    digest = hashlib.blake2b()
    for axis in values:
        digest.update(axis)
    return digest.hexdigest()


def _raw_codes(path: Path, column: pa.DictionaryArray, labels: _LabelDictionary) -> np.ndarray:
    """Return each sample's raw label as an index into the dictionary, or ``_NO_LABEL``.

    Raises:
        TimeNetConnectorError: If the recording writes a raw label the dictionary does not list.
    """
    written = column.dictionary.to_pylist()
    unknown = sorted(label for label in written if label not in labels.code_of_raw)
    if unknown:
        raise TimeNetConnectorError(f"{path}: raw labels missing from the label dictionary: {unknown}")
    # One extra slot maps a null's filled index to _NO_LABEL.
    codes = np.array([labels.code_of_raw[label] for label in written] + [_NO_LABEL], dtype=np.int16)
    return codes[column.indices.fill_null(len(written)).to_numpy()]


def _make_windows(
    path: Path, offsets_us: np.ndarray, raw_codes: np.ndarray, segments: tuple[_Segment, ...], labels: _LabelDictionary
) -> _Windows:
    """Return the windows the authors' ``prepare_data.make_windows`` keeps, with each one's classes.

    ``make_windows`` groups the samples into 10 s bins anchored at the first sample. After a gap the bins
    stay on the first segment's grid. It keeps a bin with exactly 1000 samples and at least one labeled
    sample. A bin that a gap cuts usually has fewer samples, but a step between one and two periods is a gap
    too, and a bin can still hold 1000 samples across it. This function fails on such a window, so no kept
    window crosses a gap.

    A window's class in a scheme is the most common class among its labeled samples. A tie goes to the
    first class in string order.

    ``make_windows`` also drops a window with a NaN acceleration value. This function never meets one,
    because the first pass fails on any value that is not finite.

    Returns:
        The kept windows.

    Raises:
        TimeNetConnectorError: If a kept window holds 1000 samples across a gap.
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
        # Count each class per window. An unlabeled sample has a class index past the last, so no count takes it.
        counts = np.stack(
            [(per_sample == index).sum(axis=1) for index in range(len(_VOCABULARIES[scheme].labels))], axis=1
        )
        # argmax takes the first of equal counts: the class first in string order.
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return _Windows(first_rows=first_rows, classes=classes)


def _read_acceleration(participant: _Participant) -> _Acceleration:
    """Read a recording's x, y and z columns as float32, and check that the file is the one the first pass read.

    Returns:
        The float32 values of each axis.

    Raises:
        TimeNetConnectorError: If a value is missing, unreadable or not finite, or the row count or the values
            changed.
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
    """Read a recording's acceleration once for all its signals, and hold only the latest recording.

    A Record's signals load one after another, so one recording suffices. lru_cache(maxsize=1) would
    read the next recording before it dropped the last, holding two at once.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._values: _Acceleration | None = None

    def __call__(self, participant: _Participant) -> _Acceleration:
        """Return the recording's x, y and z, reading it unless it was the last one read.

        Returns:
            The float32 values of each axis.
        """
        if self._values is None or self._path != participant.person.path:
            # Release the previous recording before reading the next.
            self._path, self._values = None, None
            self._values = _read_acceleration(participant)
            self._path = participant.person.path
        return self._values


def _load_segment(
    acceleration: _ReadAcceleration, participant: _Participant, segment: _Segment, column: str
) -> pa.Array:
    """Return one segment's values of one acceleration column.

    Returns:
        The segment's float32 values.
    """
    values = getattr(acceleration(participant), column)
    return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=_AXIS_TYPE)


def _activity_annotations(runs: _LabelRuns, labels: _LabelDictionary) -> Iterator[Annotation]:
    """Turn a segment's raw label runs into one annotation per run.

    Yields:
        An annotation whose value is the raw label, whose span is the run, and whose metadata gives the
        raw label's class in each of the six schemes, keyed by scheme name.
    """
    for start_us, end_us, code in zip(runs.starts_us.tolist(), runs.ends_us.tolist(), runs.codes.tolist(), strict=True):
        yield Annotation(
            key=_Key.ACTIVITY,
            value=labels.raw_labels[code],
            span=TimeInterval.micros(start_us, end_us),
            metadata={scheme.value: name for scheme, name in labels.scheme_classes_of_raw[code].items()},
        )


def _record(participant: _Participant, labels: _LabelDictionary, acceleration: _ReadAcceleration) -> Record:
    """Build a participant's Record: one Source of lazy x, y and z signals per segment.

    Each Source also carries the segment's raw label runs as ``activity`` annotations.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    sources = []
    for index, (segment, runs) in enumerate(zip(participant.segments, participant.label_runs, strict=True)):
        time_axis = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = tuple(
            Signal.from_loader(
                name=column,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(_load_segment, acceleration, participant, segment, column),
            )
            for column in _AXES
        )
        # One constructor call checks every span, with no copy of the annotations per run.
        sources.append(
            Source(
                name=f"{_DEVICE} segment {index}",
                signals=signals,
                annotations=tuple(_activity_annotations(runs, labels)),
            )
        )
    person = participant.person
    record = Record(sources=tuple(sources), metadata={_RECORD_FILE_KEY: person.path.name})
    record.add_annotations(
        (
            Annotation(key=_Key.SUBJECT_IDS, value=[person.pid]),
            Annotation(key=_Key.AGE_BAND, value=person.age_band.value, description=_AGE_BAND_DESCRIPTION),
            Annotation(key=_Key.SEX, value=person.sex.value),
            Annotation(
                key=_Key.RECORDING_START_LOCAL,
                value=participant.start_local,
                description=_RECORDING_START_DESCRIPTION,
            ),
        )
    )
    return record


def _scheme_tasks(scheme: _Scheme, participant: _Participant, record: Record) -> Iterator[ClassificationTask]:
    """Yield one participant's tasks in one scheme: one per kept window, in time order.

    Yields:
        A task whose scope is the window and whose target is the window's class in the scheme.
    """
    vocabulary = _VOCABULARIES[scheme]
    starts_us = participant.window_starts_us.tolist()
    classes = participant.window_classes[scheme].tolist()
    for start_us, class_index in zip(starts_us, classes, strict=True):
        yield ClassificationTask(
            inputs=(record,),
            scope=TimeInterval.micros(start_us, start_us + _WINDOW_US),
            targets=(vocabulary.labels[class_index],),
            target_schema=vocabulary.name,
            split=participant.person.split,
        )


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from the pinned zip of its Oxford Research Archive record."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use] - the connector interface
        """Fetch the pinned archive, check its digest, and extract it once.

        Returns:
            The extracted ``capture24`` directory.
        """
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=ARCHIVE_SHA256)
        return [find_dir_containing(root, f"{_ARCHIVE_ROOT}/{_METADATA_FILE}")]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each recording into one lazy Record, and stream one task per window and scheme.

        Every recording is read once up front to check it and keep its segments, windows and a digest of
        its acceleration values. The values stay in the files until written, so the files must stay
        available and unchanged until then.

        Args:
            raw_refs: The one extracted ``capture24`` directory.

        Returns:
            A dataset with the vocabulary annotations, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If ``raw_refs`` is not one directory.
            TimeNetConnectorError: If a table or recording is malformed.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
        if len(raw_refs) != 1 or not raw_refs[0].is_dir():
            raise TimeFValidationError(f"CAPTURE-24 loads from one directory, got {raw_refs}")
        (root,) = raw_refs
        labels = _read_labels(root / _DICTIONARY_FILE)
        # One participant's samples are held at a time: the first pass keeps only small results.
        participants = [_read_participant(person, labels) for person in _read_people(root)]
        acceleration = _OneRecording()
        records = [_record(participant, labels, acceleration) for participant in participants]

        dataset = TimeFDataset(metadata=self.metadata())
        for record in records:
            dataset.add_record(record=record)
        for vocabulary in _VOCABULARIES.values():
            dataset.annotate(Annotation(key=vocabulary.name, value=list(vocabulary.labels)))

        def tasks() -> Iterator[ClassificationTask]:
            # As the authors' benchmark.py: one benchmark per scheme, over every participant's windows.
            for scheme in _VOCABULARIES:
                for participant, record in zip(participants, records, strict=True):
                    yield from _scheme_tasks(scheme, participant, record)

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = Capture24Connector
