"""Read and check the published CAPTURE-24 files, and select the windows of the authors' benchmark."""

from dataclasses import dataclass
from enum import StrEnum, unique
import gzip
import hashlib
from itertools import pairwise
from pathlib import Path
import re
from typing import Literal, NamedTuple
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
from pydantic import Field

from timenet.errors import TimeNetConnectorError
from timenet.types import US_PER_S
from timenet.types.metadata_fields import NonEmptyText
from timenet_connectors.sources.readers import CsvDataReader, DataRow


METADATA_FILE = "metadata.csv"
"""The participant table, which also marks the release directory."""
_METADATA_COLUMNS = ("pid", "age", "sex")

DICTIONARY_FILE = "annotation-label-dictionary.csv"
_DICTIONARY_LABEL_COLUMN = "annotation"
# A recording's file name, such as P001.csv.gz. The group is the participant id.
_RECORDING_FILE = re.compile(r"(P[0-9]{3})\.csv\.gz")
_TIME_COLUMN = "time"
_LABEL_COLUMN = "annotation"


class AccelerationValues(NamedTuple):
    """A recording's x, y, and z acceleration arrays with one float32 value per sample."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


AXES = AccelerationValues._fields
_RECORDING_COLUMNS = [_TIME_COLUMN, *AXES, _LABEL_COLUMN]

# The paper's Methods: an Axivity AX3 on the dominant wrist, at 100 Hz with a range of ±8 g. The
# authors resampled every recording to exactly 100 Hz before release.
DEVICE = "Axivity AX3"
PLACEMENT = "dominant wrist"
SAMPLE_RATE_HZ = 100
_SAMPLE_PERIOD_US = US_PER_S // SAMPLE_RATE_HZ
# The source writes eight significant digits. float32 keeps about seven, which is still far more
# precise than the sensor's resolution, and it halves the stored size. The authors' own loader
# reads x, y and z as float32 too.
AXIS_TYPE = pa.float32()
_COLUMN_TYPES = {
    _TIME_COLUMN: pa.timestamp("us"),
    **dict.fromkeys(AXES, AXIS_TYPE),
    _LABEL_COLUMN: pa.dictionary(pa.int32(), pa.string()),
}


@unique
class Sex(StrEnum):
    """A participant's recorded sex."""

    FEMALE = "female"
    MALE = "male"


_SEX_FROM_LETTER = {"F": Sex.FEMALE, "M": Sex.MALE}


@unique
class AgeBand(StrEnum):
    """A participant's age range from the metadata table."""

    AGE_18_29 = "18-29"
    AGE_30_37 = "30-37"
    AGE_38_52 = "38-52"
    AGE_53_PLUS = "53+"


class _MetadataRow(DataRow):
    """A row of the metadata table."""

    pid: NonEmptyText
    age: AgeBand
    sex: Literal["F", "M"]


# What the authors' prepare_data.py (``--winsec 10``) fixes.
WINDOW_US = 10 * US_PER_S
_WINDOW_SAMPLES = WINDOW_US // _SAMPLE_PERIOD_US
_NO_LABEL = -1
"""The raw label code of an unlabeled sample.

As an index, it selects the "no class" entry at the end of each class array.
"""


@unique
class LabelScheme(StrEnum):
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
        return f"label:{self.value}"


_DICTIONARY_COLUMNS = (_DICTIONARY_LABEL_COLUMN, *(scheme.column for scheme in LabelScheme))


class _DictionaryRow(DataRow):
    """A row of the label dictionary: a raw label and its class in each scheme, none of them blank."""

    annotation: NonEmptyText
    willetts_specific_2018: NonEmptyText = Field(alias=LabelScheme.WILLETTS_SPECIFIC_2018.column)
    willetts_met_2018: NonEmptyText = Field(alias=LabelScheme.WILLETTS_MET_2018.column)
    doherty_specific_2018: NonEmptyText = Field(alias=LabelScheme.DOHERTY_SPECIFIC_2018.column)
    willetts_2018: NonEmptyText = Field(alias=LabelScheme.WILLETTS_2018.column)
    doherty_2018: NonEmptyText = Field(alias=LabelScheme.DOHERTY_2018.column)
    walmsley_2020: NonEmptyText = Field(alias=LabelScheme.WALMSLEY_2020.column)


# The classes of each scheme, as the label dictionary names them. Keep each list in alphabetical
# order: on a tie, the authors' benchmark picks the class that comes first alphabetically, and so
# does the connector.
SCHEME_CLASSES = {
    LabelScheme.WILLETTS_SPECIFIC_2018: (
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
    LabelScheme.WILLETTS_MET_2018: (
        "bicycling",
        "gym",
        "sitstand+activity",
        "sitstand+lowactivity",
        "sitting",
        "sleep",
        "sports",
        "standing",
        "vehicle",
        "walking",
        "walking+activity",
    ),
    LabelScheme.DOHERTY_SPECIFIC_2018: (
        "bicycling",
        "sedentary-non-screen",
        "sedentary-screen",
        "sleep",
        "sport-interrupted",
        "sports-continuous",
        "tasks-light",
        "tasks-moderate",
        "vehicle",
        "walking",
    ),
    LabelScheme.WILLETTS_2018: ("bicycling", "mixed", "sit-stand", "sleep", "vehicle", "walking"),
    LabelScheme.DOHERTY_2018: ("moderate", "sedentary", "sleep", "tasks-light", "walking"),
    LabelScheme.WALMSLEY_2020: ("light", "moderate-vigorous", "sedentary", "sleep"),
}
# The two schemes that the authors' benchmark.py scores, in its order.
BENCHMARK_SCHEMES = (LabelScheme.WILLETTS_SPECIFIC_2018, LabelScheme.WALMSLEY_2020)


@dataclass(frozen=True)
class LabelDictionary:
    """Raw activity labels and their class mappings."""

    raw_labels: tuple[str, ...]
    """The raw labels in dictionary row order. A raw label's code is its index here."""
    scheme_classes_of_raw: tuple[dict[LabelScheme, str], ...]
    """The class of each raw label in all six schemes."""
    class_of_raw: dict[LabelScheme, np.ndarray]
    """The class indices for each benchmark scheme in raw label order.

    The final entry represents unlabeled samples and is selected by ``_NO_LABEL``.
    On a tie in class counts, the lower index wins, as in the authors' ``make_windows``.
    """


@dataclass(frozen=True)
class LabelRuns:
    """A segment's consecutive sample runs with the same nonempty raw label."""

    starts_us: np.ndarray
    """Each run's start offset in microseconds from the recording's first sample."""
    ends_us: np.ndarray
    """Each run's exclusive end offset in microseconds from the recording's first sample.

    The end falls one sample period after the last sample.
    """
    codes: np.ndarray
    """The code of each run's raw label."""


@dataclass(frozen=True)
class Segment:
    """A sequence of samples spaced exactly one sample period apart."""

    first_row: int
    n_values: int
    offset_us: int
    """The segment's start offset in microseconds from the recording's first sample."""
    runs: LabelRuns


@dataclass(frozen=True)
class Participant:
    """A participant's metadata and recording path."""

    pid: str
    age_band: AgeBand
    sex: Sex
    path: Path


@dataclass(frozen=True)
class Recording:
    """One participant's recording file, summarized without its acceleration samples."""

    participant: Participant
    n_rows: int
    """The expected recording row count."""
    acceleration_digest: str
    """The expected digest of the x, y, and z acceleration values."""
    start_local: str
    """The first sample's published ISO 8601 timestamp with microseconds and no time zone.

    The authors randomized the date and shifted the time of day.
    """
    segments: tuple[Segment, ...]
    window_starts_us: np.ndarray
    """Each selected window's start offset in microseconds from the recording's first sample."""
    window_classes: dict[LabelScheme, np.ndarray]
    """The class index of each selected window in each benchmark scheme's ``SCHEME_CLASSES``."""


def read_labels(path: Path) -> LabelDictionary:
    """Read raw activity labels and their classes in all six schemes.

    Returns:
        The raw labels and class mappings.

    Raises:
        TimeNetConnectorError: If reading fails or columns, fields, raw labels, or classes are invalid.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the check helpers
    reader = CsvDataReader(path, _DictionaryRow, expected_columns=_DICTIONARY_COLUMNS)
    rows = [row.model_dump(by_alias=True) for row in reader.iter_rows()]
    raw_labels = [row[_DICTIONARY_LABEL_COLUMN] for row in rows]
    _check_raw_labels_are_unique(path, raw_labels)
    _check_classes_are_listed(path, rows)
    # The class of each raw label in each of the six schemes.
    scheme_classes_of_raw: list[dict[LabelScheme, str]] = []
    for row in rows:
        scheme_classes_of_raw.append({scheme: row[scheme.column] for scheme in LabelScheme})
    # The class index of each raw label in the class list of each benchmark scheme.
    class_of_raw: dict[LabelScheme, np.ndarray] = {}
    for scheme in BENCHMARK_SCHEMES:
        classes = [row[scheme.column] for row in rows]
        class_of_raw[scheme] = _class_codes(classes, scheme)
    return LabelDictionary(
        raw_labels=tuple(raw_labels),
        scheme_classes_of_raw=tuple(scheme_classes_of_raw),
        class_of_raw=class_of_raw,
    )


def _check_raw_labels_are_unique(path: Path, raw_labels: list[str]) -> None:
    """Check that the label dictionary lists each raw label only once.

    Raises:
        TimeNetConnectorError: If a raw label is listed twice.
    """
    if len(set(raw_labels)) != len(raw_labels):
        raise TimeNetConnectorError(f"{path}: a raw label is listed twice")


def _check_classes_are_listed(path: Path, rows: list[dict[str, str]]) -> None:
    """Check that every class in the label dictionary is in the class list of its scheme.

    Raises:
        TimeNetConnectorError: If a class is absent from the class list of its scheme.
    """
    for scheme in LabelScheme:
        unknown = sorted({row[scheme.column] for row in rows} - set(SCHEME_CLASSES[scheme]))
        if unknown:
            raise TimeNetConnectorError(f"{path}: classes missing from the class list of {scheme}: {unknown}")


def _class_codes(classes: list[str], scheme: LabelScheme) -> np.ndarray:
    """Map activity classes to indices in the class list of a scheme.

    Returns:
        Class indices in input order followed by the class count for unlabeled samples.
    """
    names = SCHEME_CLASSES[scheme]
    index_of_class = {name: index for index, name in enumerate(names)}
    codes = [index_of_class[name] for name in classes]
    unlabeled = len(names)
    return np.array([*codes, unlabeled], dtype=np.int16)


def read_participants(root: Path) -> list[Participant]:
    """Read participant metadata and match each participant to a recording.

    Returns:
        Participants sorted by ID.

    Raises:
        TimeNetConnectorError: If reading fails or metadata is invalid or inconsistent with recording files.
    """
    path = root / METADATA_FILE
    rows = list(CsvDataReader(path, _MetadataRow, expected_columns=_METADATA_COLUMNS).iter_rows())
    pids = [row.pid for row in rows]
    if len(set(pids)) != len(pids):
        raise TimeNetConnectorError(f"{path}: a pid is listed twice")
    recordings = _recordings_by_pid(root)
    if set(recordings) != set(pids):
        raise TimeNetConnectorError(
            f"{root}: recordings missing for {sorted(set(pids) - set(recordings))}, "
            f"unlisted recordings {sorted(set(recordings) - set(pids))}"
        )
    participants = [
        Participant(pid=row.pid, age_band=row.age, sex=_SEX_FROM_LETTER[row.sex], path=recordings[row.pid])
        for row in rows
    ]
    return sorted(participants, key=lambda participant: participant.pid)


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


def read_recording(participant: Participant, labels: LabelDictionary) -> Recording:
    """Summarize a recording's segments, labels, windows, and acceleration digest.

    An empty annotation field marks an unlabeled sample.
    All other annotation values must appear in the label dictionary.

    Returns:
        The recording summary without acceleration samples.

    Raises:
        TimeNetConnectorError: If reading fails or the header, timestamps, labels, windows, or acceleration are invalid.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
    path = participant.path
    table = _read_table(path)
    stamps = _read_stamps(path, table)
    offsets = stamps - stamps[0]
    raw_codes = _raw_codes(path, table[_LABEL_COLUMN].combine_chunks(), labels)
    first_rows, window_classes = _make_windows(path, offsets, raw_codes, labels)
    return Recording(
        participant=participant,
        n_rows=len(stamps),
        acceleration_digest=_digest(_axis_values(path, table)),
        start_local=table[_TIME_COLUMN][0].as_py().isoformat(timespec="microseconds"),
        segments=_make_segments(stamps, raw_codes),
        window_starts_us=offsets[first_rows],
        window_classes=window_classes,
    )


def _read_csv_gz(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read a gzipped recording, with all its columns or only the given ones.

    Reading all columns checks that the header is exactly ``time,x,y,z,annotation``, so no extra
    column is dropped unnoticed. Only empty fields count as missing values. Python's gzip inflates
    the file, which is about twice as fast as Arrow's own decompression.

    Returns:
        The recording's columns.

    Raises:
        TimeNetConnectorError: If the file cannot be read or parsed, or its header is wrong.
    """
    options = pa_csv.ConvertOptions(
        include_columns=columns or [],
        column_types=_COLUMN_TYPES,
        strings_can_be_null=True,
        null_values=[""],
    )
    try:
        with gzip.open(path, "rb") as stream:
            table = pa_csv.read_csv(stream, convert_options=options)
    except (OSError, EOFError, zlib.error, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if columns is None and table.column_names != _RECORDING_COLUMNS:
        raise TimeNetConnectorError(
            f"{path}: expected the header {','.join(_RECORDING_COLUMNS)}, got {','.join(table.column_names)}"
        )
    return table


def _read_table(path: Path) -> pa.Table:
    """Read all recording columns into an Arrow table.

    Returns:
        The recording table with dictionary-encoded activity labels.

    Raises:
        TimeNetConnectorError: If reading fails, the header is invalid, or the recording is empty.
    """
    table = _read_csv_gz(path)
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


def _make_segments(stamps: np.ndarray, raw_codes: np.ndarray) -> tuple[Segment, ...]:
    """Split samples at timestamp steps that differ from one sample period.

    Returns:
        Sample segments in time order, each with its label runs.
    """
    bounds = [0, *(np.flatnonzero(np.diff(stamps) != _SAMPLE_PERIOD_US) + 1).tolist(), len(stamps)]
    segments = []
    for start, end in pairwise(bounds):
        offset_us = int(stamps[start] - stamps[0])
        runs = _label_runs(raw_codes[start:end], offset_us)
        segments.append(Segment(first_row=start, n_values=end - start, offset_us=offset_us, runs=runs))
    return tuple(segments)


def _label_runs(codes: np.ndarray, offset_us: int) -> LabelRuns:
    """Find consecutive samples with the same nonempty raw label within a segment.

    Returns:
        Label runs in time order.
    """
    changes = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(codes)]))
    labeled = codes[starts] != _NO_LABEL
    starts, ends = starts[labeled], ends[labeled]
    return LabelRuns(
        starts_us=offset_us + starts * _SAMPLE_PERIOD_US,
        ends_us=offset_us + ends * _SAMPLE_PERIOD_US,
        codes=codes[starts],
    )


def _axis_values(path: Path, table: pa.Table) -> AccelerationValues:
    """Read the x, y, and z acceleration columns from a recording table.

    Returns:
        One float32 array per acceleration axis.

    Raises:
        TimeNetConnectorError: If an acceleration value is missing or nonfinite.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _finite_column
    return AccelerationValues._make(_finite_column(path, table, column) for column in AXES)


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


def _digest(values: AccelerationValues) -> str:
    """Compute a digest of acceleration values in x, y, and z order.

    Returns:
        The hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for axis in values:
        digest.update(axis)
    return digest.hexdigest()


def _check_raw_labels_are_in_dictionary(path: Path, written: list[str], labels: LabelDictionary) -> None:
    """Check that the label dictionary lists every raw label that a recording writes.

    Raises:
        TimeNetConnectorError: If a raw label is absent from the dictionary.
    """
    unknown = sorted(set(written) - set(labels.raw_labels))
    if unknown:
        raise TimeNetConnectorError(f"{path}: raw labels missing from the label dictionary: {unknown}")


def _raw_codes(path: Path, column: pa.DictionaryArray, labels: LabelDictionary) -> np.ndarray:
    """Map sample labels to raw label codes.

    Returns:
        Raw label codes with ``_NO_LABEL`` for unlabeled samples.

    Raises:
        TimeNetConnectorError: If a raw label is absent from the dictionary.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _check_raw_labels_are_in_dictionary
    written = column.dictionary.to_pylist()
    _check_raw_labels_are_in_dictionary(path, written, labels)
    code_of_raw = {label: code for code, label in enumerate(labels.raw_labels)}
    # Replace each null index with the index of an extra entry whose value is _NO_LABEL.
    codes = np.array([code_of_raw[label] for label in written] + [_NO_LABEL], dtype=np.int16)
    return codes[column.indices.fill_null(len(written)).to_numpy()]


def _make_windows(
    path: Path, offsets_us: np.ndarray, raw_codes: np.ndarray, labels: LabelDictionary
) -> tuple[np.ndarray, dict[LabelScheme, np.ndarray]]:
    """Select 10-second windows and assign their activity classes.

    Bins start at the first sample and require 1000 samples with at least one label.
    Each scheme uses the most common labeled class and breaks ties by class name order.

    Returns:
        The first row of each selected window, and the window classes for each benchmark scheme.

    Raises:
        TimeNetConnectorError: If a selected window crosses a gap.
    """
    # Each sample's 10 s bin on the first sample's grid, and the first row of each bin that has samples.
    bins = offsets_us // WINDOW_US
    bin_starts = np.flatnonzero(np.diff(bins, prepend=bins[0] - 1))
    # Keep the bins that hold exactly 1000 samples: the rows up to the next bin's first row.
    full = bin_starts[np.diff(bin_starts, append=len(bins)) == _WINDOW_SAMPLES]
    # One row of 1000 raw label codes per full bin. Keep the bins with at least one labeled sample.
    window_codes = raw_codes[full[:, None] + np.arange(_WINDOW_SAMPLES)]
    labeled = (window_codes != _NO_LABEL).any(axis=1)
    first_rows, window_codes = full[labeled], window_codes[labeled]
    # No step is shorter than one period, so a window spans 999 periods only if no gap falls inside it.
    spans = offsets_us[first_rows + _WINDOW_SAMPLES - 1] - offsets_us[first_rows]
    crossing = spans != (_WINDOW_SAMPLES - 1) * _SAMPLE_PERIOD_US
    if crossing.any():
        row = int(first_rows[crossing][0])
        raise TimeNetConnectorError(f"{path}: the window at row {row} holds {_WINDOW_SAMPLES} samples across a gap")
    classes = {}
    for scheme, class_of_raw in labels.class_of_raw.items():
        per_sample = class_of_raw[window_codes]
        # Count each class per window. Unlabeled samples use an index beyond the classes, so the counts exclude them.
        counts = np.stack([(per_sample == index).sum(axis=1) for index in range(len(SCHEME_CLASSES[scheme]))], axis=1)
        # argmax takes the first of equal counts: the class first in string order.
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return first_rows, classes


def read_acceleration(recording: Recording) -> AccelerationValues:
    """Read and validate a recording's acceleration against its saved digest.

    Returns:
        One float32 array per acceleration axis.

    Raises:
        TimeNetConnectorError: If reading fails, a value is missing or nonfinite, or the row count or values changed.
    """
    path = recording.participant.path
    table = _read_csv_gz(path, list(AXES))
    if table.num_rows != recording.n_rows:
        raise TimeNetConnectorError(
            f"{path}: holds {table.num_rows} rows, {recording.n_rows} when it was read; the file changed"
        )
    values = _axis_values(path, table)
    if _digest(values) != recording.acceleration_digest:
        raise TimeNetConnectorError(f"{path}: the acceleration values differ from when it was read; the file changed")
    return values
