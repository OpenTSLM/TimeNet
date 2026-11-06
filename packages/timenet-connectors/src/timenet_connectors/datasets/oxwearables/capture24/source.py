"""Read and check the files of the CAPTURE-24 release.

The release has a participant table (``metadata.csv``), a label dictionary
(``annotation-label-dictionary.csv``), and one gzipped CSV recording per participant.

The first pass reads one recording at a time. It checks every row and keeps a small summary: the
segments between gaps, the raw label runs, each kept window's start and class, and a digest of the
acceleration values. The writer reads the acceleration values again when it stores the signals.
The second read must match the first pass in both row count and acceleration digest.
"""

from collections.abc import Callable
import csv
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum, unique
import gzip
import hashlib
from itertools import islice
from pathlib import Path
import re
from typing import TypeVar
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.errors import TimeNetConnectorError
from timenet.types import LabelSet
from timenet_connectors.datasets.oxwearables.capture24.schemes import LABEL_SETS, Scheme
from timenet_connectors.datasets.oxwearables.capture24.windowing import (
    NO_LABEL,
    SAMPLE_PERIOD_US,
    LabelRuns,
    Segment,
    label_runs,
    make_segments,
    make_windows,
)
from timenet_connectors.sources.csv import CsvRow, check_unique, read_csv_table


METADATA_FILE = "metadata.csv"
_DICTIONARY_FILE = "annotation-label-dictionary.csv"


@unique
class _MetadataColumn(StrEnum):
    """A column of ``metadata.csv``."""

    PID = "pid"
    AGE = "age"
    SEX = "sex"


_DICTIONARY_LABEL_COLUMN = "annotation"
# A recording's file name, such as P001.csv.gz. The group is the participant id.
_RECORDING_FILE = re.compile(r"(P[0-9]{3})\.csv\.gz")
_TIME_COLUMN = "time"
_LABEL_COLUMN = "annotation"


Acceleration = dict[str, np.ndarray]
"""Float32 recording values keyed by axis name."""
AXES = ("x", "y", "z")
_RECORDING_COLUMNS = [_TIME_COLUMN, *AXES, _LABEL_COLUMN]
_FIRST_DATA_LINE = 2
"""The line number of the first data row, after the header."""

# The source writes eight significant digits. float32 keeps about seven, a step far below the sensor's
# resolution, and halves the stored size. The authors' own loader reads x, y and z as float32 too.
AXIS_TYPE = pa.float32()
_COLUMN_TYPES = {
    _TIME_COLUMN: pa.timestamp("us"),
    **dict.fromkeys(AXES, AXIS_TYPE),
    _LABEL_COLUMN: pa.dictionary(pa.int32(), pa.string()),
}


@unique
class Sex(StrEnum):
    """A participant's sex, as the connector stores it.

    The values match ``Sex`` in ``timenet_connectors.demographics``, which PR #677 adds.
    TODO(#677): use that enum and its annotation builders after PR #677 reaches the base branch.
    """

    FEMALE = "female"
    MALE = "male"


_SEX_OF_LETTER = {"F": Sex.FEMALE, "M": Sex.MALE}
"""The stored sex value for each letter in ``metadata.csv``."""


@unique
class AgeBand(StrEnum):
    """A participant's age band, as ``metadata.csv`` writes it. The authors give no exact age."""

    AGE_18_29 = "18-29"
    AGE_30_37 = "30-37"
    AGE_38_52 = "38-52"
    AGE_53_PLUS = "53+"


@dataclass(frozen=True)
class LabelDictionary:
    """The raw labels and class mappings used to build windows and activity annotations."""

    raw_labels: tuple[str, ...]
    """Each raw label, in file order."""
    raw_code_by_label: dict[str, int]
    """Each raw label's index, in file order."""
    scheme_classes_of_raw: tuple[dict[Scheme, str], ...]
    """Each raw label's class in all six schemes, as the dictionary names it."""
    class_index_by_raw_code: dict[Scheme, np.ndarray]
    """For each benchmarked scheme, this array maps raw labels to class indices in its label set.

    The final entry represents an unlabeled sample. Indexing with ``NO_LABEL`` selects that entry.
    A class's index breaks ties. ``Series.mode`` sorts the modes, and the authors' ``make_windows`` takes
    the first."""


@dataclass(frozen=True)
class ParticipantMetadata:
    """A participant listed in ``metadata.csv``, with the recording that holds the samples."""

    pid: str
    age_band: AgeBand
    sex: Sex
    path: Path


@dataclass(frozen=True)
class RecordingSummary:
    """The recording summary used to build one participant's Record and tasks.

    The first pass keeps this summary and discards the samples.
    """

    participant: ParticipantMetadata
    n_rows: int
    """The row count from the first pass. A later read with another count fails the build."""
    acceleration_digest: str
    """The digest of the x, y and z values from the first pass. A later read with another digest
    fails the build."""
    start_local: str
    """The first sample's published timestamp in ISO 8601 with microseconds and no zone. The date is
    randomized, and the authors shifted the time of day by a small random amount."""
    segments: tuple[Segment, ...]
    label_runs: tuple[LabelRuns, ...]
    """Each segment's runs of a raw label, in segment order."""
    window_starts_us: np.ndarray
    """The time between the recording's first sample and each kept window's first sample."""
    window_classes: dict[Scheme, np.ndarray]
    """Each kept window's class per benchmarked scheme, as an index into that scheme's label set."""


def read_labels(root: Path) -> LabelDictionary:
    """Read the label dictionary: each raw label and its class in each scheme.

    Args:
        root: The release directory.

    Returns:
        The raw labels and class mappings used to build windows and activity annotations.

    Raises:
        TimeNetConnectorError: If the file is unreadable or its columns are not the raw label and the six schemes.
            Also raised if a row has missing or extra fields, or a raw label appears twice.
            Also raised if a class of a benchmarked scheme is not in that scheme's label set.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by table and class checks
    path = root / _DICTIONARY_FILE
    rows = read_csv_table(path, columns=[_DICTIONARY_LABEL_COLUMN, *(scheme.column for scheme in Scheme)])
    raw_labels = [row.values[_DICTIONARY_LABEL_COLUMN] for row in rows]
    check_unique(path, ((row.line, row.values[_DICTIONARY_LABEL_COLUMN]) for row in rows), name="raw label")
    class_index_by_raw_code = {
        scheme: _class_codes(path, rows, scheme, label_set) for scheme, label_set in LABEL_SETS.items()
    }
    return LabelDictionary(
        raw_labels=tuple(raw_labels),
        raw_code_by_label={label: code for code, label in enumerate(raw_labels)},
        scheme_classes_of_raw=tuple({scheme: row.values[scheme.column] for scheme in Scheme} for row in rows),
        class_index_by_raw_code=class_index_by_raw_code,
    )


def _class_codes(path: Path, rows: list[CsvRow], scheme: Scheme, label_set: LabelSet) -> np.ndarray:
    """Map each raw label's class in one scheme to its index in the scheme's label set.

    Returns:
        One class index per raw label, in file order, followed by an entry for unlabeled samples.
        The final entry uses an index beyond the label set's classes, so class counts exclude it.
        Indexing with ``NO_LABEL`` selects this entry.

    Raises:
        TimeNetConnectorError: If a class is not in the label set.
    """
    index_of_class = {name: index for index, name in enumerate(label_set.names)}
    codes = []
    for row in rows:
        name = row.values[scheme.column]
        if name not in index_of_class:
            raise TimeNetConnectorError(
                f"{path}:{row.line}: {scheme.column}: class {name!r} is not in label set {label_set.name!r}"
            )
        codes.append(index_of_class[name])
    unlabeled = len(label_set.labels)
    return np.array([*codes, unlabeled], dtype=np.int16)


_FieldValue = TypeVar("_FieldValue")


def _parse_metadata_field(column: str, value: str, parse: Callable[[str], _FieldValue]) -> _FieldValue:
    """Parse one cell of ``metadata.csv``.

    Returns:
        The parsed value.

    Raises:
        TimeNetConnectorError: If ``parse`` does not accept the value.
    """
    try:
        return parse(value)
    except (KeyError, ValueError) as exc:
        raise TimeNetConnectorError(f"unknown {column} {value!r}") from exc


def read_participant_metadata(root: Path) -> list[ParticipantMetadata]:
    """Read ``metadata.csv`` and pair each row with its recording.

    Args:
        root: The release directory.

    Returns:
        Each participant's metadata and recording path, sorted by participant ID.

    Raises:
        TimeNetConnectorError: If the file is unreadable, its columns are not pid, age and sex, or it lists
            no participants. Also raised if a row has missing or extra fields.
            Also raised if an age band or sex is unknown, or a pid repeats.
            Also raised if the table and the recording files name different participants.
    """
    path = root / METADATA_FILE
    rows = read_csv_table(path, columns=[column.value for column in _MetadataColumn])
    if not rows:
        raise TimeNetConnectorError(f"{path}: lists no participants")
    pids = [row.values[_MetadataColumn.PID] for row in rows]
    check_unique(path, ((row.line, row.values[_MetadataColumn.PID]) for row in rows), name="pid")
    recordings = _recordings_by_pid(root)
    if set(recordings) != set(pids):
        raise TimeNetConnectorError(
            f"{root}: recordings missing for {sorted(set(pids) - set(recordings))}, "
            f"unlisted recordings {sorted(set(recordings) - set(pids))}"
        )
    participants = []
    for row in rows:
        try:
            participants.append(_parse_participant_metadata(row.values, recordings))
        except TimeNetConnectorError as exc:
            raise TimeNetConnectorError(f"{path}:{row.line}: {exc}") from exc
    return sorted(participants, key=lambda participant: participant.pid)


def _parse_participant_metadata(row: dict[str, str], recordings: dict[str, Path]) -> ParticipantMetadata:
    """Parse a metadata row and attach the participant's recording path.

    Returns:
        The ``ParticipantMetadata`` with the participant's id, age band, sex, and recording path.

    Raises:
        TimeNetConnectorError: If the connector does not recognize the age band or sex.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _parse_metadata_field
    pid = row[_MetadataColumn.PID]
    age_band = _parse_metadata_field(_MetadataColumn.AGE, row[_MetadataColumn.AGE], AgeBand)
    sex_letter = row[_MetadataColumn.SEX]
    sex = _parse_metadata_field(_MetadataColumn.SEX, sex_letter, _SEX_OF_LETTER.__getitem__)
    return ParticipantMetadata(pid=pid, age_band=age_band, sex=sex, path=recordings[pid])


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


def summarize_recording(participant: ParticipantMetadata, labels: LabelDictionary) -> RecordingSummary:
    """Summarize a recording's segments, raw label runs, kept windows, and acceleration digest.

    Only an empty ``annotation`` field marks an unlabeled sample. Any other text must be a raw label
    from the label dictionary.

    Args:
        participant: The participant whose recording to read.
        labels: The label dictionary, which every raw label in the recording must be in.

    Returns:
        The recording summary, without the sample arrays.

    Raises:
        TimeNetConnectorError: If the file is unreadable, its header is wrong, or it has no rows.
            Also raised if a timestamp is missing or unreadable, or a step is shorter than one period.
            Also raised if a raw label is not in the dictionary, or if a bin with 1000 samples and a
            labeled sample crosses a gap. Also raised if an acceleration value is missing or not finite.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
    path = participant.path
    table = _read_table(path)
    timestamps_us = _read_timestamps(path, table)
    offsets = timestamps_us - timestamps_us[0]
    raw_codes = _raw_codes(path, table[_LABEL_COLUMN].combine_chunks(), labels)
    segments = make_segments(timestamps_us)
    windows = make_windows(path, offsets, raw_codes, segments, labels.class_index_by_raw_code)
    return RecordingSummary(
        participant=participant,
        n_rows=len(timestamps_us),
        acceleration_digest=_acceleration_digest(_axis_values(path, table)),
        start_local=table[_TIME_COLUMN][0].as_py().isoformat(timespec="microseconds"),
        segments=segments,
        label_runs=tuple(label_runs(raw_codes, segment) for segment in segments),
        window_starts_us=offsets[windows.first_rows],
        window_classes=windows.classes,
    )


def _line_number(row: int) -> int:
    """Return the file line number of a recording's data row, counted from 0."""
    return row + _FIRST_DATA_LINE


def _first_null_row(array: pa.Array) -> int:
    """Return the first row whose value is missing. The array must have one."""
    return int(np.flatnonzero(array.is_null().to_numpy(zero_copy_only=False))[0])


def _convert_options(columns: list[str]) -> pa_csv.ConvertOptions:
    """Return the same parsing configuration for both reads of a recording's ``columns``.

    Only an empty field counts as missing. A literal ``nan`` stays a value. In x, y or z it reads as
    NaN, and ``_axis_values`` then fails the build because the value is not finite.
    """
    return pa_csv.ConvertOptions(
        include_columns=columns,
        column_types={column: _COLUMN_TYPES[column] for column in columns},
        strings_can_be_null=True,
        null_values=[""],
    )


def _read_table(path: Path) -> pa.Table:
    """Read every column of a recording.

    The ``annotation`` column stores repeated labels through a shared dictionary.

    Returns:
        The recording's table.

    Raises:
        TimeNetConnectorError: If the file cannot be read, the header is wrong, or the file has no rows.
    """
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error, zlib.error) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if header != _RECORDING_COLUMNS:
        raise TimeNetConnectorError(f"{path}: expected the header {','.join(_RECORDING_COLUMNS)}, got {header}")
    try:
        table = pa_csv.read_csv(
            path,
            parse_options=pa_csv.ParseOptions(ignore_empty_lines=False),
            convert_options=_convert_options(_RECORDING_COLUMNS),
        )
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if table.num_rows == 0:
        raise TimeNetConnectorError(f"{path}: holds no samples")
    return table


def _read_timestamps(path: Path, table: pa.Table) -> np.ndarray:
    """Return each sample's time in microseconds.

    Raises:
        TimeNetConnectorError: If a timestamp is missing, or a step is shorter than one period.
    """
    times = table[_TIME_COLUMN].combine_chunks()
    if times.null_count:
        raise TimeNetConnectorError(f"{path}:{_line_number(_first_null_row(times))}: no timestamp")
    timestamps_us = times.cast(pa.int64()).to_numpy()
    # Reject any step shorter than one period: it is neither a regular step nor a gap.
    short = np.flatnonzero(np.diff(timestamps_us) < SAMPLE_PERIOD_US)
    if len(short):
        raise TimeNetConnectorError(
            f"{path}:{_line_number(int(short[0]) + 1)}: the timestamp doesn't follow the one before it by a full period"
        )
    return timestamps_us


def _axis_values(path: Path, table: pa.Table) -> Acceleration:
    """Return the x, y and z columns of a recording's table.

    Raises:
        TimeNetConnectorError: If a value is missing, nonfinite or overflows float32.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _finite_column
    return {column: _finite_column(path, table, column) for column in AXES}


def _finite_column(path: Path, table: pa.Table, column: str) -> np.ndarray:
    """Return one acceleration column of a recording's table.

    Raises:
        TimeNetConnectorError: If a value is missing, nonfinite or overflows float32. The error names the first such row.
    """
    array = table[column].combine_chunks()
    if array.null_count:
        raise TimeNetConnectorError(f"{path}:{_line_number(_first_null_row(array))}: no {column} value")
    values = array.to_numpy()
    not_finite = np.flatnonzero(~np.isfinite(values))
    if len(not_finite):
        row = int(not_finite[0])
        # Only reread source text on failure: a finite decimal may have overflowed float32.
        reason = "out of float32 range" if _source_value_is_finite(path, row, column) else "not finite"
        raise TimeNetConnectorError(f"{path}:{_line_number(row)}: the {column} value is {reason}")
    return values


def _source_value_is_finite(path: Path, row: int, column: str) -> bool:
    """Check the original decimal after Arrow produced a nonfinite float32 value.

    Returns:
        Whether the source value is finite, even if it exceeds float64 range.

    Raises:
        TimeNetConnectorError: If the source cannot be reread.
    """
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cells = next(islice(reader, row, row + 1))
            return Decimal(cells[column]).is_finite()
    except (
        OSError,
        EOFError,
        UnicodeDecodeError,
        csv.Error,
        zlib.error,
        StopIteration,
        ArithmeticError,
        ValueError,
    ) as exc:
        raise TimeNetConnectorError(f"{path}:{_line_number(row)}: cannot reread {column} value: {exc}") from exc


def _acceleration_digest(values: Acceleration) -> str:
    """Return a digest of the x, y and z values, in that order.

    The digest detects changes between reads.
    """
    # Hash the float32 values we store, in a fixed axis order. Text formatting changes and
    # differences below float32 precision do not change the stored signal and need not fail a reread.
    digest = hashlib.blake2b()
    for column in AXES:
        digest.update(values[column])
    return digest.hexdigest()


def _raw_codes(path: Path, column: pa.DictionaryArray, labels: LabelDictionary) -> np.ndarray:
    """Return each sample's raw label as an index into the dictionary, or ``NO_LABEL``.

    Raises:
        TimeNetConnectorError: If the recording writes a raw label the dictionary does not list.
    """
    dictionary_labels = column.dictionary.to_pylist()
    unknown = sorted(label for label in dictionary_labels if label not in labels.raw_code_by_label)
    if unknown:
        unknown_indices = [index for index, label in enumerate(dictionary_labels) if label in unknown]
        indices = column.indices.fill_null(-1).to_numpy()
        row = int(np.flatnonzero(np.isin(indices, unknown_indices))[0])
        label = dictionary_labels[indices[row]]
        raise TimeNetConnectorError(
            f"{path}:{_line_number(row)}: raw label missing from the label dictionary: {label!r}"
        )
    # Replace each null index with the index of an extra entry whose value is NO_LABEL.
    codes = np.array([labels.raw_code_by_label[label] for label in dictionary_labels] + [NO_LABEL], dtype=np.int16)
    return codes[column.indices.fill_null(len(dictionary_labels)).to_numpy()]


def read_acceleration(summary: RecordingSummary) -> Acceleration:
    """Read a recording's x, y and z columns as float32.

    The build fails if the row count or the acceleration values differ from the first pass.
    Timestamps and labels come from the first-pass summary; this read checks only the signal payload.

    Args:
        summary: The recording summary from the first pass.

    Returns:
        Float32 arrays keyed by axis name.

    Raises:
        TimeNetConnectorError: If a value is missing, unreadable or not finite, or the row count or the values
            changed.
    """
    path = summary.participant.path
    try:
        table = pa_csv.read_csv(
            path,
            parse_options=pa_csv.ParseOptions(ignore_empty_lines=False),
            convert_options=_convert_options(list(AXES)),
        )
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
    if table.num_rows != summary.n_rows:
        raise TimeNetConnectorError(
            f"{path}: holds {table.num_rows} rows, {summary.n_rows} when it was read; the file changed"
        )
    values = _axis_values(path, table)
    if _acceleration_digest(values) != summary.acceleration_digest:
        raise TimeNetConnectorError(f"{path}: the acceleration values differ from when it was read; the file changed")
    return values


class RecordingCache:
    """Read a recording's acceleration once for all its signals and keep only the latest recording.

    A Record's signals load one after another, so the cache needs only one recording. The class does
    not use ``lru_cache(maxsize=1)``, because that cache reads the next recording before it releases
    the previous one, so both are in memory for a moment. This class releases the previous one first.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._values: Acceleration | None = None

    def __call__(self, summary: RecordingSummary) -> Acceleration:
        """Return the recording's x, y and z values.

        If the cache holds this recording, the method returns its values.
        Otherwise, it releases the previous recording before reading this one.

        """
        if self._values is None or self._path != summary.participant.path:
            # Release the previous recording before reading the next.
            self._path, self._values = None, None
            self._values = read_acceleration(summary)
            self._path = summary.participant.path
        return self._values
