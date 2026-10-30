"""Read a CAPTURE-24 recording: one ``P<nnn>.csv.gz`` of ``time,x,y,z,annotation`` rows, one per sample.

``time`` is a timestamp with no zone, whose date the authors randomized. Its step is one sample period
everywhere except at eight gaps of 1 to 4 hours, one in each of eight recordings. A gap starts a new
segment. Nothing fills it.

The columns are read twice, apart. :func:`read_recording` reads ``time`` and ``annotation`` when the
release is parsed. It returns what the Record keeps, a :class:`Recording`, and apart from it the
per-sample :class:`Samples` that only the windowing needs. :func:`read_acceleration` reads ``x``, ``y``
and ``z`` when a signal is written.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import gzip
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.oxwearables.capture24.release import PARTICIPANT_COLUMNS, SAMPLE_PERIOD_US
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import NO_LABEL, LabelDictionary


AXES = ("x", "y", "z")
"""The acceleration columns, which name the signals."""
_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class Segment:
    """A run of samples exactly one period apart, between gaps."""

    first_row: int
    """The row of the segment's first sample."""
    n_values: int
    offset_us: int
    """The first sample's time from the recording's first sample."""


@dataclass(frozen=True)
class Recording:
    """What a recording's ``time`` and ``annotation`` columns state, without one value per sample.

    This is all the Record keeps. :class:`Samples` holds the per-sample columns apart from it.
    """

    path: Path
    """The ``P<nnn>.csv.gz`` it was parsed from, which :func:`read_acceleration` rereads."""
    n_rows: int
    """The rows it held when it was parsed, which a reread must match."""
    start_local: datetime
    """The first sample's timestamp, as written: no zone, and a randomized date."""
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class Samples:
    """A recording's clock and labels, one value per sample, about 100 MB for a day.

    Only the windowing reads them. Hold them for one participant at a time.
    """

    offsets_us: np.ndarray
    """Each sample's time from the first, int64 microseconds."""
    label_codes: np.ndarray
    """Each sample's raw label as an index into ``LabelDictionary.raw_labels``, or ``NO_LABEL``."""


def _check_header(path: Path) -> None:
    """Check a recording's header is exactly ``time,x,y,z,annotation``.

    Raises:
        TimeFFormatError: If it isn't.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header != list(PARTICIPANT_COLUMNS):
        raise TimeFFormatError(f"{path}: expected the header {','.join(PARTICIPANT_COLUMNS)}, got {header}")


def read_recording(path: Path, dictionary: LabelDictionary) -> tuple[Recording, Samples]:
    """Read a recording's ``time`` and ``annotation`` columns, and split them into segments and label runs.

    An empty ``annotation`` field is an unlabelled sample. No other value is.

    Args:
        path: The recording.
        dictionary: The label dictionary, which must list every raw label the recording writes.

    Returns:
        The recording's segments and runs, and apart from them its per-sample clock and labels.

    Raises:
        TimeFFormatError: If the header is wrong, the file has no rows, a timestamp is missing or unreadable,
            the timestamps don't increase, or a raw label isn't in the dictionary.
    """
    _check_header(path)
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
    except pa.ArrowInvalid as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows == 0:
        raise TimeFFormatError(f"{path}: holds no samples")
    times = table["time"].combine_chunks()
    if times.null_count:
        raise TimeFFormatError(f"{path}: {times.null_count} samples have no timestamp")
    stamps = times.cast(pa.int64()).to_numpy()
    steps = np.diff(stamps)
    if (steps <= 0).any():
        row_index = int(np.flatnonzero(steps <= 0)[0]) + 1
        raise TimeFFormatError(f"{path}: the timestamp of row {row_index} doesn't follow the one before it")
    offsets = stamps - stamps[0]
    labels = _raw_label_codes(path, table["annotation"].combine_chunks(), dictionary)
    breaks = np.flatnonzero(steps != SAMPLE_PERIOD_US) + 1
    bounds = [0, *breaks.tolist(), len(stamps)]
    segments = tuple(
        Segment(
            first_row=start,
            n_values=end - start,
            offset_us=int(offsets[start]),
        )
        for start, end in pairwise(bounds)
    )
    recording = Recording(
        path=path,
        n_rows=len(stamps),
        start_local=_EPOCH + timedelta(microseconds=int(stamps[0])),
        segments=segments,
    )
    return recording, Samples(offsets_us=offsets, label_codes=labels)


def read_acceleration(recording: Recording) -> dict[str, np.ndarray]:
    """Reread a parsed recording's x, y and z columns as float64.

    Args:
        recording: The recording, as :func:`read_recording` parsed it.

    Returns:
        One array per acceleration column.

    Raises:
        TimeFFormatError: If a value is missing, unreadable or not finite, or the row count changed.
    """
    path = recording.path
    try:
        table = pa_csv.read_csv(
            path,
            # Only an empty field is missing, so a written nan reaches the finite check below.
            convert_options=pa_csv.ConvertOptions(
                include_columns=list(AXES), column_types=dict.fromkeys(AXES, pa.float64()), null_values=[""]
            ),
        )
    except pa.ArrowInvalid as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows != recording.n_rows:
        raise TimeFFormatError(f"{path}: holds {table.num_rows} rows, {recording.n_rows} when it was parsed")
    values = {}
    for column in AXES:
        array = table[column].combine_chunks()
        if array.null_count:
            raise TimeFFormatError(f"{path}: {array.null_count} samples have no {column} value")
        values[column] = array.to_numpy()
        if not np.isfinite(values[column]).all():
            raise TimeFFormatError(f"{path}: a {column} value is not finite")
    return values


def _raw_label_codes(path: Path, column: pa.DictionaryArray, dictionary: LabelDictionary) -> np.ndarray:
    """Return each sample's raw label as an index into the dictionary, or ``NO_LABEL``.

    Returns:
        One int16 code per sample.

    Raises:
        TimeFFormatError: If the recording writes a raw label the dictionary doesn't list.
    """
    code_of = {label: code for code, label in enumerate(dictionary.raw_labels)}
    written = column.dictionary.to_pylist()
    unknown = sorted(label for label in written if label not in code_of)
    if unknown:
        raise TimeFFormatError(f"{path}: raw labels missing from the label dictionary: {unknown}")
    # One extra slot maps a null's filled index to NO_LABEL.
    codes = np.array([code_of[label] for label in written] + [NO_LABEL], dtype=np.int16)
    positions = column.indices.fill_null(len(written)).to_numpy()
    return codes[positions]
