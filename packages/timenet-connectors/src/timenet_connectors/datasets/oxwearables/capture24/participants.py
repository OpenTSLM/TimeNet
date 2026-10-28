"""Read CAPTURE-24's participants: the metadata table, and each recording's clock and labels.

A recording is one ``P<nnn>.csv.gz`` of ``time,x,y,z,annotation`` rows, one per sample. ``time`` is a
wall-clock timestamp with no zone. Its step is one sample period everywhere except at eight gaps of 1 to
4 hours, one in each of eight recordings. A gap starts a new segment. Nothing fills it.

:func:`read_timeline` reads only ``time`` and ``annotation``. The acceleration columns stay in the file
until a signal is written; ``records`` rereads them.
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
from timenet_connectors.datasets.oxwearables.capture24.release import (
    METADATA_FILE,
    PARTICIPANT_COLUMNS,
    PARTICIPANT_FILE,
    SAMPLE_PERIOD_US,
)
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import NO_LABEL, LabelDictionary


_METADATA_COLUMNS = ["pid", "age", "sex"]
_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class Participant:
    """One participant: their recording file and the row ``metadata.csv`` states for them."""

    pid: str
    """The participant id, such as ``P001``, which also names the file."""
    number: int
    path: Path
    age: str
    sex: str


@dataclass(frozen=True)
class Segment:
    """A run of samples exactly one period apart, between gaps."""

    first_index: int
    """The row of the segment's first sample."""
    n_values: int
    offset_us: int
    """The first sample's time from the recording's first sample."""


@dataclass(frozen=True)
class LabelRun:
    """Consecutive samples of one segment that carry one raw label."""

    segment: int
    raw_label: int
    """An index into ``LabelDictionary.raw_labels``."""
    start_us: int
    end_us: int
    """One period past the run's last sample."""


@dataclass(frozen=True)
class Timeline:
    """What a recording's ``time`` and ``annotation`` columns state.

    ``offsets_us`` and ``labels`` hold one value per sample. The loader keeps only what it derives from them.
    """

    n_rows: int
    start_local: datetime
    """The first sample's timestamp, as written: local time, no zone."""
    offsets_us: np.ndarray
    """Each sample's time from the first, int64 microseconds."""
    labels: np.ndarray
    """Each sample's raw label as an index into ``LabelDictionary.raw_labels``, or ``NO_LABEL``."""
    segments: tuple[Segment, ...]
    label_runs: tuple[LabelRun, ...]


def read_participants(root: Path) -> tuple[Participant, ...]:
    """Read ``metadata.csv`` and pair each row with its recording.

    Returns:
        One participant per recording, in id order.

    Raises:
        TimeFFormatError: If the table's header isn't ``pid,age,sex``, a field is empty, a pid repeats, or
            the table and the recordings don't name the same participants.
    """
    path = root / METADATA_FILE
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != _METADATA_COLUMNS:
        raise TimeFFormatError(f"{path}: expected the header {','.join(_METADATA_COLUMNS)}")
    if any(len(row) != len(_METADATA_COLUMNS) or not all(row) for row in rows[1:]):
        raise TimeFFormatError(f"{path}: every row needs a pid, an age and a sex")
    table = {row[0]: row for row in rows[1:]}
    if len(table) != len(rows) - 1:
        raise TimeFFormatError(f"{path}: a pid is listed twice")
    files = {}
    for file in root.iterdir():
        match = PARTICIPANT_FILE.fullmatch(file.name)
        if match is not None:
            files[file.name.removesuffix(".csv.gz")] = (int(match["number"]), file)
    if set(files) != set(table):
        missing, extra = sorted(set(table) - set(files)), sorted(set(files) - set(table))
        raise TimeFFormatError(f"{root}: recordings missing for {missing}, unlisted recordings {extra}")
    return tuple(
        Participant(pid=pid, number=files[pid][0], path=files[pid][1], age=table[pid][1], sex=table[pid][2])
        for pid in sorted(table)
    )


def check_header(path: Path) -> None:
    """Check a recording's header is exactly ``time,x,y,z,annotation``.

    Raises:
        TimeFFormatError: If it isn't.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header != list(PARTICIPANT_COLUMNS):
        raise TimeFFormatError(f"{path}: expected the header {','.join(PARTICIPANT_COLUMNS)}, got {header}")


def read_timeline(path: Path, dictionary: LabelDictionary) -> Timeline:
    """Read a recording's ``time`` and ``annotation`` columns, and split them into segments and label runs.

    An empty ``annotation`` field is an unlabelled sample. No other value is.

    Args:
        path: The recording.
        dictionary: The label dictionary, which must list every raw label the recording writes.

    Returns:
        The recording's timeline.

    Raises:
        TimeFFormatError: If the header is wrong, the file has no rows, a timestamp is missing or unreadable,
            the timestamps don't increase, or a raw label isn't in the dictionary.
    """
    check_header(path)
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
        row = int(np.flatnonzero(steps <= 0)[0]) + 1
        raise TimeFFormatError(f"{path}: the timestamp of row {row} doesn't follow the one before it")
    offsets = stamps - stamps[0]
    labels = _raw_label_codes(path, table["annotation"].combine_chunks(), dictionary)
    breaks = np.flatnonzero(steps != SAMPLE_PERIOD_US) + 1
    bounds = [0, *breaks.tolist(), len(stamps)]
    segments = tuple(
        Segment(first_index=start, n_values=end - start, offset_us=int(offsets[start]))
        for start, end in pairwise(bounds)
    )
    return Timeline(
        n_rows=len(stamps),
        start_local=_EPOCH + timedelta(microseconds=int(stamps[0])),
        offsets_us=offsets,
        labels=labels,
        segments=segments,
        label_runs=_label_runs(segments, offsets, labels),
    )


def _raw_label_codes(path: Path, column: pa.DictionaryArray, dictionary: LabelDictionary) -> np.ndarray:
    """Return each sample's raw label as an index into the dictionary, or ``NO_LABEL``.

    Returns:
        One int16 code per sample.

    Raises:
        TimeFFormatError: If the recording writes a raw label the dictionary doesn't list.
    """
    index = {label: code for code, label in enumerate(dictionary.raw_labels)}
    written = column.dictionary.to_pylist()
    unknown = sorted(label for label in written if label not in index)
    if unknown:
        raise TimeFFormatError(f"{path}: raw labels missing from the label dictionary: {unknown}")
    # One extra slot maps a null's filled index to NO_LABEL.
    codes = np.array([index[label] for label in written] + [NO_LABEL], dtype=np.int16)
    positions = column.indices.fill_null(len(written)).to_numpy()
    return codes[positions]


def _label_runs(segments: tuple[Segment, ...], offsets: np.ndarray, labels: np.ndarray) -> tuple[LabelRun, ...]:
    """Split each segment into runs of one raw label, leaving out the unlabelled runs.

    Returns:
        The runs, in time order.
    """
    runs: list[LabelRun] = []
    for number, segment in enumerate(segments):
        stop = segment.first_index + segment.n_values
        segment_labels = labels[segment.first_index : stop]
        changes = np.flatnonzero(np.diff(segment_labels)) + 1
        starts = [0, *changes.tolist()]
        ends = [*changes.tolist(), segment.n_values]
        for start, end in zip(starts, ends, strict=True):
            label = int(segment_labels[start])
            if label == NO_LABEL:
                continue
            first = segment.first_index + start
            last = segment.first_index + end - 1
            runs.append(
                LabelRun(
                    segment=number,
                    raw_label=label,
                    start_us=int(offsets[first]),
                    end_us=int(offsets[last]) + SAMPLE_PERIOD_US,
                )
            )
    return tuple(runs)
