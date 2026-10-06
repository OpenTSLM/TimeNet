"""Build records from HEARTS frames and waveforms."""

from collections.abc import Callable, Mapping
import functools
import operator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
from pydantic.dataclasses import dataclass

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.json import JsonMapping
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, frame_times_us
from timenet_connectors.datasets.yang_ai_lab.hearts_core.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import AUDIO, DESCRIPTIONS, Waveform
from timenet_connectors.time_axes import axis_for_offsets


@dataclass(frozen=True)
class FrameLayout:
    """The time columns, value specifications, and ignored columns of a frame."""

    column_specs: Mapping[str, TimeSeriesSpec]
    time_columns: tuple[str, ...]
    ignored_columns: tuple[str, ...] = ()


class Built(NamedTuple):
    """A Record and the source time its relative zero stands for."""

    record: Record
    origin_us: int


def frame_record(
    case: Case, name: str | None, frames: Mapping[tuple[str, ...], Any], layout: FrameLayout, *, subject: Any
) -> Built:
    """Build one record from frames that share a clock, one Source per frame.

    The record's zero is the earliest time any frame states. A record whose frames carry wall-clock
    time keeps that start as a local, timezone-less annotation rather than inventing a UTC anchor.

    Returns:
        The record and its source origin.

    Raises:
        TimeFFormatError: If the frames use incompatible clocks.
    """
    record_id = case.id if name is None else f"{case.id}-{name}"
    columns = {keys: _columns(frame, layout) for keys, frame in frames.items()}
    timed = {keys: frame_times_us(frame, columns[keys][0]) for keys, frame in frames.items()}
    if len({wall_clock for _, wall_clock in timed.values()}) > 1:
        raise TimeFFormatError(f"HEARTS {record_id} mixes wall-clock and relative time columns")
    origin_us = min(int(times[0]) for times, _ in timed.values())
    sources = [
        _frame_source(
            case,
            record_id,
            keys,
            timed[keys][0] - origin_us,
            {column: layout.column_specs[column] for column in columns[keys][1]},
        )
        for keys in frames
    ]
    subject, metadata = _identity(case, subject)
    record = Record(record_id=record_id, sources=tuple(sources), metadata=metadata)
    if subject is not None:
        record.annotate(subject)
    if all(wall_clock for _, wall_clock in timed.values()):
        record.annotate(
            Annotation(
                key="recording_start_local",
                value=str(np.datetime_as_string(np.datetime64(origin_us, "us"), unit="us")),
                description=DESCRIPTIONS["recording_start_local"],
            )
        )
    return Built(record, origin_us)


def _columns(frame: Any, layout: FrameLayout) -> tuple[str, list[str]]:
    """Split a frame's columns into its one time column and its value columns.

    Returns:
        The time column, and the value columns in the frame's order.

    Raises:
        TimeFFormatError: If the frame has no single time column, or a column with no spec.
    """
    time = [column for column in frame.columns if column in layout.time_columns]
    values = [
        column for column in frame.columns if column not in layout.time_columns and column not in layout.ignored_columns
    ]
    if len(time) != 1 or any(column not in layout.column_specs for column in values):
        raise TimeFFormatError(
            f"HEARTS frame with columns {list(frame.columns)} needs one time column and only known value columns"
        )
    return time[0], values


def _frame_source(
    case: Case, record_id: str, keys: tuple[str, ...], offsets: np.ndarray, specs: Mapping[str, TimeSeriesSpec]
) -> Source:
    """Build one Source with a lazy Signal for each value column of a frame.

    Returns:
        The frame's source, named after its key path in the case.
    """
    name = ".".join(keys)
    source_id = f"{record_id}-{name}"
    axis = axis_for_offsets(offsets)
    offsets_loader = (lambda: pa.array(offsets)) if isinstance(axis, IrregularAxis) else None
    signals = tuple(
        Signal.from_loader(
            spec=spec,
            name=column,
            time_axis=axis,
            loader=_column_loader(case.path, keys, column, spec.dtype),
            time_offsets_loader=offsets_loader,
            source_id=record_id,
            id=f"{source_id}-{column}",
            n_values=len(offsets),
        )
        for column, spec in specs.items()
    )
    return Source(id=source_id, name=name, signals=signals)


def _column_loader(path: Path, keys: tuple[str, ...], column: str, dtype: str) -> Callable[[], pa.Array]:
    """Build a lazy Arrow loader for one frame column.

    Returns:
        The column loader.
    """
    location = str(path)

    def load() -> pa.Array:
        frame = functools.reduce(operator.getitem, keys, load_payload(location))
        return pa.array(np.ascontiguousarray(frame[column].to_numpy(), dtype=np.dtype(dtype)))

    return load


def waveform_record(case: Case, waveform: Waveform, *, subject: Any) -> Record:
    """Build the record of a case's waveform: one audio Signal at the stated rate.

    Returns:
        The audio record.
    """
    array = case.at(waveform.keys)
    rate = waveform.rate if isinstance(waveform.rate, int) else int(case.at(waveform.rate))
    name = ".".join(waveform.keys)
    source_id = f"{case.id}-{name}"
    signal = Signal.from_loader(
        spec=AUDIO,
        name="audio",
        time_axis=RegularAxis.from_rate_hz(rate),
        loader=_array_loader(case.path, waveform.keys),
        source_id=case.id,
        id=f"{source_id}-audio",
        n_values=len(array),
    )
    subject, metadata = _identity(case, subject)
    record = Record(
        record_id=case.id,
        sources=(Source(id=source_id, name=name, signals=(signal,)),),
        metadata=metadata,
    )
    if subject is not None:
        record.annotate(subject)
    return record


def _array_loader(path: Path, keys: tuple[str, ...]) -> Callable[[], pa.Array]:
    """Build a lazy Arrow float32 loader for a waveform.

    Returns:
        The waveform loader.
    """
    location = str(path)

    def load() -> pa.Array:
        array = functools.reduce(operator.getitem, keys, load_payload(location))
        return pa.array(np.ascontiguousarray(array, dtype=np.float32))

    return load


def _identity(case: Case, subject: Any) -> tuple[Annotation | None, JsonMapping]:
    """Build a corpus-qualified subject annotation and record metadata.

    Returns:
        The subject annotation and metadata.
    """
    annotation = (
        None
        if subject is None
        else Annotation(
            key="subject_id",
            value=f"{case.corpus}:{subject}",
            description=DESCRIPTIONS["subject_id"],
        )
    )
    return annotation, {"corpus": case.corpus}
