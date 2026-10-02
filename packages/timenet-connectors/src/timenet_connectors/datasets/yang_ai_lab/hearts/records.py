"""Turn the frames, waveforms, and photographs of a HEARTS case into TimeF Records."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import functools
import operator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.release import (
    AUDIO,
    CGM,
    COLUMN_SPECS,
    DERIVED_COLUMNS,
    DESCRIPTIONS,
    PHOTOGRAPHS,
    TIME_COLUMNS,
    US_PER_MINUTE,
    TaskDef,
    Waveform,
)
from timenet_connectors.sources.images import image_signal
from timenet_connectors.time_axes import axis_for_offsets


@dataclass(frozen=True)
class Case:
    """One test case: its task directory and definition, its index, its file, and its loaded payload."""

    directory: str
    definition: TaskDef
    index: int
    path: Path
    payload: dict[str, Any]

    @property
    def corpus(self) -> str:
        """Return the corpus directory of the case, such as ``cgmacros``."""
        return self.directory.partition("/")[0]

    @property
    def task(self) -> str:
        """Return the task directory inside the corpus, such as ``meal_forecasting``."""
        return self.directory.partition("/")[2]

    @property
    def id(self) -> str:
        """Return the stable id the case's records and task are named after."""
        return f"hearts-{self.corpus}-{self.task}-{self.index:02d}"

    def at(self, keys: tuple[str, ...]) -> Any:
        """Return the value at a key path of the payload."""
        return functools.reduce(operator.getitem, keys, self.payload)


class Built(NamedTuple):
    """A Record and the source time its relative zero stands for."""

    record: Record
    origin_us: int


def input_records(case: Case, entries: tuple[str | tuple[str, ...] | Waveform, ...]) -> list[Built]:
    """Build the records the entries name, in order.

    A frame key gives one record, a tuple of frame keys one record whose frames share a clock, the
    key of a dictionary of frames one record per entry in key order, and a waveform one record on
    its own sampling clock.

    Returns:
        The records with their source origins.
    """
    built = []
    for entry in entries:
        if isinstance(entry, Waveform):
            built.append(Built(_waveform_record(case, entry), 0))
        elif isinstance(entry, tuple):
            built.append(_frame_record(case, None, {(key,): case.payload[key] for key in entry}))
        elif isinstance(case.payload[entry], dict):
            frames = sorted(case.payload[entry].items())
            built.extend(_frame_record(case, name, {(entry, name): frame}) for name, frame in frames)
        else:
            built.append(_frame_record(case, entry if len(entries) > 1 else None, {(entry,): case.payload[entry]}))
    return built


def frame_times_us(frame: Any, column: str) -> tuple[np.ndarray, bool]:
    """Read a frame's time column as whole microseconds.

    Returns:
        One microsecond value per row, and whether they are naive wall-clock datetimes.

    Raises:
        TimeFFormatError: If the time column is empty or contains a missing time.
    """
    values = frame[column].to_numpy()
    if len(values) == 0:
        raise TimeFFormatError(f"HEARTS frame time column {column!r} has no values")
    if values.dtype.kind == "O":  # CGMacros writes its timestamps as text
        values = values.astype("datetime64[us]")
    if values.dtype.kind in "Mm":
        if bool(np.isnat(values).any()):
            raise TimeFFormatError(f"HEARTS frame time column {column!r} has missing values")
        unit = "datetime64[us]" if values.dtype.kind == "M" else "timedelta64[us]"
        return values.astype(unit).astype(np.int64), values.dtype.kind == "M"
    return np.rint(values * US_PER_MINUTE).astype(np.int64), False  # iAUC frames count minutes since the meal


def answer_record(case: Case, built: Built, offsets: np.ndarray, values: np.ndarray) -> Record:
    """Build the Record holding the glucose readings a forecast or imputation asks for, on the input's clock.

    Returns:
        The answer record, sharing the input's origin, subject, and local start time.
    """
    record_id = f"{case.id}-GT"
    ids = {"id": f"{record_id}-Libre GL", "source_id": record_id}
    axis = axis_for_offsets(offsets)
    if isinstance(axis, IrregularAxis):
        signal = Signal.from_irregular(values, time_offsets_us=offsets, spec=CGM, name="Libre GL", **ids)
    else:
        signal = Signal.from_values(values, spec=CGM, name="Libre GL", time_axis=axis, **ids)
    source = Source(id=f"{record_id}-source", name="HEARTS held-out answer", signals=(signal,))
    record = Record(
        record_id=record_id,
        start_time=built.record.start_time,
        sources=(source,),
        metadata=dict(built.record.metadata),
    )
    record.add_annotations(
        annotation
        for annotation in built.record.annotations
        if annotation.key in {"recording_start_local", "subject_id"}
    )
    return record


def _frame_record(case: Case, name: str | None, frames: Mapping[tuple[str, ...], Any]) -> Built:
    """Build one record from frames that share a clock, one Source per frame.

    The record's zero is the earliest time any frame states. A record whose frames carry wall-clock
    time keeps that start as a local, timezone-less annotation rather than inventing a UTC anchor.

    Returns:
        The record and its source origin.

    Raises:
        TimeFFormatError: If the frames use incompatible clocks.
    """
    record_id = case.id if name is None else f"{case.id}-{name}"
    timed = {keys: frame_times_us(frame, _columns(frame)[0]) for keys, frame in frames.items()}
    if len({wall_clock for _, wall_clock in timed.values()}) > 1:
        raise TimeFFormatError(f"HEARTS {record_id} mixes wall-clock and relative time columns")
    origin_us = min(int(times[0]) for times, _ in timed.values())
    sources = [
        _frame_source(case, record_id, keys, frame, timed[keys][0] - origin_us) for keys, frame in frames.items()
    ]
    if case.definition.images and name is None:
        sources.append(_images_source(case, record_id))
    subject, metadata = _identity(case, name)
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


def _columns(frame: Any) -> tuple[str, list[str]]:
    """Split a frame's columns into its one time column and its value columns.

    Returns:
        The time column, and the value columns in the frame's order.

    Raises:
        TimeFFormatError: If the frame has no single time column, or a column with no spec.
    """
    time = [column for column in frame.columns if column in TIME_COLUMNS]
    values = [column for column in frame.columns if column not in TIME_COLUMNS and column not in DERIVED_COLUMNS]
    if len(time) != 1 or any(column not in COLUMN_SPECS for column in values):
        raise TimeFFormatError(
            f"HEARTS frame with columns {list(frame.columns)} needs one time column and only known value columns"
        )
    return time[0], values


def _frame_source(case: Case, record_id: str, keys: tuple[str, ...], frame: Any, offsets: np.ndarray) -> Source:
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
            spec=COLUMN_SPECS[column],
            name=column,
            time_axis=axis,
            loader=_column_loader(case.path, keys, column, COLUMN_SPECS[column].dtype),
            time_offsets_loader=offsets_loader,
            source_id=record_id,
            id=f"{source_id}-{column}",
            n_values=len(offsets),
        )
        for column in _columns(frame)[1]
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


def _waveform_record(case: Case, waveform: Waveform) -> Record:
    """Build the record of a case's waveform: one audio Signal at the stated rate.

    Returns:
        The record, with the annotators' quality rating when the corpus has one.
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
    subject, metadata = _identity(case, None)
    record = Record(
        record_id=case.id,
        sources=(Source(id=source_id, name=name, signals=(signal,)),),
        metadata=metadata,
    )
    if subject is not None:
        record.annotate(subject)
    if waveform.quality is not None:
        record.annotate(
            Annotation(
                key="audio_quality", value=int(case.at(waveform.quality)), description=DESCRIPTIONS["audio_quality"]
            )
        )
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


def _images_source(case: Case, record_id: str) -> Source:
    """Build the Source holding the four meal photographs as image Signals.

    Returns:
        The image source.
    """
    mapping = case.payload["image_mapping"]
    source_id = f"{record_id}-image_mapping"
    signals = tuple(image_signal(mapping[name], signal_id=f"{source_id}-{name}", name=name) for name in PHOTOGRAPHS)
    return Source(id=source_id, name="image_mapping", signals=signals)


def _identity(case: Case, name: str | None) -> tuple[Annotation | None, dict[str, object]]:
    """Return a record's subject annotation and metadata: its corpus and any recording id.

    A compared window names its subject under its own key. Every other case names one subject, or
    a speaker for VCTK. The corpus-qualified annotation persists the subject identity.

    Returns:
        The subject annotation, and the metadata.
    """
    payload = case.payload
    subject = payload.get(f"{name}_subject") if name is not None else None
    if subject is None:
        subject = payload.get("subject_id", payload.get("speaker_id"))
    metadata: dict[str, object] = {"corpus": case.corpus}
    if "recording_id" in payload:
        metadata["recording_id"] = str(payload["recording_id"])
    annotation = (
        None
        if subject is None
        else Annotation(
            key="subject_id",
            value=f"{case.corpus}:{subject}",
            description=DESCRIPTIONS["subject_id"],
        )
    )
    return annotation, metadata
