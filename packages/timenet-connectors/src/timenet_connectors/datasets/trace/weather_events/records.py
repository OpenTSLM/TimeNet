"""Group TRACE's windows into Records of distinct measurements, whose values load lazily from the JSON files."""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache, partial
import hashlib
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimeInterval
from timenet_connectors.datasets.trace.weather_events.source import (
    CADENCE_US,
    CHANNEL_NAMES,
    CHANNELS,
    FORECAST_HORIZON,
    MIN_TASK_VALUES,
    MODES,
    NO_EVENT,
    SKY_COVER_LABELS,
    EventType,
    Mode,
    PretrainRow,
    TraceFileRef,
    WindowRow,
    check_inventory,
)
from timenet_connectors.sources.json_stream import iter_object, read_entry
from timenet_connectors.time_axes import axis_for_offsets


@dataclass(frozen=True)
class WindowRef:
    """Where one published window sits in its file."""

    file: TraceFileRef
    key: str
    offset: int
    length: int

    @property
    def window_id(self) -> int:
        """The window's id, as the authors number it."""
        return int(self.key)

    @property
    def context(self) -> str:
        """The file and key that error messages name."""
        return f"{self.file.path} (key {self.key})"

    @property
    def provenance(self) -> dict[str, Any]:
        """The window's published identity: its id, anchor, and file."""
        return {
            "window_id": self.window_id,
            "anchor": self.window_id // len(MODES),
            "source_file": self.file.archive_path,
        }

    @property
    def task_suffix(self) -> str:
        """An id part for the window's tasks. The authors' ids are sorted by event type, so hash them."""
        return hashlib.sha256(self.key.encode()).hexdigest()[:16]


def _clock_offsets_us(dates: list[str]) -> np.ndarray:
    """Return each naive ISO timestamp's offset from the first on the station's local clock.

    The source states no time zone, so a timestamp is not a moment in Unix time. Only the difference
    between two timestamps means anything: the local-clock time between them. Across a daylight
    saving change, that differs from the elapsed time by an hour.

    Returns:
        One int64 microsecond offset per date, starting at 0.
    """
    readings = np.array(dates, dtype="datetime64[us]").astype(np.int64)
    return readings - readings[0]


def _time_offsets(dates: list[str], mode: Mode) -> np.ndarray:
    """Read naive ISO timestamps on the local clock and check they sit on the mode's cadence.

    Returns:
        Every timestamp's local-clock offset from the first, in microseconds.

    Raises:
        TimeFValidationError: If a timestamp is not a calendar date, out of order, or off the cadence.
    """
    try:
        offsets = _clock_offsets_us(dates)
    except ValueError as exc:
        # The row model fixed the format, so only an impossible date such as 02-30 reaches here.
        raise TimeFValidationError(f"DATE: {exc}") from exc
    steps = np.diff(offsets)
    if (steps <= 0).any() or (steps % CADENCE_US[mode]).any():
        raise TimeFValidationError(f"DATE must increase on the {mode} cadence; gaps are allowed")
    return offsets


def _parse_window(row: Any, ref: WindowRef) -> tuple[WindowRow, np.ndarray]:
    """Validate one window and decode its timing, naming the source location on error.

    Returns:
        The validated row and its time offsets.

    Raises:
        TimeFValidationError: If the row is malformed, with file and key context.
    """
    try:
        # Pretrain rows must carry an event label, and forecasting rows must not.
        window = (PretrainRow if ref.file.task == "pretrain" else WindowRow).model_validate(row)
        if window.mode != MODES[ref.window_id % len(MODES)]:
            raise TimeFValidationError(f"mode {window.mode.value!r} does not match the id's anchor position")
        offsets = _time_offsets(window.DATE, window.mode)
        for name in CHANNEL_NAMES:
            values = getattr(window, name)
            if len(values) != len(offsets):
                raise TimeFValidationError(f"{name} has {len(values)} values for {len(offsets)} timestamps")
    # A pydantic ValidationError is a ValueError, so it gains the same context.
    except (ValueError, OverflowError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc
    return window, offsets


def _read_window(ref: WindowRef) -> dict[str, Any]:
    return read_entry(ref.file.path, ref.offset, ref.length)


class WindowLoader:
    """Load values lazily, reusing one decoded window across its channels and time offsets.

    ``convert`` validated each window's byte range when it indexed the extracted file, and a
    build reads the same file, so a reread is not validated again. This assumes the extracted
    files do not change between ``convert`` and writing.
    """

    def __init__(self) -> None:
        # A window's channels and time offsets load one after another, so one cached window suffices.
        self._row = lru_cache(maxsize=1)(_read_window)

    def values(self, ref: WindowRef, name: str) -> pa.Array:
        """Return one channel of a window: float32 values, or ``sky_code`` as codebook indices.

        Returns:
            The channel's values as an Arrow array.
        """
        values = self._row(ref)[name]
        if name == "sky_code":
            # The published codes are already indices into the codebook.
            return pa.DictionaryArray.from_arrays(pa.array(values, type=pa.int32()), SKY_COVER_LABELS)
        return pa.array(values, type=pa.float32())

    def time_offsets(self, ref: WindowRef) -> pa.Array:
        """Return a window's local-clock time offsets, for a Record whose axis has gaps.

        Returns:
            One int64 microsecond offset per value, starting at 0.
        """
        return pa.array(_clock_offsets_us(self._row(ref)["DATE"]), type=pa.int64())


class _PublishedWindow(NamedTuple):
    """One published window of a measurement, as the Record's metadata lists it."""

    ref: WindowRef
    event_type: int | None
    """The published label, ``NO_EVENT`` included, or ``None`` outside ``pretrain``."""
    location: str
    """The window's location: an event window names the county of its event."""


@dataclass
class Measurement:
    """One distinct measurement window, shared by every published window that holds it."""

    record_id: str
    first: WindowRef
    """The first window seen with these measurements; the Record's values load from it."""
    row: WindowRow
    axis: RegularAxis | IrregularAxis
    n_values: int
    windows: list[_PublishedWindow] = field(default_factory=list)
    """Every published window with these measurements."""
    record: Record | None = None


@dataclass(frozen=True)
class IndexedWindow:
    """What the task stream needs of one published window."""

    ref: WindowRef
    measurement: Measurement
    event_type: EventType | None
    """The window's label if the authors classify it, else ``None``."""
    forecast: TimeInterval | None
    """The span of the window's last ``FORECAST_HORIZON`` values, if it gets a forecasting task,
    else ``None``."""

    @property
    def record(self) -> Record:
        """The Record that holds the window's measurements.

        Raises:
            TimeFValidationError: If the task stream asks for it before ``convert`` built it.
        """
        record = self.measurement.record
        if record is None:
            raise TimeFValidationError(f"{self.ref.context}: the window's Record was never built")
        return record


def index_windows(file: TraceFileRef, measurements: dict[str, Measurement], windows: dict[str, IndexedWindow]) -> None:
    """Index one window file: group its windows by measurements, and note each window's tasks.

    Raises:
        TimeFValidationError: If a window id, a window, the coordinates of a copy, or the file's pinned
            inventory is invalid.
    """
    rows = tasks = 0
    for key, row, offset, length in iter_object(file.path):
        # Retrieval rows give integer ids, so a key must read exactly as str(id), such as 12, not 012.
        if not key.isascii() or not key.isdecimal() or key != str(int(key)):
            raise TimeFValidationError(f"{file.path}: expected a canonical decimal window id, got {key!r}")
        if key in windows:
            raise TimeFValidationError(f"{file.path}: window {key} also appears in {windows[key].ref.file.path}")
        ref = WindowRef(file, key, offset, length)
        window, offsets = _parse_window(row, ref)
        digest = window.measurement_key()
        measurement = measurements.get(digest)
        if measurement is None:
            # One axis per window: its seven channels share every timestamp. A gap-free window is
            # regular at its mode's cadence; a window with missing timestamps keeps explicit offsets.
            axis = axis_for_offsets(offsets, period_us=Fraction(CADENCE_US[window.mode]))
            measurement = Measurement(f"trace-{digest[:16]}", ref, window, axis, len(offsets))
            measurements[digest] = measurement
        elif (window.latitude, window.longitude) != (measurement.row.latitude, measurement.row.longitude):
            raise TimeFValidationError(
                f"{ref.context}: same measurements as window {measurement.first.key}, but at "
                f"({window.latitude}, {window.longitude}) instead of "
                f"({measurement.row.latitude}, {measurement.row.longitude})"
            )
        published = window.event_type if isinstance(window, PretrainRow) else None
        measurement.windows.append(_PublishedWindow(ref, published, window.place))
        # The authors classify and forecast only windows with at least MIN_TASK_VALUES values.
        long_enough = len(offsets) >= MIN_TASK_VALUES
        event_type = EventType(published) if long_enough and published not in {None, NO_EVENT} else None
        forecast = None
        if file.task == "forecasting" and long_enough:
            # The last FORECAST_HORIZON values, however much time they span in a window with gaps.
            forecast = TimeInterval(start_us=int(offsets[-FORECAST_HORIZON]), end_us=int(offsets[-1]) + 1)
        windows[key] = IndexedWindow(ref, measurement, event_type, forecast)
        rows += 1
        tasks += event_type is not None or forecast is not None
    check_inventory(file, rows, tasks)


def make_record(measurement: Measurement, loader: WindowLoader) -> Record:
    """Build one seven-channel station Record whose values stay in the source file.

    Returns:
        A Record with lazy signals on one shared time axis, its dates as annotations, and its station
        and published windows as metadata.
    """
    row, first, record_id = measurement.row, measurement.first, measurement.record_id
    axis = measurement.axis
    offsets_loader = partial(loader.time_offsets, first) if isinstance(axis, IrregularAxis) else None
    signals = tuple(
        Signal.from_loader(
            id=f"{record_id}-{name.replace('_', '-')}",
            name=name,
            spec=spec,
            time_axis=axis,
            time_offsets_loader=offsets_loader,
            n_values=measurement.n_values,
            loader=partial(loader.values, first, name),
        )
        for name, spec in CHANNELS
    )
    # No start_time: the timestamps are local clock time in an unstated zone, so the Record has no
    # absolute anchor. The published first and last timestamps stay in the annotations below.
    record = Record(
        record_id=record_id,
        sources=(
            Source(
                id=f"{record_id}-station",
                # A generic name: the station id is a strong prior on the event type, so it stays in
                # metadata, which is never model input.
                name="station",
                signals=signals,
                # TimeF does not keep signal order on read, so record the source's channel order.
                metadata={"channel_order": list(CHANNEL_NAMES)},
            ),
        ),
        metadata={
            "station_id": row.station_id,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "mode": row.mode.value,
            # Every published window with these measurements, with its location and published event
            # label. An event window's location is the county of its event, so it stays here.
            "windows": [
                window.ref.provenance
                | {"location": window.location}
                | ({} if window.event_type is None else {"event_type": window.event_type})
                for window in sorted(measurement.windows, key=lambda window: window.ref.window_id)
            ],
        },
    )
    for key, value, description in (
        ("window_start", row.DATE[0], "First timestamp, as published: station local clock time, no time zone"),
        ("window_end", row.DATE[-1], "Last timestamp, as published: station local clock time, no time zone"),
    ):
        record.annotate(
            Annotation(key=key, value=value, description=description, id=f"{record_id}-{key.replace('_', '-')}")
        )
    measurement.record = record
    return record
