"""Group the published windows into distinct measurements, and build one Record per measurement.

The authors publish one window per NOAA event, so windows of different events can hold the same
measurements. ``WindowIndex`` reads the window files once (pass one) and keeps each distinct
measurement once, with every window that shares it. ``WindowReader`` rereads a measurement's values
only when the dataset is written (pass two), and checks they did not change.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import TimeInterval
from timenet_connectors.datasets.trace.weather_events.archive import TraceFile
from timenet_connectors.datasets.trace.weather_events.fields import as_object, field
from timenet_connectors.datasets.trace.weather_events.vocabulary import (
    FORECAST_HORIZON,
    MIN_TASK_VALUES,
    Channel,
    EventLabel,
    MetadataKey,
    Sampling,
    SourceField,
)
from timenet_connectors.datasets.trace.weather_events.window_rows import (
    Coordinates,
    County,
    Readings,
    WindowRow,
    parse_sampling,
)
from timenet_connectors.sources.json_stream import iter_object, read_entry
from timenet_connectors.time_axes import axis_for_offsets


_STATION_SOURCE = "station"
"""The name of each Record's one Source. It is generic: the station id is a strong prior on the event type."""


def _key_context(path: Path, key: str) -> str:
    """Return the file and key that error messages name for a window."""
    return f"{path} (key {key})"


def _is_canonical_window_id(key: str) -> bool:
    """Check that a JSON key reads exactly as the integer id a retrieval row gives, such as 12, not 012.

    Returns:
        Whether the key is canonical.
    """
    return key.isascii() and key.isdecimal() and key == str(int(key))


class WindowLocation(NamedTuple):
    """Where a window's values sit in its JSON file, and the digest they had at conversion."""

    path: Path
    offset: int
    length: int
    digest: bytes

    @property
    def error_context(self) -> str:
        """The file and byte range that error messages name."""
        return f"{self.path} (bytes {self.offset}-{self.offset + self.length})"


class WindowReader:
    """Load a Record's values lazily, one decoded window at a time.

    Reading one Record's channels in order hits the cached window. The writer sorts series by spec type,
    so it returns to each window once per channel and rereads it: about seven rereads per window. Each
    reread must give back the series that conversion saw: the digest covers the timestamps and every value.
    """

    def __init__(self) -> None:
        self._readings = lru_cache(maxsize=1)(self._read)

    @staticmethod
    def _read(location: WindowLocation) -> Readings:
        raw = read_entry(location.path, location.offset, location.length)
        try:
            raw = as_object(raw)
            sampling = parse_sampling(raw)
            readings = Readings.parse(raw, sampling)
            digest = readings.digest(field(raw, SourceField.STATION_ID), sampling.value)
        except TimeFValidationError as exc:
            raise TimeFFormatError(f"{location.error_context}: the window changed since conversion: {exc}") from exc
        if digest != location.digest:
            raise TimeFFormatError(
                f"{location.error_context}: the window differs from the one converted; the file changed"
            )
        return readings

    def values(self, location: WindowLocation, channel: Channel) -> pa.Array:
        """Return one channel of a window, as the channel stores it."""
        return channel.to_arrow(self._readings(location).values[channel])

    def time_offsets(self, location: WindowLocation) -> pa.Array:
        """Return a window's local-clock time offsets in microseconds, for a Record whose axis has gaps."""
        return pa.array(self._readings(location).timestamps.offsets_us)


@dataclass(eq=False)
class Measurement:
    """One distinct set of measurements: the values that several windows can share, and one Record.

    The values stay in the file of the first window seen with them. ``eq=False``: a window points to its
    measurement, which lists its windows, so a field-by-field comparison would recurse between them.
    """

    station_id: str
    coordinates: Coordinates
    sampling: Sampling
    first_date: str
    last_date: str
    n_values: int
    axis: RegularAxis | IrregularAxis
    """The time axis the seven channels share: regular at the cadence if the window has no gaps."""
    location: WindowLocation
    windows: list["Window"]

    @classmethod
    def first_seen_in(cls, row: WindowRow, location: WindowLocation) -> "Measurement":
        """Return the measurement of a window not seen before, with no windows yet.

        Returns:
            The measurement, whose values stay where ``location`` points.
        """
        timestamps = row.readings.timestamps
        return cls(
            station_id=row.station_id,
            coordinates=row.coordinates,
            sampling=row.sampling,
            first_date=timestamps.first,
            last_date=timestamps.last,
            n_values=row.readings.n_values,
            axis=axis_for_offsets(timestamps.offsets_us, period_us=Fraction(row.sampling.cadence_us)),
            location=location,
            windows=[],
        )

    @property
    def is_long_enough(self) -> bool:
        """Whether the authors give a window of this length a task."""
        return self.n_values >= MIN_TASK_VALUES

    def metadata(self) -> dict[str, JsonValue]:
        """Return the Record's metadata: its station, where it stands, its sampling and every window."""
        windows = sorted(self.windows, key=lambda window: window.window_id)
        return {
            MetadataKey.STATION: self.station_id,
            MetadataKey.COORDINATES: self.coordinates._asdict(),
            MetadataKey.SAMPLING: self.sampling.value,
            MetadataKey.WINDOW_START: self.first_date,
            MetadataKey.WINDOW_END: self.last_date,
            MetadataKey.WINDOWS: [window.metadata() for window in windows],
        }

    def to_record(self, reader: WindowReader) -> Record:
        """Build the measurement's seven-channel station Record, whose values stay in the source file.

        The Record has no start time: the timestamps are local clock time in an unstated zone, so it has no
        absolute anchor. The published first and last timestamps stay in its metadata.

        Returns:
            A Record with lazy signals on one shared time axis.
        """
        location, axis = self.location, self.axis
        offsets_loader = partial(reader.time_offsets, location) if isinstance(axis, IrregularAxis) else None
        signals = tuple(
            Signal.from_loader(
                name=channel.value,
                spec=channel.spec,
                time_axis=axis,
                time_offsets_loader=offsets_loader,
                n_values=self.n_values,
                loader=partial(reader.values, location, channel),
            )
            for channel in Channel
        )
        source = Source(
            name=_STATION_SOURCE,
            signals=signals,
            # TimeF does not keep signal order on read, so record the source's channel order.
            metadata={MetadataKey.CHANNEL_ORDER: [channel.value for channel in Channel]},
        )
        return Record(sources=(source,), metadata=self.metadata())


@dataclass(frozen=True, eq=False)
class Window:
    """One window as the authors number it: one station and sampling, before an event or not.

    Copies of one measurement share their measurement, and so their Record, but each keeps its own id,
    label, county, split and tasks.
    """

    file: TraceFile
    key: str
    measurement: Measurement
    county: County
    label: EventLabel | None
    forecast_target: TimeInterval | None
    """The span of a forecasting window's last ``FORECAST_HORIZON`` values, in offsets from its first."""

    @classmethod
    def from_row(cls, file: TraceFile, key: str, row: WindowRow, measurement: Measurement) -> "Window":
        """Return a checked window of ``file``, with its forecast target if it gets a forecasting task.

        Returns:
            The window. It is not yet in its measurement's windows.
        """
        gets_forecast = file.kind.holds_forecasts and measurement.is_long_enough
        target = row.readings.timestamps.span_of_last(FORECAST_HORIZON) if gets_forecast else None
        return cls(file, key, measurement, row.county, row.label, target)

    @property
    def window_id(self) -> int:
        """The window's id, as the authors number it."""
        return int(self.key)

    @property
    def error_context(self) -> str:
        """The file and key that error messages name."""
        return _key_context(self.file.path, self.key)

    @property
    def is_classified(self) -> bool:
        """Whether the authors classify the window: it names an event and is long enough."""
        return self.label is not None and self.label.is_event and self.measurement.is_long_enough

    def provenance(self) -> dict[str, JsonValue]:
        """Return the window's published identity: its id, anchor and file."""
        return {
            MetadataKey.WINDOW_ID: self.window_id,
            MetadataKey.ANCHOR: Sampling.anchor_of(self.window_id),
            MetadataKey.SOURCE_FILE: self.file.archive_path,
        }

    def metadata(self) -> dict[str, JsonValue]:
        """Return the window as its Record's metadata lists it: its identity, county and stated label."""
        return self.provenance() | {
            MetadataKey.COUNTY: self.county._asdict(),
            MetadataKey.EVENT_TYPE: None if self.label is None else self.label.value,
        }


class WindowIndex:
    """Pass one over the window files: every window by its key, and the measurements they share."""

    def __init__(self) -> None:
        self.windows: dict[str, Window] = {}
        self.measurements: dict[bytes, Measurement] = {}
        """Every distinct measurement, by its digest, in the order its first window was read."""

    def add_file(self, file: TraceFile) -> None:
        """Check every window of one file and group it with the windows that share its measurements.

        Raises:
            TimeFValidationError: If a window is invalid, naming the file and key, or the file has no windows.
        """
        n_windows = 0
        for key, published, offset, length in iter_object(file.path):
            try:
                self._add(file, key, published, offset, length)
            except TimeFValidationError as exc:
                raise TimeFValidationError(f"{_key_context(file.path, key)}: {exc}") from exc
            n_windows += 1
        # A truncated file can still be valid JSON. Every split file of the archive holds windows.
        if n_windows == 0:
            raise TimeFValidationError(f"{file.path}: holds no windows")

    def get(self, window_id: int) -> Window | None:
        """Return the window with this id, or ``None`` if no window file has it."""
        return self.windows.get(str(window_id))

    def _add(self, file: TraceFile, key: str, published: object, offset: int, length: int) -> None:
        if not _is_canonical_window_id(key):
            raise TimeFValidationError(f"expected a canonical decimal window id, got {key!r}")
        if key in self.windows:
            raise TimeFValidationError(f"window {key} also appears in {self.windows[key].file.path}")
        row = WindowRow.parse(published, window_id=int(key), kind=file.kind)
        digest = row.digest()
        measurement = self.measurements.get(digest)
        if measurement is None:
            measurement = Measurement.first_seen_in(row, WindowLocation(file.path, offset, length, digest))
            self.measurements[digest] = measurement
        elif row.coordinates != measurement.coordinates:
            raise TimeFValidationError(
                f"same measurements as window {measurement.windows[0].key}, but at "
                f"{row.coordinates} instead of {measurement.coordinates}"
            )
        window = Window.from_row(file, key, row, measurement)
        measurement.windows.append(window)
        self.windows[key] = window
