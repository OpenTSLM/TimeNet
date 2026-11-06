"""Read and validate published windows, index their copies, and lazily load measurements.

The authors publish one window per NOAA event, so windows of different events can hold the same
measurements. ``WindowIndex`` reads the window files once (pass one) and groups the copies under one
measurement. Each window still gets its own Record, as the authors count it, and names the window it
copies. ``WindowReader`` rereads the values only when the dataset is written (pass two), and checks
they did not change.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types import TimeInterval
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.datasets.trace.weather_events.archive import TraceFile
from timenet_connectors.datasets.trace.weather_events.vocabulary import (
    FORECAST_HORIZON,
    MIN_TASK_VALUES,
    Channel,
    EventLabel,
    MetadataKey,
    Sampling,
)
from timenet_connectors.datasets.trace.weather_events.window_rows import (
    Coordinates,
    County,
    Readings,
    WindowRow,
    read_window_fields,
)
from timenet_connectors.sources.errors import error_context
from timenet_connectors.sources.json_stream import iter_object, read_entry
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.time_axes import axis_for_offsets


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
        self.shared = SharedChannelLoader(self._read_channels)

    @staticmethod
    def _read(location: WindowLocation) -> Readings:
        raw = read_entry(location.path, location.offset, location.length)
        with error_context(location.error_context):
            raw = read_window_fields(raw)
            return checked_reread(
                lambda: Readings.parse(raw, raw.mode),
                fingerprint=lambda readings: readings.digest(raw.station_id, raw.mode.value),
                expected=location.digest,
                context="measurements",
            )

    def _read_channels(self, location: WindowLocation) -> tuple[pa.Array, ...]:
        readings = self._read(location)
        return (
            *(channel.to_arrow(readings.values[channel]) for channel in Channel),
            pa.array(readings.timestamps.offsets_us),
        )

    def time_offsets(self, location: WindowLocation) -> pa.Array:
        """Return local-clock offsets from the same cached read as the seven channels."""
        return self.shared.channel_loader(location, len(Channel))()


@dataclass(eq=False)
class Measurement:
    """One distinct set of measurements: the values that several windows can share.

    Values stay in the first source window. The index owns the windows; this descriptor retains
    only their first source key and lowest ID. Equality uses identity for shared measurements.
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
    first_source_key: str
    first_window_id: int

    @classmethod
    def first_seen_in(cls, row: WindowRow, location: WindowLocation, key: str) -> "Measurement":
        """Return the measurement of a window not seen before, retaining its source identity.

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
            first_source_key=key,
            first_window_id=int(key),
        )

    @property
    def is_long_enough(self) -> bool:
        """Whether the authors give a window of this length a task."""
        return self.n_values >= MIN_TASK_VALUES

    def metadata(self) -> dict[str, JsonValue]:
        """Return the metadata that every copy's Record shares: the station, its place, sampling and dates."""
        return {
            MetadataKey.STATION: self.station_id,
            MetadataKey.COORDINATES: self.coordinates._asdict(),
            MetadataKey.SAMPLING: self.sampling.value,
            MetadataKey.WINDOW_START: self.first_date,
            MetadataKey.WINDOW_END: self.last_date,
        }


@dataclass(frozen=True, eq=False)
class Window:
    """One window as the authors number it: one station and sampling, before an event or not.

    Copies of one measurement share their measurement, but each gets its own Record, as the authors
    count it, with its own id, label, county, split and tasks.
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
            The window. Its measurement is shared with any copies.
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

    @property
    def copy_of(self) -> int | None:
        """The lowest id of the other windows with the same measurements, or ``None`` if the window has none.

        The window with the lowest id is the original, so its own value is also ``None``.
        """
        first = self.measurement.first_window_id
        return None if first == self.window_id else first

    def metadata(self) -> dict[str, JsonValue]:
        """Return the Record's metadata: the measurement's, the window's identity, county, label and original."""
        return (
            self.measurement.metadata()
            | self.provenance()
            | {
                MetadataKey.COUNTY: self.county._asdict(),
                MetadataKey.EVENT_TYPE: None if self.label is None else self.label.value,
                MetadataKey.COPY_OF: self.copy_of,
            }
        )


class WindowIndex:
    """Pass one over the window files: every window by its key, and the measurements that copies share."""

    def __init__(self) -> None:
        self.windows: dict[str, Window] = {}
        self.measurements: dict[bytes, Measurement] = {}
        """Every distinct measurement, by its digest, in the order its first window was read."""

    def add_file(self, file: TraceFile) -> None:
        """Check every window of one file and group it with the windows that share its measurements.

        Raises:
            TimeNetConnectorError: If a window is invalid, naming the file and key, or the file has no windows.
        """
        n_windows = 0
        for entry in iter_object(file.path):
            with error_context(_key_context(file.path, entry.key)):
                self._add(file, entry.key, entry.value, entry.offset, entry.length)
            n_windows += 1
        # A truncated file can still be valid JSON. Every split file of the archive holds windows.
        if n_windows == 0:
            raise TimeNetConnectorError(f"{file.path}: holds no windows")

    def get(self, window_id: int) -> Window | None:
        """Return the window with this id, or ``None`` if no window file has it."""
        return self.windows.get(str(window_id))

    def _add(self, file: TraceFile, key: str, published: object, offset: int, length: int) -> None:
        if not _is_canonical_window_id(key):
            raise TimeNetConnectorError(f"expected a canonical decimal window id, got {key!r}")
        if key in self.windows:
            raise TimeNetConnectorError(f"window {key} also appears in {self.windows[key].file.path}")
        row = WindowRow.parse(published, window_id=int(key), kind=file.kind)
        digest = row.digest()
        measurement = self.measurements.get(digest)
        if measurement is None:
            measurement = Measurement.first_seen_in(row, WindowLocation(file.path, offset, length, digest), key)
            self.measurements[digest] = measurement
        elif row.coordinates != measurement.coordinates:
            raise TimeNetConnectorError(
                f"same measurements as window {measurement.first_source_key}, but at "
                f"{row.coordinates} instead of {measurement.coordinates}"
            )
        window = Window.from_row(file, key, row, measurement)
        measurement.first_window_id = min(measurement.first_window_id, window.window_id)
        self.windows[key] = window
