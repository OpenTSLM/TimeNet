"""One distinct measurement and its Record, and the published windows that share it."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property, lru_cache, partial
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimeInterval
from timenet_connectors.datasets.trace.weather_events.archive import TraceTaskKind
from timenet_connectors.datasets.trace.weather_events.metadata import MetadataExtension, RecordMetadata
from timenet_connectors.datasets.trace.weather_events.retrieval import EventReport
from timenet_connectors.datasets.trace.weather_events.vocabulary import (
    FORECAST_HORIZON,
    MIN_TASK_VALUES,
    Channel,
    EventLabel,
    SkyCover,
    WindowSampling,
)
from timenet_connectors.datasets.trace.weather_events.windows import (
    Coordinates,
    County,
    Station,
    WindowRef,
    WindowTimestamps,
)
from timenet_connectors.sources.json_stream import read_entry
from timenet_connectors.time_axes import axis_for_offsets


_SKY_COVER_CODEBOOK = pa.array(SkyCover.labels(), type=pa.string())


class TraceWindowMetadata(BaseModel):
    """One published window, as its Record's TRACE metadata lists it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: int
    anchor: int
    """The window's anchor triplet: three windows, one per sampling, cut before the same event."""
    source_file: str
    county: County
    """The county the window names: an event window names the county of its event, not the station's."""
    event_type: EventLabel | None = None
    """The label as the file states it, ``NO_EVENT`` included; only ``pretrain`` windows have one."""


class TraceRecordMetadata(MetadataExtension):
    """TRACE's extension of a Record's metadata: where its station stands, its sampling, and its windows.

    The station's id is the core ``subject_id``; this holds what only TRACE has.
    """

    NAME: ClassVar[str] = "trace"

    coordinates: Coordinates
    mode: WindowSampling
    windows: list[TraceWindowMetadata]
    """Every published window with these measurements, in id order. Copies keep their own county and label."""


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

    def values(self, ref: WindowRef, channel: Channel) -> pa.Array:
        """Return one channel of a window: float32 values, or ``sky_code`` as codebook indices.

        Returns:
            The channel's values as an Arrow array.
        """
        values = self._row(ref)[channel]
        if channel is Channel.SKY_COVER:
            # The published codes are already indices into the codebook.
            return pa.DictionaryArray.from_arrays(pa.array(values, type=pa.int32()), _SKY_COVER_CODEBOOK)
        return pa.array(values, type=pa.float32())

    def time_offsets(self, ref: WindowRef) -> pa.Array:
        """Return a window's local-clock time offsets, for a Record whose axis has gaps.

        Returns:
            One int64 microsecond offset per value, starting at 0.
        """
        # Validated when convert indexed the file, so the reread is not validated again.
        timestamps = WindowTimestamps.model_construct(self._row(ref)["DATE"])
        return pa.array(timestamps.offsets_us, type=pa.int64())


@dataclass(frozen=True, eq=False)
class WindowTimeline:
    """What a Record keeps of its window's timestamps: the two it publishes, and every offset.

    The published list holds one string per value; 63,366 Measurements would keep about 10.8 million.
    The Record needs only its first and last timestamps, for its annotations, and the offsets, for its
    time axis and its forecasting target.
    """

    first: str
    """The first timestamp, as published: local clock time, no time zone."""
    last: str
    """The last timestamp, as published."""
    offsets_us: np.ndarray
    """Every timestamp's local-clock offset from the first, in microseconds, starting at 0."""

    @classmethod
    def of(cls, timestamps: WindowTimestamps) -> "WindowTimeline":
        """Return what a Record needs of a window's validated timestamps, without the published list.

        Returns:
            The first and last timestamps as published, and every timestamp's offset.
        """
        return cls(first=timestamps.root[0], last=timestamps.root[-1], offsets_us=timestamps.offsets_us)

    def __len__(self) -> int:
        return len(self.offsets_us)

    def last_values_span(self, count: int) -> TimeInterval:
        """Return the span of the last ``count`` values, however much time they cover across gaps.

        Returns:
            The interval from the first of those values to just after the last, in offsets.
        """
        return TimeInterval(start_us=int(self.offsets_us[-count]), end_us=int(self.offsets_us[-1]) + 1)


# eq=False on both types: a window points to its measurement, which lists its windows, so a
# field-by-field comparison would recurse between them. Both hash by identity.
@dataclass(eq=False)
class Measurement:
    """One distinct set of measurements: the values that several TraceWindows can share, and one Record.

    Two windows share a Measurement when their station, sampling, timestamps, and every value of every
    channel are equal.
    """

    station: Station
    sampling: WindowSampling
    timeline: WindowTimeline
    value_source: WindowRef
    """The first window seen with these measurements; the Record's values load from its bytes."""
    windows: list["TraceWindow"] = field(default_factory=list)
    """Every TraceWindow with these measurements."""

    @property
    def n_values(self) -> int:
        """Return the number of values in each channel."""
        return len(self.timeline)

    @cached_property
    def axis(self) -> RegularAxis | IrregularAxis:
        """Return the time axis the seven channels share.

        A gap-free window is regular at its sampling's cadence; one with missing timestamps keeps its
        explicit offsets.
        """
        return axis_for_offsets(self.timeline.offsets_us, period_us=Fraction(self.sampling.cadence_us))

    def is_long_enough_for_tasks(self) -> bool:
        """Return whether the authors give a window this long a task: at least ``MIN_TASK_VALUES`` values."""
        return self.n_values >= MIN_TASK_VALUES

    def check_station(self, station: Station, ref: WindowRef) -> None:
        """Check that a copy of these measurements was taken at the same coordinates.

        Raises:
            TimeFValidationError: If the copy's coordinates differ from the first window's.
        """
        if station.coordinates != self.station.coordinates:
            raise TimeFValidationError(
                f"{ref.context}: same measurements as window {self.value_source.key}, but at "
                f"{station.coordinates} instead of {self.station.coordinates}"
            )

    @property
    def metadata(self) -> RecordMetadata:
        """Return the Record's metadata: its station as the subject, and TRACE's extension."""
        windows = sorted(self.windows, key=lambda window: window.ref.window_id)
        return RecordMetadata(subject_id=self.station.station_id).extend(
            TraceRecordMetadata(
                coordinates=self.station.coordinates,
                mode=self.sampling,
                windows=[window.metadata for window in windows],
            )
        )

    def to_record(self, loader: WindowLoader) -> Record:
        """Build the seven-channel station Record, whose values stay in the source file.

        Returns:
            A Record with lazy signals on one shared time axis, its dates as annotations, and its station
            and TraceWindows as metadata.
        """
        source, axis = self.value_source, self.axis
        offsets_loader = partial(loader.time_offsets, source) if isinstance(axis, IrregularAxis) else None
        signals = tuple(
            Signal.from_loader(
                name=channel.value,
                spec=channel.spec,
                time_axis=axis,
                time_offsets_loader=offsets_loader,
                n_values=self.n_values,
                loader=partial(loader.values, source, channel),
            )
            for channel in Channel
        )
        # No start_time: the timestamps are local clock time in an unstated zone, so the Record has no
        # absolute anchor. The published first and last timestamps stay in the annotations below.
        record = Record(
            sources=(
                Source(
                    # A generic name: the station id is a strong prior on the event type, so it stays in
                    # metadata, which is never model input.
                    name="station",
                    signals=signals,
                    # TimeF does not keep signal order on read, so record the source's channel order.
                    metadata={"channel_order": [channel.value for channel in Channel]},
                ),
            ),
            metadata=self.metadata.to_json(),
        )
        for key, value, description in (
            (
                "window_start",
                self.timeline.first,
                "First timestamp, as published: station local clock time, no time zone. "
                "Not a retrieval input: every description restates these dates",
            ),
            (
                "window_end",
                self.timeline.last,
                "Last timestamp, as published: station local clock time, no time zone. "
                "Not a retrieval input: every description restates these dates",
            ),
        ):
            record.annotate(Annotation(key=key, value=value, description=description))
        return record


@dataclass(frozen=True, eq=False)
class TraceWindow:
    """One window as the authors number it: one station and sampling, before an event or not.

    The archive's 74,337 windows hold 63,366 distinct Measurements. Copies share their Measurement,
    and so their Record, but each keeps its own id, label, county, split, and tasks.
    """

    ref: WindowRef
    measurement: Measurement
    county: County
    """The county the window names: an event window names the county of its event."""
    label: EventLabel | None
    """The label as the file states it, ``NO_EVENT`` included; ``None`` outside ``pretrain``."""

    def is_eligible_for_classification(self) -> bool:
        """Return whether the authors classify the window: it names an event and is long enough."""
        return self.label is not None and self.label.is_event and self.measurement.is_long_enough_for_tasks()

    def is_eligible_for_forecasting(self) -> bool:
        """Return whether the authors forecast the window: it is a forecasting window and long enough."""
        return self.ref.file.kind is TraceTaskKind.FORECASTING and self.measurement.is_long_enough_for_tasks()

    def check_report(self, report: EventReport | None) -> None:
        """Check that the window has an event report exactly when it is classified, and the one for its label.

        Raises:
            TimeFValidationError: If the window's event report is missing, mismatched, or unexpected.
        """
        if self.is_eligible_for_classification():
            if report is None or report.event_type != self.label:
                raise TimeFValidationError(f"{self.ref.context}: no event report with event_type {self.label}")
        elif report is not None:
            raise TimeFValidationError(f"{self.ref.context}: event report for a window without an event-type task")

    def forecast_target_span(self) -> TimeInterval:
        """Return the span of the window's last ``FORECAST_HORIZON`` values, the forecasting target."""
        return self.measurement.timeline.last_values_span(FORECAST_HORIZON)

    @property
    def metadata(self) -> TraceWindowMetadata:
        """Return the window as its Record's metadata lists it: its identity, county, and stated label."""
        return TraceWindowMetadata(**self.ref.provenance, county=self.county, event_type=self.label)


class TraceRecords:
    """The Record of each Measurement, built once every window is indexed."""

    def __init__(self, records: dict[Measurement, Record]) -> None:
        self._records = records

    def of(self, window: TraceWindow) -> Record:
        """Return the Record that holds a window's measurements."""
        return self._records[window.measurement]

    def distinct_of(self, windows: Iterable[TraceWindow]) -> tuple[Record, ...]:
        """Return the Records of some windows, each once, in the windows' order.

        Copies of one measurement share a Record, so there can be fewer Records than windows.

        Returns:
            The distinct Records.
        """
        return tuple({id(record): record for record in map(self.of, windows)}.values())

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records.values())
