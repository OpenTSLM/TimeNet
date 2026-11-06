"""Read and check one window as the JSON files publish it.

A window is one station's readings at one sampling: its timestamps, and the values of seven channels
at them. ``WindowRow.parse`` checks a whole window, and ``Readings.parse`` checks only the series,
which is all a reread needs.
"""

import hashlib
import json
from typing import Annotated, NamedTuple

import numpy as np
from pydantic import Field, StringConstraints, field_validator
from pydantic_core import PydanticCustomError

from timenet.errors import TimeNetConnectorError
from timenet.types import TimeInterval
from timenet_connectors.datasets.trace.weather_events.archive import Kind
from timenet_connectors.datasets.trace.weather_events.vocabulary import (
    SKY_COVER_LABELS,
    Channel,
    EventLabel,
    PublishedEventLabel,
    Sampling,
)
from timenet_connectors.sources.rows import NonBlankText, PublishedStrEnum, SourceRow, validate_row


# YYYY-MM-DDTHH:MM:SS with no fractional seconds or time zone. NumPy would parse looser inputs such as
# "NaT" or a "Z" suffix, so the format is checked first.
TimestampText = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")]
"""A published local timestamp; calendar validity and cadence are checked when building offsets."""
SkyCode = Annotated[int, Field(ge=0, lt=len(SKY_COVER_LABELS))]
"""An integer index into TRACE's sky-cover codebook."""


class County(NamedTuple):
    """The window's county and state, preserving their published spelling.

    Counties belong to windows: equal measurements can precede events in different counties.
    """

    name: NonBlankText
    state: NonBlankText


class WindowFields(SourceRow):
    """Published window fields; channel values are validated against their signal contracts."""

    station_id: NonBlankText
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location: County
    mode: PublishedStrEnum[Sampling]
    dates: list[TimestampText] = Field(alias="DATE", min_length=1)
    temperature: object
    precipitation: object
    relative_humidity: object
    visibility: object
    wind_u: object
    wind_v: object
    sky_code: list[SkyCode]

    @field_validator("location", mode="before")
    @classmethod
    def _county_parts(cls, value: object) -> tuple[object, ...]:
        # Both folders publish "PIKE,KENTUCKY" and ["Los Alamos County", "New Mexico"].
        if isinstance(value, str):
            return tuple(value.split(","))
        if isinstance(value, list):
            return tuple(value)
        raise PydanticCustomError("county", 'expected "COUNTY,STATE" or ["County", "State"]')


class _PretrainFields(WindowFields):
    event_type: PublishedEventLabel


def read_window_fields(published: object, *, kind: Kind | None = None) -> WindowFields:
    """Return typed source fields for a window or a lazy reread.

    Raises:
        TimeNetConnectorError: If the fields violate the source schema or folder contract.
    """
    labelled = kind is Kind.PRETRAIN or (kind is None and isinstance(published, dict) and "event_type" in published)
    if kind is not None and kind is not Kind.PRETRAIN and isinstance(published, dict) and "event_type" in published:
        raise TimeNetConnectorError(f"a {kind} window must not publish an event_type")
    return validate_row(published, _PretrainFields if labelled else WindowFields, location="window")


class Coordinates(NamedTuple):
    """A point on the Earth, in decimal degrees."""

    latitude: float
    longitude: float


class LocalTimestamps(NamedTuple):
    """A window's timestamps: naive local clock time at the station, with no time zone.

    The zone is unstated, so the timestamps have no absolute anchor. They are kept as published, and
    as offsets from the first one, which is all a TimeF time axis needs.
    """

    dates: list[str]
    """The timestamps as published."""
    offsets_us: np.ndarray
    """Each timestamp's local-clock offset from the first, in microseconds, starting at 0."""

    @classmethod
    def parse(cls, dates: list[str], sampling: Sampling) -> "LocalTimestamps":
        """Check published timestamps: real dates that increase on the sampling's cadence. Gaps are allowed.

        Returns:
            The timestamps and their offsets.

        Raises:
            TimeNetConnectorError: If a timestamp is malformed, out of order or off the cadence.
        """
        try:
            clock_us = np.array(dates, dtype="datetime64[us]").astype(np.int64)
        except (ValueError, OverflowError) as exc:
            raise TimeNetConnectorError(f"DATE: {exc}") from exc
        offsets = clock_us - clock_us[0]
        steps = np.diff(offsets)
        if (steps <= 0).any() or (steps % sampling.cadence_us).any():
            raise TimeNetConnectorError(f"DATE must increase on the {sampling} cadence; gaps are allowed")
        return cls(dates, offsets)

    @property
    def first(self) -> str:
        """The first timestamp, as published."""
        return self.dates[0]

    @property
    def last(self) -> str:
        """The last timestamp, as published."""
        return self.dates[-1]

    def span_of_last(self, count: int) -> TimeInterval:
        """Return the span of the last ``count`` timestamps, in offsets from the first.

        Returns:
            The interval from the ``count``-th last timestamp to just after the last one.
        """
        return TimeInterval(start_us=int(self.offsets_us[-count]), end_us=int(self.offsets_us[-1]) + 1)


class Readings(NamedTuple):
    """A window's timestamps and the values of every channel at them."""

    timestamps: LocalTimestamps
    values: dict[Channel, np.ndarray]
    """Float64 values, or int64 sky cover codes, one per timestamp."""

    @classmethod
    def parse(cls, published: WindowFields, sampling: Sampling) -> "Readings":
        """Check a window's timestamps and channels.

        Returns:
            The timestamps and the values of every channel.
        """
        dates = published.dates
        timestamps = LocalTimestamps.parse(dates, sampling)
        values = {channel: channel.parse_values(getattr(published, channel.value), len(dates)) for channel in Channel}
        return cls(timestamps, values)

    @property
    def n_values(self) -> int:
        """The number of timestamps, and of values in each channel."""
        return len(self.timestamps.dates)

    def digest(self, station_id: object, mode: object) -> bytes:
        """Return a digest of what makes two windows the same measurements.

        Two windows match when their station, sampling, timestamps and every value of every channel are
        equal. The digest also tells a reread of a window from another series with the same lengths.

        Returns:
            A SHA-256 digest.
        """
        digest = hashlib.sha256(json.dumps([station_id, mode, self.timestamps.dates]).encode())
        for channel in Channel:
            digest.update(self.values[channel].tobytes())
        return digest.digest()


class WindowRow(NamedTuple):
    """One window as the JSON files publish it, checked."""

    station_id: str
    coordinates: Coordinates
    county: County
    sampling: Sampling
    readings: Readings
    label: EventLabel | None
    """The label as the file states it, ``NO_EVENT`` included; only ``pretrain`` windows have one."""

    @classmethod
    def parse(cls, published: object, *, window_id: int, kind: Kind) -> "WindowRow":
        """Check one published window against its id and its folder.

        Returns:
            The checked window.

        Raises:
            TimeNetConnectorError: If a field is missing or invalid, the mode does not match the id's
                anchor position, or the event label does not suit the folder.
        """
        published = read_window_fields(published, kind=kind)
        sampling = published.mode
        if sampling is not Sampling.for_window_id(window_id):
            raise TimeNetConnectorError(f"mode {sampling.value!r} does not match the id's anchor position")
        # Only pretrain windows publish a label, and they must. A published null is not an absent label.
        label = None
        if isinstance(published, _PretrainFields):
            label = published.event_type
        return cls(
            station_id=published.station_id,
            coordinates=Coordinates(published.latitude, published.longitude),
            county=published.location,
            sampling=sampling,
            readings=Readings.parse(published, sampling),
            label=label,
        )

    def digest(self) -> bytes:
        """Return the digest of the window's measurements: see ``Readings.digest``."""
        return self.readings.digest(self.station_id, self.sampling.value)
