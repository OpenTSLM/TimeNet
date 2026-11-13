"""Read and check one window as the JSON files publish it.

A window is one station's readings at one sampling: its timestamps, and the values of seven channels
at them. ``WindowRow.parse`` checks a whole window, and ``Readings.parse`` checks only the series,
which is all a reread needs.
"""

import hashlib
import json
import re
from typing import Any, NamedTuple

import numpy as np

from timenet.errors import TimeFValidationError
from timenet.types import TimeInterval
from timenet_connectors.datasets.trace.weather_events.archive import Kind
from timenet_connectors.datasets.trace.weather_events.fields import as_object, degrees, field, text
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel, EventLabel, Sampling, SourceField


# YYYY-MM-DDTHH:MM:SS with no fractional seconds or time zone. NumPy would parse looser inputs such as
# "NaT" or a "Z" suffix, so the format is checked first.
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
_LOCATION_PARTS = 2
"""A published ``location`` names a county and its state."""
_MAX_LATITUDE = 90
_MAX_LONGITUDE = 180
_PREVIEW_CHARS = 80
"""How much of a bad published value an error message shows."""


class Coordinates(NamedTuple):
    """A point on the Earth, in decimal degrees."""

    latitude: float
    longitude: float

    @classmethod
    def parse(cls, published: dict[str, Any]) -> "Coordinates":
        """Return a window's published latitude and longitude, checked to lie on the Earth.

        Returns:
            The coordinates.
        """
        latitude = degrees(field(published, SourceField.LATITUDE), SourceField.LATITUDE, _MAX_LATITUDE)
        longitude = degrees(field(published, SourceField.LONGITUDE), SourceField.LONGITUDE, _MAX_LONGITUDE)
        return cls(latitude, longitude)


class County(NamedTuple):
    """The county a window names, from NOAA Storm Events.

    It belongs to the window, not the station: an event window names the county of its event. A
    station's windows without an event all name one county, but 17 of the 122 stations with event
    windows name up to three, so copies of one measurement can name different counties.
    """

    name: str
    state: str

    @classmethod
    def parse(cls, location: object) -> "County":
        """Split a published ``location`` into a name and a state, each as published.

        Both folders publish both forms: ``"PIKE,KENTUCKY"``, or ``["Los Alamos County", "New Mexico"]``.

        Returns:
            The county's name and state.

        Raises:
            TimeFValidationError: If the location is in neither form.
        """
        if isinstance(location, str) and location.count(",") == 1:
            parts = location.split(",")
        elif isinstance(location, list) and len(location) == _LOCATION_PARTS:
            parts = location
        else:
            raise TimeFValidationError(f'location: expected "COUNTY,STATE" or ["County", "State"], got {location!r}')
        return cls(text(parts[0], "location name"), text(parts[1], "location state"))


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
    def parse(cls, dates: object, sampling: Sampling) -> "LocalTimestamps":
        """Check published timestamps: real dates that increase on the sampling's cadence. Gaps are allowed.

        Returns:
            The timestamps and their offsets.

        Raises:
            TimeFValidationError: If a timestamp is malformed, out of order or off the cadence.
        """
        if not isinstance(dates, list) or not dates:
            raise TimeFValidationError(f"DATE: expected a non-empty list, got {dates!r:.{_PREVIEW_CHARS}}")
        for index, date in enumerate(dates):
            if not isinstance(date, str) or not _TIMESTAMP.fullmatch(date):
                raise TimeFValidationError(f"DATE[{index}]: expected YYYY-MM-DDTHH:MM:SS, got {date!r}")
        try:
            clock_us = np.array(dates, dtype="datetime64[us]").astype(np.int64)
        except (ValueError, OverflowError) as exc:
            raise TimeFValidationError(f"DATE: {exc}") from exc
        offsets = clock_us - clock_us[0]
        steps = np.diff(offsets)
        if (steps <= 0).any() or (steps % sampling.cadence_us).any():
            raise TimeFValidationError(f"DATE must increase on the {sampling} cadence; gaps are allowed")
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


def parse_sampling(published: dict[str, Any]) -> Sampling:
    """Return the sampling that a published ``mode`` names.

    Returns:
        The sampling.

    Raises:
        TimeFValidationError: If ``mode`` is not one of the three samplings.
    """
    mode = field(published, SourceField.MODE)
    try:
        return Sampling(mode)
    except ValueError as exc:
        raise TimeFValidationError(f"mode: expected one of {[s.value for s in Sampling]}, got {mode!r}") from exc


class Readings(NamedTuple):
    """A window's timestamps and the values of every channel at them."""

    timestamps: LocalTimestamps
    values: dict[Channel, np.ndarray]
    """Float64 values, or int64 sky cover codes, one per timestamp."""

    @classmethod
    def parse(cls, published: dict[str, Any], sampling: Sampling) -> "Readings":
        """Check a window's timestamps and channels.

        Returns:
            The timestamps and the values of every channel.
        """
        dates = field(published, SourceField.DATE)
        timestamps = LocalTimestamps.parse(dates, sampling)
        values = {channel: channel.parse_values(field(published, channel.value), len(dates)) for channel in Channel}
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
            TimeFValidationError: If a field is missing or invalid, the mode does not match the id's
                anchor position, or the event label does not suit the folder.
        """
        published = as_object(published)
        sampling = parse_sampling(published)
        if sampling is not Sampling.for_window_id(window_id):
            raise TimeFValidationError(f"mode {sampling.value!r} does not match the id's anchor position")
        # Only pretrain windows publish a label, and they must. A published null is not an absent label.
        label = None
        if kind is Kind.PRETRAIN:
            label = EventLabel.parse(field(published, SourceField.EVENT_TYPE), SourceField.EVENT_TYPE)
        elif SourceField.EVENT_TYPE in published:
            raise TimeFValidationError(f"a {kind} window must not publish an event_type")
        return cls(
            station_id=text(field(published, SourceField.STATION_ID), SourceField.STATION_ID),
            coordinates=Coordinates.parse(published),
            county=County.parse(field(published, SourceField.LOCATION)),
            sampling=sampling,
            readings=Readings.parse(published, sampling),
            label=label,
        )

    def digest(self) -> bytes:
        """Return the digest of the window's measurements: see ``Readings.digest``."""
        return self.readings.digest(self.station_id, self.sampling.value)
