"""One window as the JSON files publish it: its station, county, sampling, label, and readings."""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from typing import Annotated, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.trace.weather_events._fields import (
    EventLabelField,
    Number,
    Numbers,
    SkyCodes,
    Text,
    Timestamp,
    nest,
)
from timenet_connectors.datasets.trace.weather_events.archive import TraceFileRef, TraceTaskKind
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel, WindowSampling


class Coordinates(BaseModel):
    """A point on the Earth, in decimal degrees."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="Coordinates")

    latitude: Annotated[Number, Field(ge=-90, le=90)]
    longitude: Annotated[Number, Field(ge=-180, le=180)]

    def __str__(self) -> str:
        return f"({self.latitude}, {self.longitude})"


# The archive's windows come from 354 stations, all with 11-character GHCN ids (351 USW, 3 USI). The
# ids are data, not a vocabulary, so there is no enum of them: the pinned archive is their source.
# A station's coordinates are not one exact pair: 272 of the 354 appear with float noise of up to
# 5.7e-14 degrees, such as USW00000184 at latitude 29.9761 and 29.976099999999995. Group windows by
# station_id, not by coordinates, and do not round them: that would repair the source. A window's
# county is not the station's: see County.
class Station(BaseModel):
    """The GHCN-h station that measured a window: its id and where it stands."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="Station")

    station_id: Text
    coordinates: Coordinates


class County(BaseModel):
    """The county a window names, from NOAA Storm Events: for an event window, its event's county.

    It belongs to the window, not the Station: an event window names the county of its event, which
    can differ from the station's. A station's windows without an event all name one county, but 17
    of the 122 stations with event windows name up to three, so copies of one measurement can name
    different counties.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="County")

    name: Text
    state: Text

    @classmethod
    def split_published(cls, location: object) -> dict[str, object]:
        """Split a published ``location`` into a name and a state, as published.

        Both folders publish both forms: ``"PIKE,KENTUCKY"``, or ``["Los Alamos County", "New Mexico"]``.
        Each keeps its casing and any "County" suffix.

        Returns:
            The ``name`` and ``state``, for the model to validate.

        Raises:
            TimeFValidationError: If the location is in neither form.
        """
        if isinstance(location, str) and location.count(",") == 1:
            name, state = location.split(",")
            return {"name": name, "state": state}
        if isinstance(location, list) and len(location) == 2:  # noqa: PLR2004 - a name and a state
            name, state = location
            return {"name": name, "state": state}
        raise TimeFValidationError(f'location: expected "COUNTY,STATE" or ["County", "State"], got {location!r}')


class WindowTimestamps(RootModel[Annotated[list[Timestamp], Field(min_length=1)]]):
    """A window's timestamps as published: naive local clock time at the station, no time zone.

    The source states no time zone, so a timestamp is not a moment in Unix time. Only the difference
    between two timestamps means anything: the local-clock time between them. Across a daylight
    saving change, that differs from the elapsed time by an hour.
    """

    model_config = ConfigDict(frozen=True)

    @cached_property
    def offsets_us(self) -> np.ndarray:
        """Return each timestamp's local-clock offset from the first, in microseconds, starting at 0."""
        readings = np.array(self.root, dtype="datetime64[us]").astype(np.int64)
        return readings - readings[0]

    def __len__(self) -> int:
        return len(self.root)

    def check_cadence(self, sampling: WindowSampling) -> None:
        """Check that the timestamps increase on the sampling's cadence. Gaps are allowed.

        Raises:
            TimeFValidationError: If a timestamp is out of order or off the cadence.
        """
        steps = np.diff(self.offsets_us)
        if (steps <= 0).any() or (steps % sampling.cadence_us).any():
            raise TimeFValidationError(f"DATE must increase on the {sampling} cadence; gaps are allowed")

    @model_validator(mode="after")
    def _real_dates(self) -> Self:
        # The pattern fixed the format, so only an impossible date such as 02-30 fails here.
        try:
            self.offsets_us  # noqa: B018 - parses and caches the offsets
        except (ValueError, OverflowError) as exc:
            raise TimeFValidationError(f"DATE: {exc}") from exc
        return self


class WindowReadings(BaseModel):
    """A window's timestamps, and the value of each Channel at every one of them."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE readings")

    timestamps: WindowTimestamps = Field(alias="DATE")
    # One field per Channel, named by its JSON key: the sky codes are bounded ints, the rest finite floats.
    temperature: Numbers
    precipitation: Numbers
    relative_humidity: Numbers
    visibility: Numbers
    wind_u: Numbers
    wind_v: Numbers
    sky_code: SkyCodes

    def values(self, channel: Channel) -> list[float] | list[int]:
        """Return one channel's values, one per timestamp."""
        return getattr(self, channel.value)

    @model_validator(mode="after")
    def _one_value_per_timestamp(self) -> Self:
        for channel in Channel:
            count = len(self.values(channel))
            if count != len(self.timestamps):
                raise TimeFValidationError(f"{channel} has {count} values for {len(self.timestamps)} timestamps")
        return self


class WindowRow(BaseModel):
    """One window as the JSON files publish it: its station, county, sampling, label, and readings.

    A window is one station's hourly, 4-hourly, or daily readings over 7, 28, or 180 days. Seven
    channels share its timestamps: temperature, precipitation, relative humidity, visibility, the
    eastward and northward wind, and sky cover. The authors cut each event's windows to end before
    the event begins, and cut the windows without an event the same way.

    Read a row with ``parse``: the model checks each field, that every timestamp is a real date, and
    that every channel has one value per timestamp; ``parse`` adds the checks that need the window's
    id and folder.
    """

    # The title names the model in validation errors, which reach users with the file and key.
    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE window")

    # Published flat as station_id, latitude, and longitude; grouped on read.
    station: Station
    # Published as "location"; see County for why it is not the station's.
    county: County
    # Unlike an IntEnum, a StrEnum field already rejects JSON numbers and booleans.
    mode: WindowSampling
    readings: WindowReadings
    # Only pretrain windows publish a label; parse checks that against the folder.
    event_type: EventLabelField | None = None

    @model_validator(mode="before")
    @classmethod
    def _group_published(cls, data: Any) -> Any:
        if isinstance(data, dict) and "location" in data:
            # The source calls the county "location"; see County for why it is not the station's.
            if "county" in data:
                raise TimeFValidationError("a published row has no 'county' key")
            location = data["location"]
            data = {key: value for key, value in data.items() if key != "location"}
            data["county"] = County.split_published(location)
        data = nest(data, ("latitude", "longitude"), "coordinates")
        data = nest(data, ("station_id", "coordinates"), "station")
        return nest(data, ("DATE", *Channel), "readings")

    @classmethod
    def parse(cls, published: Any, *, window_id: int, kind: TraceTaskKind, location: str) -> Self:
        """Read one published window, and check it against its id and its folder.

        Args:
            published: The window as its JSON file publishes it.
            window_id: The window's id, whose anchor position gives its sampling.
            kind: The window's folder: only ``pretrain`` windows publish an event label.
            location: The file and key that an error message names.

        Returns:
            The validated window.

        Raises:
            TimeFValidationError: If the window is malformed, naming its file and key.
        """
        try:
            window = cls.model_validate(published)
            window._check_event_label(kind)
            window._check_sampling(window_id)
        # A pydantic ValidationError is a ValueError, so it gains the same context.
        except ValueError as exc:
            raise TimeFValidationError(f"{location}: {exc}") from exc
        return window

    def _check_sampling(self, window_id: int) -> None:
        # The sampling must be the one the id's anchor position gives, then the timestamps on its cadence.
        if self.mode is not WindowSampling.for_window_id(window_id):
            raise TimeFValidationError(f"mode {self.mode.value!r} does not match the id's anchor position")
        self.readings.timestamps.check_cadence(self.mode)

    def _check_event_label(self, kind: TraceTaskKind) -> None:
        # Pretrain windows must publish an event label, and other windows must not.
        if kind is TraceTaskKind.PRETRAIN and self.event_type is None:
            raise TimeFValidationError(f"a {kind} window must publish an event_type")
        # The fields set tell an absent event_type from a published null.
        if kind is not TraceTaskKind.PRETRAIN and "event_type" in self.model_fields_set:
            raise TimeFValidationError(f"a {kind} window must not publish an event_type")

    def measurement_key(self) -> str:
        """Return a digest of what makes two windows the same measurements.

        Two windows match when their station, mode, timestamps, and every value of every channel are
        equal. Floats keep their shortest round-trip form, so equal values give equal text.

        Returns:
            The SHA-256 hex digest.
        """
        readings = self.readings
        fields = [
            self.station.station_id,
            self.mode.value,
            readings.timestamps.root,
            *(readings.values(channel) for channel in Channel),
        ]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class WindowRef:
    """Where one TraceWindow sits in its file."""

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
            "anchor": self.window_id // len(WindowSampling),
            "source_file": self.file.archive_path,
        }
