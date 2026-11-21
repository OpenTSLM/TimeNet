"""The connector base for a wind farm that Cubico released as Greenbyte exports.

A release has one or more zip archives for each year. An archive holds two files for each
turbine: a table of ten-minute SCADA statistics and a status log. A subclass states the files of
its release. One Record is one turbine for one UTC day.
"""

from collections import defaultdict
from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import MissingReason, MissingValues, TimeInterval, TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.datasets.cubico.kelmarsh import tasks
from timenet_connectors.datasets.cubico.kelmarsh.greenbyte import (
    DAY_US,
    STEP_US,
    EventKind,
    OpenStop,
    StatusEvent,
    StatusLog,
    TurbineDay,
    read_status_log,
    read_turbine_days,
    read_turbine_name,
)
from timenet_connectors.download import Artifact, download_files, ensure_archive


DATA_PREFIX = "Turbine_Data_"
STATUS_PREFIX = "Status_"

_GAPS = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="The source has no value for the ten-minute period, for example after a communication loss.",
)


def _spec(spec_type: str, name: str, unit: object) -> TimeSeriesSpec:
    return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=unit, dtype="float64", missing=_GAPS)


# The signal name, the source column, and the spec. Each value is the mean of a ten-minute period.
SIGNALS: tuple[tuple[str, str, TimeSeriesSpec], ...] = (
    ("power", "Power (kW)", _spec("active_power", "Active power", ureg.kilowatt)),
    ("wind_speed", "Wind speed (m/s)", _spec("wind_speed", "Wind speed", ureg.meter / ureg.second)),
    (
        "pitch_angle",
        "Blade angle (pitch position) A (°)",
        _spec("blade_pitch_angle", "Blade pitch angle", ureg.degree),
    ),
    ("rotor_speed", "Rotor speed (RPM)", _spec("rotor_speed", "Rotor speed", ureg.rpm)),
    ("nacelle_direction", "Nacelle position (°)", _spec("nacelle_direction", "Nacelle direction", ureg.degree)),
    (
        "ambient_temperature",
        "Nacelle ambient temperature (°C)",
        _spec("ambient_temperature", "Ambient temperature", ureg.degC),
    ),
)
SOURCE_COLUMNS = tuple(column for _, column, _ in SIGNALS)
# A value is at the start of its period, as in the source.
EVERY_TEN_MINUTES = RegularAxis(period_us=Fraction(STEP_US))


@dataclass(frozen=True)
class TurbineYearFiles:
    """The files of one turbine for one year, and the table that describes all turbines."""

    data_path: Path
    status_path: Path
    static_path: Path


@dataclass
class _TurbineLog:
    """The status entries of one turbine, from the logs of every year."""

    events: set[StatusEvent]
    open_stops: set[OpenStop]
    ok_times_us: set[int]

    def add(self, log: StatusLog) -> None:
        self.events.update(log.events)
        self.open_stops.update(log.open_stops)
        self.ok_times_us.update(log.ok_times_us)


class GreenbyteFarmConnector(BaseConnector[list[TurbineYearFiles]]):
    """Connector for one wind farm release of Greenbyte exports on Zenodo."""

    FILES_URL: ClassVar[str]
    """The URL of the folder that holds the files of the Zenodo record."""
    STATIC_FILE: ClassVar[tuple[str, str]]
    """The name and the SHA-256 hash of the table that describes the turbines."""
    YEAR_ARCHIVES: ClassVar[Mapping[int, tuple[tuple[str, str], ...]]]
    """The name and the SHA-256 hash of each archive of a year."""
    SPLIT_CUTOFFS: ClassVar[tasks.SplitCutoffs] = tasks.SplitCutoffs(
        validation=datetime(2021, 7, 1, tzinfo=UTC), test=datetime(2022, 1, 1, tzinfo=UTC)
    )
    """Of the four years, the first two and a half are train, the next half is validation, and the last is test."""

    async def download_async(self, cache_dir: Path) -> list[TurbineYearFiles]:
        """Download the turbine table and the archives of each year, and extract each archive once.

        Args:
            cache_dir: The directory that holds the downloaded and extracted files.

        Returns:
            The files of each turbine and year, in year and turbine order.

        Raises:
            TimeNetConnectorError: If a turbine table of an archive has no status log.
        """
        static_name, static_sha256 = self.STATIC_FILE
        static_path = cache_dir / static_name
        await download_files(
            [Artifact(f"{self.FILES_URL}/{static_name}?download=1", static_path, sha256=static_sha256)]
        )
        refs: list[TurbineYearFiles] = []
        for year, archives in self.YEAR_ARCHIVES.items():
            for name, sha256 in archives:
                root = await ensure_archive(
                    f"{self.FILES_URL}/{name}?download=1", cache_dir / str(year), filename=name, sha256=sha256
                )
            for data_path in sorted(root.glob(f"{DATA_PREFIX}*.csv")):
                status_path = root / f"{STATUS_PREFIX}{data_path.name.removeprefix(DATA_PREFIX)}"
                if not status_path.is_file():
                    raise TimeNetConnectorError(f"{year}: {data_path.name} has no status log")
                refs.append(TurbineYearFiles(data_path=data_path, status_path=status_path, static_path=static_path))
        return refs

    def convert(self, raw_refs: list[TurbineYearFiles], /) -> TimeFDataset:
        """Create one Record for each turbine and day, with its annotations and tasks.

        A day gets no task if a stop without an end time leaves the state of the turbine unknown,
        or if a stop or curtailment of the day runs across a split cutoff.

        Args:
            raw_refs: The files from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        # A stop can run past the end of a year, so each day sees the entries of every year.
        names = [read_turbine_name(ref.data_path) for ref in raw_refs]
        logs: dict[str, _TurbineLog] = defaultdict(lambda: _TurbineLog(set(), set(), set()))
        logs_curtailment: list[bool] = []
        for ref, name in zip(raw_refs, names, strict=True):
            log = read_status_log(ref.status_path)
            logs[name].add(log)
            logs_curtailment.append(any(event.kind is EventKind.CURTAILMENT for event in log.events))
        for ref, name, curtailment in zip(raw_refs, names, logs_curtailment, strict=True):
            turbine = _read_turbine_metadata(ref.static_path, name)
            events = sorted(logs[name].events, key=lambda event: (event.start_us, event.end_us))
            open_stops = sorted(logs[name].open_stops, key=lambda stop: (stop.start_us, stop.message))
            unknown = tasks.unknown_windows(open_stops, sorted(logs[name].ok_times_us))
            crossing = self.SPLIT_CUTOFFS.crossing_days([(event.start_us, event.end_us) for event in events])
            for day in read_turbine_days(ref.data_path, SOURCE_COLUMNS):
                record = dataset.add_record(record=_create_record(day, turbine))
                day_events = tasks.events_of_day(events, day.start_us)
                tasks.annotate_day(record, day_events)
                tasks.annotate_open_stops(record, open_stops, day.start_us)
                if any(start < day.start_us + DAY_US and (end is None or end > day.start_us) for start, end in unknown):
                    continue
                if day.start_us in crossing:
                    continue
                dataset.add_tasks(
                    tasks=list(
                        tasks.iter_day_tasks(
                            record,
                            day_events,
                            split=self.SPLIT_CUTOFFS.split_of_day(day.start_us),
                            curtailment=curtailment,
                        )
                    )
                )
        return dataset


def _create_record(day: TurbineDay, turbine: JsonMapping) -> Record:
    signals = tuple(
        Signal.from_values(day.values[column].to_pylist(), spec=spec, name=name, time_axis=EVERY_TEN_MINUTES)
        for name, column, spec in SIGNALS
    )
    return Record(
        sources=(Source(name="SCADA", signals=signals),),
        start_time=TimeOrigin(day.start_us),
        # The number of steps without a power value, so that a user can keep days with full coverage.
        metadata={**turbine, "power_missing_steps": int(day.values[SIGNALS[0][1]].null_count)},
        time_span=TimeInterval.micros(0, DAY_US),
    )


def _read_turbine_metadata(static_path: Path, name: str) -> JsonMapping:
    """Read the row of one turbine from the table that describes all turbines.

    Returns:
        The facts of the turbine, as the metadata of each of its Records.

    Raises:
        TimeNetConnectorError: If the table has no row for the turbine.
    """
    with static_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Title"] == name:
                return {
                    "wind_farm": row["Wind Farm"],
                    "turbine": name,
                    "manufacturer": row["Manufacturer"],
                    "model": row["Model"],
                    "rated_power_kw": float(row["Rated power (kW)"]),
                    "hub_height_m": float(row["Hub Height (m)"]),
                    "rotor_diameter_m": float(row["Rotor Diameter (m)"]),
                    "latitude": float(row["Latitude"]),
                    "longitude": float(row["Longitude"]),
                }
    raise TimeNetConnectorError(f"{static_path.name}: no row for turbine {name!r}")
