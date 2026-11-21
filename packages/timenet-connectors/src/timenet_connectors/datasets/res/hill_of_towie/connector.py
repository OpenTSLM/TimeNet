"""The Hill of Towie connector: one Record for each turbine and UTC day, with shutdown labels.

The release has one zip archive of ten-minute SCADA tables for each year, and a file with the
seconds of shutdown of each turbine and ten-minute period. RES computed those durations from the
SCADA data and the logs of its operators. They cover January 2016 to August 2024, so this
connector reads those months. The archives stay zipped, because a year is several gigabytes of
tables and the connector reads three of them.
"""

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    MissingReason,
    MissingValues,
    ScalarPredictionTask,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    unix_us,
    ureg,
)
from timenet_connectors.datasets.cubico.kelmarsh.tasks import (
    DOWNTIME_60_PROMPT,
    STOPPED_MINUTES_PROMPT,
    STOPPED_PROMPT,
    SplitCutoffs,
)
from timenet_connectors.datasets.res.hill_of_towie.scada import (
    DAY_S,
    STEP_S,
    STEPS_PER_DAY,
    Grid,
    read_shutdown_seconds,
    read_signals,
)
from timenet_connectors.download import Artifact, download_files


SHUTDOWN_KEY = "shutdown"
STOPPED = LabelSet(
    annotations=[Annotation("name", value="stopped")],
    labels=(
        Label(name="stopped", description="The turbine was shut down for some of this day."),
        Label(name="running", description="The turbine was not shut down on this day."),
    ),
)
DOWNTIME_60 = LabelSet(
    annotations=[Annotation("name", value="downtime_60")],
    labels=(
        Label(
            name="downtime",
            description="The turbine was shut down for 60 minutes or more of this day while the wind was 4 m/s or more.",
        ),
        Label(
            name="no_downtime",
            description="The turbine had no shutdown time at wind of 4 m/s or more on this day, and almost no shutdown "
            "time without a wind value.",
        ),
    ),
)
# Shutdown at less wind is not counted: the signals cannot tell it from waiting for wind. A day with some
# of it, or with shutdown time in periods that have no wind value, is in between and gets no label.
DOWNTIME_WIND_MS = 4.0
DOWNTIME_S = 3600
UNREAD_WIND_S = 600
SHUTDOWN_PERIODS_PROMPT = (
    "In which ten-minute periods was the turbine shut down for some or all of the time on this day?"
)

_GAPS = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="The source has no row or no value for the ten-minute period.",
)


def _spec(spec_type: str, name: str, unit: object) -> TimeSeriesSpec:
    return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=unit, dtype="float64", missing=_GAPS)


# The signal name, the source table and column, and the spec. Each value is the mean of a
# ten-minute period. The names and units are those of the Kelmarsh and Penmanshiel connectors.
SIGNALS: tuple[tuple[str, str, str, TimeSeriesSpec], ...] = (
    ("power", "tblSCTurGrid", "wtc_ActPower_mean", _spec("active_power", "Active power", ureg.kilowatt)),
    (
        "wind_speed",
        "tblSCTurbine",
        "wtc_AcWindSp_mean",
        _spec("wind_speed", "Wind speed", ureg.meter / ureg.second),
    ),
    (
        "pitch_angle",
        "tblSCTurbine",
        "wtc_PitcPosA_mean",
        _spec("blade_pitch_angle", "Blade pitch angle", ureg.degree),
    ),
    ("rotor_speed", "tblSCTurbine", "wtc_MainSRpm_mean", _spec("rotor_speed", "Rotor speed", ureg.rpm)),
    (
        "nacelle_direction",
        "tblSCTurbine",
        "wtc_NacelPos_mean",
        _spec("nacelle_direction", "Nacelle direction", ureg.degree),
    ),
    (
        "ambient_temperature",
        "tblSCTurTemp",
        "wtc_AmbieTmp_mean",
        _spec("ambient_temperature", "Ambient temperature", ureg.degC),
    ),
)
# The source states the end of each period. A value is at the start of its period here.
EVERY_TEN_MINUTES = RegularAxis(period_us=Fraction(STEP_S * US_PER_S))
DAY_US = DAY_S * US_PER_S


def _tables() -> dict[str, list[str]]:
    tables: dict[str, list[str]] = {}
    for _, table, column, _ in SIGNALS:
        tables.setdefault(table, []).append(column)
    return tables


TABLES = _tables()
# The order of the signals in the values that the reader gives.
_VALUE_ORDER = [name for table in TABLES for name, signal_table, _, _ in SIGNALS if signal_table == table]


@dataclass(frozen=True)
class HillOfTowieFiles:
    """The downloaded files of the release."""

    archives: tuple[Path, ...]
    shutdown_path: Path
    metadata_path: Path


class HillOfTowieConnector(BaseConnector[list[HillOfTowieFiles]]):
    """Connector for the Hill of Towie wind farm release on Zenodo."""

    FILES_URL: ClassVar[str] = "https://zenodo.org/records/22662930/files"
    # Zenodo states an MD5 hash for each file. These SHA-256 hashes are of the files with those MD5 hashes.
    METADATA_FILE: ClassVar[tuple[str, str]] = (
        "Hill_of_Towie_turbine_metadata.csv",
        "dd40789c2bf2268da9d77890cc48617266f3ed9d13c5af2be6d6535671f261bc",
    )
    SHUTDOWN_FILE: ClassVar[tuple[str, str]] = (
        "Hill_of_Towie_ShutdownDuration.zip",
        "b69c5cb02e3c3e804b16ecdd024cead76edaba51a49a73ac12cc33f13170cab2",
    )
    YEAR_ARCHIVES: ClassVar[Mapping[int, tuple[str, str]]] = {
        2016: ("2016.zip", "800bb3d399e6b79278758bbd95e234e364b062a9ff7643a7f9c045b0db9e9738"),
        2017: ("2017.zip", "a49c11def85623b7f0537988debfa499cca15fc6ec971256051597d91224ab1f"),
        2018: ("2018.zip", "4d59191bd12ad0e85eb4de44f2f8f7d8e2556452341e89cd194a816debc48761"),
        2019: ("2019.zip", "24a9b9bfdf32558d80c518a11954d7230254f5ba384671d3df6ba3e53deb5b26"),
        2020: ("2020.zip", "3e55de08be27e6a6e702a99bfcccd57e20abfa69e7c65d5901caf581f128f020"),
        2021: ("2021.zip", "2284741792b4bfc7ca08234f49440dad4820bac5e73098011745a3f69e295c19"),
        2022: ("2022.zip", "1b0f8657273ddbb8c4f84042a0bcb1c5eb1fb7833849b4c491f2b9f10604700d"),
        2023: ("2023.zip", "e65f7acd65664584e5b72e4ad7ccd4c0f0af4ad68f618b8cf0b4e5c1f4a69521"),
        2024: ("2024.zip", "b6bb540e3d2fcc2cff8756d96ef4155037b1cc41b284e6249b41054f4e50fbc3"),
    }
    FIRST_DAY: ClassVar[datetime] = datetime(2016, 1, 1, tzinfo=UTC)
    END_DAY: ClassVar[datetime] = datetime(2024, 9, 1, tzinfo=UTC)
    """The day after the last day. The shutdown durations end with August 2024."""
    SPLIT_CUTOFFS: ClassVar[SplitCutoffs] = SplitCutoffs(
        validation=datetime(2022, 1, 1, tzinfo=UTC), test=datetime(2023, 1, 1, tzinfo=UTC)
    )
    """The first six years are train, 2022 is validation, and 2023 to August 2024 is test."""

    async def download_async(self, cache_dir: Path) -> list[HillOfTowieFiles]:
        """Download the turbine table, the shutdown durations, and the archive of each year.

        Args:
            cache_dir: The directory that holds the downloaded files.

        Returns:
            One entry with the paths of the files.
        """
        names = [self.METADATA_FILE, self.SHUTDOWN_FILE, *self.YEAR_ARCHIVES.values()]
        await download_files(
            [Artifact(f"{self.FILES_URL}/{name}?download=1", cache_dir / name, sha256=sha256) for name, sha256 in names]
        )
        return [
            HillOfTowieFiles(
                archives=tuple(cache_dir / name for name, _ in self.YEAR_ARCHIVES.values()),
                shutdown_path=cache_dir / self.SHUTDOWN_FILE[0],
                metadata_path=cache_dir / self.METADATA_FILE[0],
            )
        ]

    def convert(self, raw_refs: list[HillOfTowieFiles], /) -> TimeFDataset:
        """Create one Record for each turbine and day, with its annotations and tasks.

        A day without a row in any of the tables gets no Record. A day gets no task if a run of
        shutdown of the day runs across a split cutoff.

        Args:
            raw_refs: The entry from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (files,) = raw_refs
        dataset = TimeFDataset(metadata=self.metadata())
        turbines = _read_turbines(files.metadata_path)
        start_s = int(self.FIRST_DAY.timestamp())
        days = (int(self.END_DAY.timestamp()) - start_s) // DAY_S
        grid = Grid(
            station_ids=tuple(station for station, _ in turbines),
            turbine_names=tuple(str(turbine["turbine"]) for _, turbine in turbines),
            start_s=start_s,
            steps=days * STEPS_PER_DAY,
        )
        shutdown = read_shutdown_seconds(files.shutdown_path, grid)
        values, present = read_signals(files.archives, TABLES, grid)
        any_row = present.any(axis=0)
        for index, (_, turbine) in enumerate(turbines):
            crossing = self._crossing_days(shutdown[index])
            for day in range(days):
                steps = slice(day * STEPS_PER_DAY, (day + 1) * STEPS_PER_DAY)
                if not any_row[index, steps].any():
                    continue
                day_start_us = unix_us(self.FIRST_DAY) + day * DAY_US
                record = dataset.add_record(record=_create_record(values[index, steps], day_start_us, turbine))
                split = None if day_start_us in crossing else self.SPLIT_CUTOFFS.split_of_day(day_start_us)
                _label_day(
                    dataset,
                    record,
                    shutdown[index, steps],
                    values[index, steps, _VALUE_ORDER.index("wind_speed")],
                    split,
                )
        return dataset

    def _crossing_days(self, seconds: np.ndarray) -> set[int]:
        """Give the days of one turbine that a run of shutdown across a split cutoff touches."""
        first_us, step_us = unix_us(self.FIRST_DAY), STEP_S * US_PER_S
        runs = [(first_us + first * step_us, first_us + end * step_us) for first, end in _shutdown_intervals(seconds)]
        return self.SPLIT_CUTOFFS.crossing_days(runs)


def _create_record(values: np.ndarray, day_start_us: int, turbine: JsonMapping) -> Record:
    signals = tuple(
        Signal.from_values(
            [None if np.isnan(value) else float(value) for value in values[:, _VALUE_ORDER.index(name)]],
            spec=spec,
            name=name,
            time_axis=EVERY_TEN_MINUTES,
        )
        for name, _, _, spec in SIGNALS
    )
    return Record(
        sources=(Source(name="SCADA", signals=signals),),
        start_time=TimeOrigin(day_start_us),
        # The number of steps without a power value, so that a user can keep days with full coverage.
        metadata={**turbine, "power_missing_steps": int(np.isnan(values[:, _VALUE_ORDER.index("power")]).sum())},
        time_span=TimeInterval.micros(0, DAY_US),
    )


def _shutdown_intervals(seconds: np.ndarray) -> list[tuple[int, int]]:
    """Give the runs of periods with shutdown time, as first step and step after the last."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], seconds > 0, [False]])))
    return [(int(first), int(end)) for first, end in zip(edges[::2], edges[1::2], strict=True)]


def _label_day(
    dataset: TimeFDataset, record: Record, seconds: np.ndarray, wind: np.ndarray, split: Split | None
) -> None:
    """Add the shutdown annotations of one turbine day, and its tasks if the day has a split."""
    intervals = _shutdown_intervals(seconds)
    for first, end in intervals:
        record.annotate(
            Annotation(key=SHUTDOWN_KEY, value=int(seconds[first:end].sum()), span=_span_of_steps(first, end))
        )
    if split is not None:
        dataset.add_tasks(tasks=list(_iter_day_tasks(record, seconds, wind, intervals, split)))


def _span_of_steps(first: int, end: int) -> TimeInterval:
    return TimeInterval.micros(first * STEP_S * US_PER_S, end * STEP_S * US_PER_S)


def _downtime_60(seconds: np.ndarray, wind: np.ndarray) -> str | None:
    """Give the downtime label of a day.

    Only the shutdown at wind of 4 m/s or more counts. A day with an hour of it is ``downtime``. A
    day with none of it, and less than ten minutes of shutdown in periods without a wind value, is
    ``no_downtime``.

    Args:
        seconds: The seconds of shutdown of each of the 144 periods of the day.
        wind: The wind speed of each period, with NaN where the source has no value.

    Returns:
        ``downtime``, ``no_downtime``, or ``None`` if the day is in between.
    """
    windy_s = float(seconds[wind >= DOWNTIME_WIND_MS].sum())
    unread_s = float(seconds[np.isnan(wind)].sum())
    if windy_s >= DOWNTIME_S:
        return "downtime"
    return "no_downtime" if windy_s == 0 and unread_s < UNREAD_WIND_S else None


def _iter_day_tasks(
    record: Record, seconds: np.ndarray, wind: np.ndarray, intervals: list[tuple[int, int]], split: Split
) -> Iterator[Task]:
    """Create the tasks of one turbine day.

    Yields:
        The tasks, not yet added to a dataset.
    """
    yield ClassificationTask(
        inputs=(record,),
        targets=("stopped" if intervals else "running",),
        label_set=STOPPED,
        prompt=STOPPED_PROMPT,
        split=split,
    )
    yield ScalarPredictionTask(
        inputs=(record,),
        targets=(float(seconds.sum()) / 60,),
        unit="minute",
        target_name="stopped_minutes",
        prompt=STOPPED_MINUTES_PROMPT,
        split=split,
    )
    if intervals:
        yield TemporalLocalizationTask(
            inputs=(record,),
            targets=tuple(_span_of_steps(first, end) for first, end in intervals),
            prompt=SHUTDOWN_PERIODS_PROMPT,
            split=split,
        )
    downtime = _downtime_60(seconds, wind)
    if downtime is not None:
        yield ClassificationTask(
            inputs=(record,), targets=(downtime,), label_set=DOWNTIME_60, prompt=DOWNTIME_60_PROMPT, split=split
        )


def _read_turbines(metadata_path: Path) -> list[tuple[int, JsonMapping]]:
    """Read the table that describes the turbines.

    Returns:
        The station id and the facts of each turbine, in station order. The facts are the metadata
        of each Record of the turbine.

    Raises:
        TimeNetConnectorError: If the table has no turbine.
    """
    turbines: list[tuple[int, JsonMapping]] = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            turbines.append(
                (
                    int(row["Station ID"]),
                    {
                        "wind_farm": row["Wind Farm"],
                        "turbine": row["Turbine Name"],
                        "manufacturer": row["Manufacturer"],
                        "model": row["Model"],
                        "rated_power_kw": float(row["Rated power (kW)"]),
                        "hub_height_m": float(row["Hub Height (m)"]),
                        "rotor_diameter_m": float(row["Rotor Diameter (m)"]),
                        "latitude": float(row["Latitude"]),
                        "longitude": float(row["Longitude"]),
                    },
                )
            )
    if not turbines:
        raise TimeNetConnectorError(f"{metadata_path.name}: no turbine")
    return sorted(turbines, key=lambda item: item[0])


CONNECTOR = HillOfTowieConnector
