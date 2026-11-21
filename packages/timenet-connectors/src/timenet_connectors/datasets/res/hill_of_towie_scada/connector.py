"""Convert the Hill of Towie SCADA release into TimeF power forecasting tasks."""

from collections.abc import Iterator
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.csv as arrow_csv

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import ForecastingTask, Split, unix_us
from timenet_connectors.datasets.res.hill_of_towie_scada.windows import (
    LAST_MONTH_END,
    STEP,
    STEP_US,
    Turbine,
    WindowSeries,
    forecasting_tasks,
    month_periods,
    prepare_windows,
)
from timenet_connectors.download import Artifact, download_files


_FILES_URL = "https://zenodo.org/records/22662930/files"
_METADATA = "Hill_of_Towie_turbine_metadata.csv"
_METADATA_SHA256 = "dd40789c2bf2268da9d77890cc48617266f3ed9d13c5af2be6d6535671f261bc"
_ARCHIVE_SHA256 = {
    2016: "800bb3d399e6b79278758bbd95e234e364b062a9ff7643a7f9c045b0db9e9738",
    2017: "a49c11def85623b7f0537988debfa499cca15fc6ec971256051597d91224ab1f",
    2018: "4d59191bd12ad0e85eb4de44f2f8f7d8e2556452341e89cd194a816debc48761",
    2019: "24a9b9bfdf32558d80c518a11954d7230254f5ba384671d3df6ba3e53deb5b26",
    2020: "3e55de08be27e6a6e702a99bfcccd57e20abfa69e7c65d5901caf581f128f020",
    2021: "2284741792b4bfc7ca08234f49440dad4820bac5e73098011745a3f69e295c19",
    2022: "1b0f8657273ddbb8c4f84042a0bcb1c5eb1fb7833849b4c491f2b9f10604700d",
    2023: "e65f7acd65664584e5b72e4ad7ccd4c0f0af4ad68f618b8cf0b4e5c1f4a69521",
    2024: "b6bb540e3d2fcc2cff8756d96ef4155037b1cc41b284e6249b41054f4e50fbc3",
    2025: "cd439de7bf8e297dd8422d5c93eda04785c56e3861be8d2a79f403ac39b873ee",
    2026: "4bcfd706769369eacde59ea4f3a45bc3f9c0fb334c67fea75fcf3aa0c424eaea",
}
_POWER = ("wtc_ActPower_mean",)
_TELEMETRY = ("wtc_AcWindSp_mean", "wtc_GenRpm_mean")


@dataclass(frozen=True)
class HillOfTowieFiles:
    """Local annual SCADA archives and their turbine lookup."""

    archives: tuple[Path, ...]
    metadata: Path


def _read_turbines(path: Path) -> tuple[Turbine, ...]:
    """Read unique station identities and rated capacities."""
    turbines: dict[int, Turbine] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            try:
                turbine = Turbine(int(row["Station ID"]), row["Turbine Name"], int(row["Rated power (kW)"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise TimeNetConnectorError(f"{path.name}:{reader.line_num}: invalid turbine metadata") from exc
            if turbine.station in turbines or not turbine.name or turbine.rated_power_kw <= 0:
                raise TimeNetConnectorError(
                    f"{path.name}:{reader.line_num}: invalid or duplicate station {turbine.station}"
                )
            turbines[turbine.station] = turbine
    if not turbines:
        raise TimeNetConnectorError(f"{path.name}: no turbine metadata")
    return tuple(turbines[key] for key in sorted(turbines))


@dataclass(frozen=True)
class _Month:
    start: datetime
    end: datetime

    @property
    def steps(self) -> int:
        return (self.end - self.start) // STEP


def _months(year: int) -> Iterator[_Month]:
    """Yield UTC months in an archive, respecting the release end in April 2026."""
    for month in range(1, 13):
        start = datetime(year, month, 1, tzinfo=UTC)
        if start >= LAST_MONTH_END:
            return
        end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
        yield _Month(start, end)


def _read_table(
    archive: ZipFile,
    month: _Month,
    table: str,
    fields: tuple[str, ...],
    stations: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Load selected columns into a station-by-time array with source blanks preserved."""
    member = f"{table}_{month.start:%Y_%m}.csv"
    options = arrow_csv.ConvertOptions(
        include_columns=["TimeStamp", "StationId", *fields],
        column_types={"TimeStamp": pa.timestamp("us"), "StationId": pa.int64(), **dict.fromkeys(fields, pa.float64())},
    )
    try:
        with archive.open(member) as raw:
            data = arrow_csv.read_csv(raw, convert_options=options)
    except (KeyError, pa.ArrowInvalid) as exc:
        raise TimeNetConnectorError(f"{archive.filename}: unable to read {member}: {exc}") from exc

    timestamps = data["TimeStamp"]
    station_ids = data["StationId"]
    if timestamps.null_count or station_ids.null_count:
        raise TimeNetConnectorError(f"{member}: missing timestamp or station ID")
    relative_us = timestamps.to_numpy().astype(np.int64) - unix_us(month.start)
    # Source labels interval ends. A row at midnight belongs to the previous interval's month.
    owned = (relative_us > 0) & (relative_us <= month.steps * STEP_US)
    owned_us = relative_us[owned]
    if np.any(owned_us % STEP_US):
        raise TimeNetConnectorError(f"{member}: timestamp is outside the ten-minute grid")
    ids = station_ids.to_numpy()[owned]
    if unknown := np.setdiff1d(ids, stations).tolist():
        raise TimeNetConnectorError(f"{member}: unknown station IDs {unknown}")
    rows = np.searchsorted(stations, ids)
    steps = owned_us // STEP_US - 1
    if np.unique(rows * month.steps + steps).size != len(steps):
        raise TimeNetConnectorError(f"{member}: duplicate measurement")
    values = np.full((len(fields), stations.size, month.steps), np.nan)
    for field, channel in zip(fields, values, strict=True):
        measurements = data[field].to_numpy()[owned]
        if np.any(np.isinf(measurements)):
            raise TimeNetConnectorError(f"{member}: infinite measurement in {field}")
        channel[rows, steps] = measurements
    return values


def _month_series(
    archive: ZipFile, month: _Month, turbines: tuple[Turbine, ...], stations: NDArray[np.int64]
) -> Iterator[WindowSeries]:
    """Prepare the turbine records and eligible cutoffs for each part of a month."""
    power = _read_table(archive, month, "tblSCTurGrid", _POWER, stations)
    telemetry = _read_table(archive, month, "tblSCTurbine", _TELEMETRY, stations)
    values = np.concatenate((power, telemetry))
    for period in month_periods(month.start, month.end):
        start = (period.start - month.start) // STEP
        stop = (period.end - month.start) // STEP
        for index, turbine in enumerate(turbines):
            yield prepare_windows(values[:, index, start:stop], turbine, period)


class HillOfTowieSCADAConnector(BaseConnector[HillOfTowieFiles]):
    """Build power forecasting tasks from the pinned 2016 to 2026 annual SCADA archives."""

    # Override: the pinned source release has a fixed set of files.
    # ruff: ignore[no-self-use]
    async def download_async(self, cache_dir: Path) -> HillOfTowieFiles:
        """Cache the Zenodo archives and turbine lookup."""
        pins = [(f"{year}.zip", digest) for year, digest in _ARCHIVE_SHA256.items()]
        files = await download_files(
            [
                Artifact(f"{_FILES_URL}/{name}?download=1", cache_dir / name, sha256=digest)
                for name, digest in [*pins, (_METADATA, _METADATA_SHA256)]
            ],
            max_concurrency=5,
        )
        return HillOfTowieFiles(tuple(files[:-1]), files[-1])

    def convert(self, raw: HillOfTowieFiles, /) -> TimeFDataset:
        """Register the SCADA records and stream tasks over qualifying windows."""
        if len(raw.archives) != len(_ARCHIVE_SHA256):
            raise TimeNetConnectorError("Hill of Towie requires all 11 annual archives and turbine metadata")
        dataset = TimeFDataset(metadata=self.metadata())
        series: list[WindowSeries] = []
        try:
            turbines = _read_turbines(raw.metadata)
            stations = np.array([turbine.station for turbine in turbines], dtype=np.int64)
            for year, path in zip(_ARCHIVE_SHA256, raw.archives, strict=True):
                with ZipFile(path) as archive:
                    for month in _months(year):
                        for entry in _month_series(archive, month, turbines, stations):
                            dataset.add_record(record=entry.record)
                            series.append(entry)
        except (BadZipFile, OSError, UnicodeError, csv.Error) as exc:
            raise TimeNetConnectorError(f"Unable to read Hill of Towie source files: {exc}") from exc
        if {entry.period.split for entry in series if entry.cutoffs} != set(Split):
            raise TimeNetConnectorError("Hill of Towie SCADA must contain complete windows in all three splits")
        dataset.set_task_stream([ForecastingTask], partial(forecasting_tasks, series))
        return dataset


CONNECTOR = HillOfTowieSCADAConnector
