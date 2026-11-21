"""Convert Hill of Towie ten-minute SCADA into turbine power forecasts."""

from collections.abc import Iterator, Sequence
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from io import TextIOWrapper
from math import isfinite
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import ForecastingTask, Split, TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


_FILES_URL = "https://zenodo.org/records/22662930/files"
_ARCHIVE = "2026.zip"
_ARCHIVE_SHA256 = "4bcfd706769369eacde59ea4f3a45bc3f9c0fb334c67fea75fcf3aa0c424eaea"
_METADATA = "Hill_of_Towie_turbine_metadata.csv"
_METADATA_SHA256 = "dd40789c2bf2268da9d77890cc48617266f3ed9d13c5af2be6d6535671f261bc"

_SPLITS = {1: Split.TRAIN, 2: Split.TRAIN, 3: Split.VALIDATION, 4: Split.TEST}
_INTERVAL = timedelta(minutes=10)
_AXIS = RegularAxis.from_rate_hz(Fraction(1, 600))
_CONTEXT_STEPS = 36
_FORECAST_STEPS = 6
_POWER = TimeSeriesSpec(spec_type="active_power", name="Active power", unit_value=ureg.kilowatt, dtype="float64")
_WIND = TimeSeriesSpec(
    spec_type="wind_speed", name="Active wind speed", unit_value=ureg.meter / ureg.second, dtype="float64"
)
_RPM = TimeSeriesSpec(spec_type="generator_speed", name="Generator speed", unit_value=ureg.rpm, dtype="float64")

_ScadaKey = tuple[int, datetime]


@dataclass(frozen=True)
class HillOfTowieFiles:
    """Local source archive and turbine metadata."""

    archive: Path
    metadata: Path


@dataclass(frozen=True)
class _Turbine:
    name: str
    rated_power_kw: int


def _read_turbines(path: Path) -> dict[int, _Turbine]:
    """Read station IDs and static turbine metadata from the published lookup."""
    turbines: dict[int, _Turbine] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            try:
                station = int(row["Station ID"])
                turbine = _Turbine(row["Turbine Name"], int(row["Rated power (kW)"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise TimeNetConnectorError(f"{path.name}:{reader.line_num}: invalid turbine metadata") from exc
            if station in turbines or not turbine.name or turbine.rated_power_kw <= 0:
                raise TimeNetConnectorError(f"{path.name}:{reader.line_num}: invalid or duplicate station {station}")
            turbines[station] = turbine
    if not turbines:
        raise TimeNetConnectorError(f"{path.name}: no turbine metadata")
    return turbines


def _read_measurements(
    archive: ZipFile, month: int, table: str, columns: tuple[str, ...]
) -> dict[_ScadaKey, tuple[float, ...]]:
    """Read complete measurements for the month owning each UTC sampling interval."""
    member = f"{table}_2026_{month:02}.csv"
    values: dict[_ScadaKey, tuple[float, ...]] = {}
    seen: set[_ScadaKey] = set()
    with archive.open(member) as raw, TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"StationId", "TimeStamp", *columns}
        if missing := required.difference(reader.fieldnames or ()):
            raise TimeNetConnectorError(f"{member}: missing columns {sorted(missing)}")
        for row in reader:
            try:
                station = int(row["StationId"])
                timestamp = datetime.fromisoformat(row["TimeStamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                timestamp = timestamp.astimezone(UTC)
                # The publisher assigns a boundary interval to the month in which it starts.
                interval_start = timestamp - _INTERVAL
                if (interval_start.year, interval_start.month) != (2026, month):
                    continue
                if timestamp.minute % 10 or timestamp.second or timestamp.microsecond:
                    raise ValueError("timestamp is outside the ten-minute grid")
                fields = tuple(row[column].strip() for column in columns)
                if any(field.casefold() in {"", "nan"} for field in fields):
                    measurement = None
                else:
                    measurement = tuple(float(field) for field in fields)
                    if not all(isfinite(value) for value in measurement):
                        raise ValueError("non-finite measurement")
            except (AttributeError, TypeError, ValueError) as exc:
                raise TimeNetConnectorError(f"{member}:{reader.line_num}: invalid SCADA row") from exc
            key = (station, timestamp)
            if key in seen:
                raise TimeNetConnectorError(f"{member}:{reader.line_num}: duplicate measurement for {key}")
            seen.add(key)
            if measurement is not None:
                values[key] = measurement
    return values


@dataclass(frozen=True)
class _ForecastWindow:
    station: int
    cutoff: datetime
    history: tuple[tuple[float, ...], ...]
    future_power: tuple[float, ...]


def _windows(
    power: dict[_ScadaKey, tuple[float, ...]], telemetry: dict[_ScadaKey, tuple[float, ...]]
) -> Iterator[_ForecastWindow]:
    """Yield hourly cutoffs with uninterrupted, complete inputs and future power."""
    for station, cutoff in sorted(power):
        if cutoff.minute:
            continue
        history_keys = [(station, cutoff - step * _INTERVAL) for step in range(_CONTEXT_STEPS - 1, -1, -1)]
        future_keys = [(station, cutoff + step * _INTERVAL) for step in range(1, _FORECAST_STEPS + 1)]
        if all(key in power and key in telemetry for key in history_keys) and all(key in power for key in future_keys):
            history = tuple(power[key] + telemetry[key] for key in history_keys)
            future = tuple(power[key][0] for key in future_keys)
            yield _ForecastWindow(station, cutoff, history, future)


def _signal(spec: TimeSeriesSpec, values: Sequence[float]) -> Signal:
    """Attach the imported values as a lazily materialized regular series."""
    return Signal.from_loader(
        spec=spec,
        name=spec.spec_type,
        time_axis=_AXIS,
        loader=lambda: pa.array(values, type=pa.float64()),
        n_values=len(values),
        null_count=0,
    )


def _add_forecast(dataset: TimeFDataset, window: _ForecastWindow, turbine: _Turbine, split: Split) -> None:
    """Register one history record and its separate power-only forecast target."""
    first_context_end = window.cutoff - (_CONTEXT_STEPS - 1) * _INTERVAL
    first_target_end = window.cutoff + _INTERVAL
    signals = tuple(
        _signal(spec, values)
        for spec, values in zip((_POWER, _WIND, _RPM), zip(*window.history, strict=True), strict=True)
    )
    context = dataset.add_record(
        record=Record(
            start_time=TimeOrigin(first_context_end),
            metadata={
                "turbine_id": turbine.name,
                "station_id": window.station,
                "rated_power_kw": turbine.rated_power_kw,
                "window_start_utc": (first_context_end - _INTERVAL).isoformat(),
                "forecast_cutoff_utc": window.cutoff.isoformat(),
            },
            sources=(Source(name="SCADA", signals=signals),),
        ),
    )
    target = dataset.add_record(
        record=Record(
            start_time=TimeOrigin(first_target_end),
            metadata={"turbine_id": turbine.name, "station_id": window.station},
            sources=(Source(name="SCADA", signals=(_signal(_POWER, window.future_power),)),),
        ),
    )
    dataset.add_task(
        task=ForecastingTask(
            inputs=(context,),
            targets=(target,),
            prompt="Forecast mean active power in kW for the next six ten-minute intervals.",
            split=split,
        ),
    )


class HillOfTowieSCADAConnector(BaseConnector[HillOfTowieFiles]):
    """Build hourly turbine power forecasting tasks from the pinned 2026 SCADA archive."""

    # Override: the source release and artifact paths are fixed.
    # ruff: ignore[no-self-use]
    async def download_async(self, cache_dir: Path) -> HillOfTowieFiles:
        """Cache the two version-pinned Zenodo artifacts."""
        archive, metadata = await download_files(
            [
                Artifact(f"{_FILES_URL}/{_ARCHIVE}?download=1", cache_dir / _ARCHIVE, sha256=_ARCHIVE_SHA256),
                Artifact(f"{_FILES_URL}/{_METADATA}?download=1", cache_dir / _METADATA, sha256=_METADATA_SHA256),
            ],
        )
        return HillOfTowieFiles(archive, metadata)

    def convert(self, raw: HillOfTowieFiles, /) -> TimeFDataset:
        """Join monthly SCADA tables by station and time and build chronological splits."""
        dataset = TimeFDataset(metadata=self.metadata())
        try:
            turbines = _read_turbines(raw.metadata)
            with ZipFile(raw.archive) as archive:
                for month, split in _SPLITS.items():
                    power = _read_measurements(archive, month, "tblSCTurGrid", ("wtc_ActPower_mean",))
                    telemetry = _read_measurements(
                        archive, month, "tblSCTurbine", ("wtc_AcWindSp_mean", "wtc_GenRpm_mean")
                    )
                    if unknown := {station for station, _ in power.keys() | telemetry.keys()}.difference(turbines):
                        raise TimeNetConnectorError(f"SCADA 2026-{month:02}: unknown station IDs {sorted(unknown)}")
                    for window in _windows(power, telemetry):
                        _add_forecast(dataset, window, turbines[window.station], split)
        except (BadZipFile, OSError, KeyError, UnicodeError, csv.Error) as exc:
            raise TimeNetConnectorError(f"Unable to read Hill of Towie source files: {exc}") from exc
        if not all((dataset.get_train(), dataset.get_validation(), dataset.get_test())):
            raise TimeNetConnectorError("Hill of Towie SCADA must contain complete windows in all three splits")
        return dataset


CONNECTOR = HillOfTowieSCADAConnector
