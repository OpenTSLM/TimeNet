"""Build the frozen AirAware forecast from the original SENAMHI CSV."""

# SPDX-License-Identifier: MIT

from collections import defaultdict, deque
from collections.abc import Iterator
import csv
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import ForecastingTask, Split, TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.download.http import download_http


SOURCE_URL = (
    "https://www.datosabiertos.gob.pe/sites/default/files/Monitoreo%20de%20los%20contaminantes%20"
    "del%20aire%20en%20Lima%20Metropolitana%20-%20%5BServicio%20Nacional%20de%20Meteorolog%C3%ADa"
    "%20e%20Hidrolog%C3%ADa%20del%20Per%C3%BA%20-%20SENAMHI%5D_1.csv"
)
STATION_FIELDS = ("ESTACION", "LONGITUD", "LATITUD", "ALTITUD", "DEPARTAMENTO", "PROVINCIA", "DISTRITO")
VALIDATION_START = datetime(2022, 3, 16, 8, tzinfo=UTC)
TEST_START = datetime(2023, 2, 25, 18, tzinfo=UTC)
HOUR = timedelta(hours=1)
HISTORY = 24
FUTURE = 6
AXIS = RegularAxis(period_us=Fraction(3_600_000_000))
PM25 = TimeSeriesSpec(
    spec_type="pm25",
    name="PM2.5",
    unit_value=ureg.microgram / ureg.meter**3,
    dtype="float64",
    missing=None,
)
type_reading = tuple[float, int]


def _key(fields: list[str | None], prefix: str = "") -> str:
    return sha256((prefix + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))).encode("utf-8")).hexdigest()


def _parse(data: bytes) -> dict[str, dict[datetime, type_reading | None]]:
    """Preserve the first equal report and leave disagreeing reports ambiguous."""
    stations: dict[str, dict[datetime, type_reading | None]] = defaultdict(dict)
    required = {*STATION_FIELDS, "FECHA", "HORA", "PM2_5"}
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            msg = "The CSV is missing one or more required SENAMHI fields"
            raise ValueError(msg)
        keys = {}
        for row_number, row in enumerate(rows, start=2):
            if row["DEPARTAMENTO"] != "LIMA" or row["PROVINCIA"] != "LIMA" or not row["ESTACION"].strip():
                continue
            try:
                value = float(row["PM2_5"])
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            clock = datetime.strptime(row["FECHA"] + row["HORA"].zfill(6), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            if clock.minute or clock.second:
                continue
            context = tuple(row[field] for field in STATION_FIELDS)
            if context not in keys:
                keys[context] = _key(list(context), "senamhi-lima-csv")
            series = stations[keys[context]]
            if clock not in series:
                series[clock] = (value, row_number)
            elif (previous := series[clock]) is not None and previous[0] != value:
                series[clock] = None
    return dict(stations)


def _split(start: datetime, end: datetime) -> Split | None:
    if end < VALIDATION_START:
        return Split.TRAIN
    if start >= VALIDATION_START and end < TEST_START:
        return Split.VALIDATION
    if start >= TEST_START:
        return Split.TEST
    return None


def _windows(
    series: dict[datetime, type_reading | None],
) -> Iterator[tuple[datetime, Split, tuple[type_reading, ...]]]:
    recent: deque[type_reading] = deque(maxlen=HISTORY + FUTURE)
    previous = None
    for tick, reading in sorted(series.items()):
        if previous is None or tick - previous != HOUR or reading is None:
            recent.clear()
        previous = tick
        if reading is None:
            continue
        recent.append(reading)
        if len(recent) == recent.maxlen:
            start = tick - (HISTORY + FUTURE - 1) * HOUR
            split = _split(start, tick)
            if split is not None:
                yield start, split, tuple(recent)


def _timestamp_key(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%d %H:%M:%S+00")


class LimaPM25Connector(BaseConnector[Path]):
    """Build the Lima retrospective PM2.5 benchmark from its pinned public export."""

    source_sha256 = "896ae8a106fb128460e81b3f58096f6b9c4d582ef28fffa46e5394c5875c40d4"

    async def download_async(self, cache_dir: Path) -> Path:
        """Resolve the pinned original, reusing a valid cached copy."""
        return await download_http(SOURCE_URL, cache_dir / "senamhi-lima.csv", sha256=self.source_sha256)

    def convert(self, path: Path) -> TimeFDataset:
        """Read one verified CSV and build chronological forecasting tasks."""
        content = path.read_bytes()
        if sha256(content).hexdigest() != self.source_sha256:
            msg = "SENAMHI source SHA-256 does not match the dataset card"
            raise ValueError(msg)
        dataset = TimeFDataset(metadata=self.metadata())
        for station, reports in sorted(_parse(content).items(), key=lambda item: self._series_key(item[0])):
            series_id = self._series_key(station)
            for start, split, observations in _windows(reports):
                history = observations[:HISTORY]
                origin = start + (HISTORY - 1) * HOUR
                end = origin + FUTURE * HOUR
                window_id = _key(
                    [
                        series_id,
                        _timestamp_key(origin),
                        _timestamp_key(start),
                        _timestamp_key(end),
                    ]
                )
                signal = Signal.from_values(
                    [value for value, _ in history],
                    name="pm25",
                    spec=PM25,
                    time_axis=AXIS,
                )
                record = dataset.add_record(
                    Record(
                        sources=[Source(name=station, signals=[signal])],
                        start_time=TimeOrigin(start),
                        metadata={
                            "station_key": station,
                            "source_rows": [row for _, row in history],
                            "unit_basis": "inferred",
                        },
                    )
                )
                dataset.add_task(
                    ForecastingTask(
                        inputs=[record],
                        targets=[value for value, _ in observations[HISTORY:]],
                        split=split,
                        metadata={
                            "window_id": window_id,
                            "series_id": series_id,
                            "origin_at": origin.isoformat(),
                            "target_start": (origin + HOUR).isoformat(),
                        },
                    )
                )
        return dataset

    @staticmethod
    def _series_key(station: str) -> str:
        return _key([station, None, "PM2_5", "PM2.5", "ug/m3"])


CONNECTOR = LimaPM25Connector
