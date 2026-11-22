"""Connector for the ``forecasttrust/india-rainfall-reliability`` dataset.

The dataset asks whether a day-ahead rainfall forecast will fail, before the rain is observed. The
forecasts are ECMWF IFS runs as originally issued, from the Open-Meteo Previous Runs API. The
verifying rainfall is ERA5 reanalysis from the Open-Meteo archive API. Both are fetched as CSV for a
fixed set of cities and a fixed date range, and every file is pinned by SHA-256, so a later revision
of the upstream data fails the build instead of silently changing the dataset.
"""

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    ScalarPredictionTask,
    TimeSeriesSpec,
    ureg,
)
from timenet.types.splits import Split
from timenet_connectors.download import Artifact, download_files


START_DATE = "2024-02-05"
END_DATE = "2025-11-15"
WINDOW_DAYS = 14
# The observation of valid day D is trusted only through day D - OBS_LAG_DAYS when the day-1 run is
# issued, so no input contains rain from the last two days before the valid day.
OBS_LAG_DAYS = 2
FAILURE_FLOOR_MM = 5.0
FAILURE_FRACTION = 0.5
TRAIN_END = date(2024, 12, 31)
VALIDATION_END = date(2025, 5, 31)

_MODEL = "ecmwf_ifs025"
_TIMEZONE = "Asia/Kolkata"
_FORECAST_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_VARIABLES = (
    "precipitation_previous_day1",
    "precipitation_previous_day2",
    "precipitation_previous_day3",
)
_HOURS_PER_DAY = 24

# City name -> (latitude, longitude). Open-Meteo snaps each point to its model grid.
CITIES: dict[str, tuple[float, float]] = {
    "Mumbai": (19.07, 72.88),
    "Pune": (18.52, 73.86),
    "Nashik": (20.00, 73.79),
    "Nagpur": (21.15, 79.09),
    "Delhi": (28.61, 77.21),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Bengaluru": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48),
    "Ahmedabad": (23.02, 72.57),
    "Guwahati": (26.14, 91.74),
    "Kochi": (9.93, 76.27),
}

# SHA-256 of each downloaded CSV, keyed by file name.
_SHA256: dict[str, str] = {
    "mumbai_ifs_previous_runs.csv": ("97077f2f91ec026d82f4f4f7dc58bac72f33e41f5c8820d20e08738d6ce7de4f"),
    "mumbai_era5_daily.csv": ("691e739ef30c64a1662fce5b16859e05daa777e247cfa4b908b3fab33aa389a9"),
    "pune_ifs_previous_runs.csv": ("935246be9b82a36dfaf2333377371364a0e68e4b8fdb0067012a7674ea9ab6e8"),
    "pune_era5_daily.csv": ("cd425e12b3a66e57ddf12526e16a5af8025fce23e5af57bb82a9f25df391333b"),
    "nashik_ifs_previous_runs.csv": ("bd7f2928a3c242d139db9402b6becdd5a04635f8868f9e908474a1fcf34a9404"),
    "nashik_era5_daily.csv": ("d4218500909cf758b9692ae7e3d321d54f36f95513400c489e4502a78f0cb2a1"),
    "nagpur_ifs_previous_runs.csv": ("461b7259d93e1a1dae6f88ad785c5533dcfdc1531a868b27da89cd3488c97cca"),
    "nagpur_era5_daily.csv": ("5e4fddc3b90335ab99ab15393e67d0ddf6fa814c466e151cb4be90e021f63ebb"),
    "delhi_ifs_previous_runs.csv": ("fe495eb068fa7663f1de664ef9afdf2f545f8da0e03b3e6edae052fdd3686f30"),
    "delhi_era5_daily.csv": ("2036fa8b1392f360b77ed91f728ff8494b43b73ae364a5b3f73b2c35d335098b"),
    "chennai_ifs_previous_runs.csv": ("60d72319013407e7451f86cdc4aa53affb9258511797765dd64481bcbcffe60f"),
    "chennai_era5_daily.csv": ("af1b671132d69996f0327c60fad335840677e0af8bc2ecae28ef46b41bcae7a0"),
    "kolkata_ifs_previous_runs.csv": ("fac850a13a18cffcef3fe16c67378394b34897e310430058e277836ec2306807"),
    "kolkata_era5_daily.csv": ("daaa784b70623cfb58da06f38aba9e1697b657b2cb4dbf4fb98d722d297602ce"),
    "bengaluru_ifs_previous_runs.csv": ("5ad2e615dcae957f0fb75e736a2cceb54427052d252e2baa329a882be324f36a"),
    "bengaluru_era5_daily.csv": ("af1055d87201379c8dea4cd254a0bab237e098b42c1418a79fee849532e5b516"),
    "hyderabad_ifs_previous_runs.csv": ("1e3d9020ce1e88ebf3bfef54861ca1e45b5001f59a51bd8ab74de1c25eacbadc"),
    "hyderabad_era5_daily.csv": ("98760ab428833a2bd4689a28b42849fd7248d1dd4058b8c9d9ab3fe27b903739"),
    "ahmedabad_ifs_previous_runs.csv": ("f1c537893ebf3ed301f5452a73827d6df17d8da284954a42928c80ce8b06d12d"),
    "ahmedabad_era5_daily.csv": ("7ca32d556c2182f72f50fac99f21272c2543ce6626dce538a629a1387e230449"),
    "guwahati_ifs_previous_runs.csv": ("80b376c9ec7018fde30bc6af7698b41bbec4c937e10f7d823f2ccda34feda95d"),
    "guwahati_era5_daily.csv": ("64ab1298bf6b58ac86d79bcd94a508f49284b4f78ae08f00fcc54b79308c9b96"),
    "kochi_ifs_previous_runs.csv": ("06e65891c6312a940951407b4f4a0a55e5f1d14342b8339651dddbead4611f16"),
    "kochi_era5_daily.csv": ("4dc5617d7e2b2b2a7a85805b362af0f61a7311424d8443619fefb3f98bc5c28a"),
}

_DAILY_AXIS = RegularAxis.from_rate_hz(Fraction(1, 86400))
_LABELS = LabelSet(
    name="forecast_reliability",
    labels=(
        Label(name="reliable", description="The day-1 forecast error stays within the failure threshold."),
        Label(
            name="failure",
            description="The day-1 absolute error exceeds the larger of 5 mm and half the forecast.",
        ),
    ),
)


def _spec(spec_type: str, name: str) -> TimeSeriesSpec:
    """Build the spec of one daily rainfall accumulation in millimetres.

    Args:
        spec_type: The modality tag.
        name: The display name.

    Returns:
        The spec.
    """
    return TimeSeriesSpec(spec_type=spec_type, name=name, unit_value=ureg.millimeter)


_FORECAST_SPECS = tuple(
    _spec(f"ifs_precip_forecast_day{lead}", f"IFS daily rainfall forecast, issued {lead} day(s) ahead")
    for lead in (1, 2, 3)
)
_OBSERVED_SPEC = _spec("era5_precip_observed", "ERA5 daily rainfall")


@dataclass(frozen=True)
class CityFiles:
    """The downloaded CSV files of one city."""

    city: str
    latitude: float
    longitude: float
    forecast: Path
    observed: Path


@dataclass(frozen=True)
class _Sample:
    """The inputs of one valid day, bundled for :meth:`IndiaRainfallReliabilityConnector._add_sample`."""

    ref: CityFiles
    valid_day: date
    window: list[date]
    """The days of the forecast window, ending on ``valid_day``."""
    history: list[date]
    """The days with a verified observation, ending ``OBS_LAG_DAYS`` before ``valid_day``."""
    forecasts: dict[date, tuple[float, float, float]]
    """The day-1, day-2 and day-3 totals of each complete day."""
    observed: dict[date, float]
    """The observed totals of each complete day."""


def _file_names(city: str) -> tuple[str, str]:
    """Return the cache file names of a city's forecast and observation CSV.

    Args:
        city: A key of :data:`CITIES`.

    Returns:
        The forecast and observation file names.
    """
    slug = city.lower()
    return f"{slug}_ifs_previous_runs.csv", f"{slug}_era5_daily.csv"


def _urls(city: str) -> tuple[str, str]:
    """Build the request URLs for a city.

    Args:
        city: A key of :data:`CITIES`.

    Returns:
        The forecast URL and the observation URL.
    """
    latitude, longitude = CITIES[city]
    common = {"latitude": latitude, "longitude": longitude, "timezone": _TIMEZONE, "format": "csv"}
    window = {"start_date": START_DATE, "end_date": END_DATE}
    forecast = urlencode(
        {**common, **window, "models": _MODEL, "hourly": ",".join(_FORECAST_VARIABLES)},
        safe=",",
    )
    observed = urlencode(
        {**common, **window, "models": "era5", "daily": "precipitation_sum"},
        safe=",",
    )
    return f"{_FORECAST_API}?{forecast}", f"{_ARCHIVE_API}?{observed}"


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read an Open-Meteo CSV, which puts a location block before the data table.

    Args:
        path: The CSV file.

    Returns:
        The column names of the data table and its rows.

    Raises:
        TimeNetConnectorError: If the file has no data table.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    for index, line in enumerate(lines):
        if line and line[0] == "time":
            return line, [row for row in lines[index + 1 :] if row]
    raise TimeNetConnectorError(f"{path.name}: no data table starting with a 'time' column")


def _daily_forecasts(path: Path) -> dict[date, tuple[float, float, float]]:
    """Sum hourly forecasts into IST calendar days, keeping only days with all 24 hours present.

    A day with any missing hour is left out and never filled.

    Args:
        path: The hourly forecast CSV.

    Returns:
        For each complete day, its day-1, day-2 and day-3 forecast totals in mm.

    Raises:
        TimeNetConnectorError: If the file lacks an expected column.
    """
    header, rows = _read_csv(path)
    columns = [f"{name} (mm)" for name in _FORECAST_VARIABLES]
    try:
        positions = [header.index(column) for column in columns]
    except ValueError as exc:
        raise TimeNetConnectorError(f"{path.name}: missing forecast column ({exc})") from exc
    hours: dict[date, list[list[float]]] = {}
    for row in rows:
        day = date.fromisoformat(row[0][:10])
        hours.setdefault(day, []).append([float(row[position]) for position in positions])
    complete: dict[date, tuple[float, float, float]] = {}
    for day, values in hours.items():
        table = np.asarray(values, dtype=np.float64)
        if len(values) == _HOURS_PER_DAY and not np.isnan(table).any():
            totals = table.sum(axis=0)
            complete[day] = (float(totals[0]), float(totals[1]), float(totals[2]))
    return complete


def _daily_observations(path: Path) -> dict[date, float]:
    """Read daily ERA5 rainfall totals, leaving out missing days.

    Args:
        path: The daily observation CSV.

    Returns:
        The rainfall in mm of each day that has a value.

    Raises:
        TimeNetConnectorError: If the file lacks the rainfall column.
    """
    header, rows = _read_csv(path)
    try:
        position = header.index("precipitation_sum (mm)")
    except ValueError as exc:
        raise TimeNetConnectorError(f"{path.name}: missing observation column") from exc
    result: dict[date, float] = {}
    for row in rows:
        value = float(row[position])
        if not np.isnan(value):
            result[date.fromisoformat(row[0])] = value
    return result


def _split_of(day: date) -> Split:
    """Assign a valid day to its chronological partition.

    Args:
        day: The valid day of the forecast.

    Returns:
        The split.
    """
    if day <= TRAIN_END:
        return Split.TRAIN
    if day <= VALIDATION_END:
        return Split.VALIDATION
    return Split.TEST


class IndiaRainfallReliabilityConnector(BaseConnector[CityFiles]):
    """Connector for the ``forecasttrust/india-rainfall-reliability`` dataset."""

    async def download_async(self, cache_dir: Path) -> list[CityFiles]:  # ruff: ignore[no-self-use]
        """Download the pinned forecast and observation CSV of every city.

        Args:
            cache_dir: Where to store the files.

        Returns:
            One reference per city.

        Raises:
            TimeNetConnectorError: If a file has no pinned checksum.
        """
        artifacts: list[Artifact] = []
        refs: list[CityFiles] = []
        for city, (latitude, longitude) in CITIES.items():
            forecast_name, observed_name = _file_names(city)
            forecast_url, observed_url = _urls(city)
            for name in (forecast_name, observed_name):
                if name not in _SHA256:
                    raise TimeNetConnectorError(f"no pinned SHA-256 for {name}")
            artifacts.append(Artifact(forecast_url, cache_dir / forecast_name, sha256=_SHA256[forecast_name]))
            artifacts.append(Artifact(observed_url, cache_dir / observed_name, sha256=_SHA256[observed_name]))
            refs.append(CityFiles(city, latitude, longitude, cache_dir / forecast_name, cache_dir / observed_name))
        # A low concurrency stays within the API's free-tier rate limits.
        await download_files(artifacts, max_concurrency=2)
        return refs

    def convert(self, raw_refs: list[CityFiles]) -> TimeFDataset:
        """Build one record per city and valid day, with a classification and a scalar task.

        A sample exists only if every forecast day of its window and every verified observation day
        is complete. Samples with a gap are skipped, not filled.

        Args:
            raw_refs: The city files from :meth:`download`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            forecasts = _daily_forecasts(ref.forecast)
            observed = _daily_observations(ref.observed)
            for valid_day in sorted(set(forecasts) & set(observed)):
                window = [valid_day - timedelta(days=lag) for lag in range(WINDOW_DAYS - 1, -1, -1)]
                # Days up to valid_day - OBS_LAG_DAYS have a verified observation at issue time.
                history = window[: WINDOW_DAYS - OBS_LAG_DAYS]
                if not all(day in forecasts for day in window) or not all(day in observed for day in history):
                    continue
                sample = _Sample(ref, valid_day, window, history, forecasts, observed)
                self._add_sample(dataset, sample)
        return dataset

    @staticmethod
    def _add_sample(dataset: TimeFDataset, sample: "_Sample") -> None:
        """Add the record and tasks of one valid day.

        Args:
            dataset: The dataset to extend.
            sample: The city, the days and the daily totals of the sample.
        """
        ref, valid_day = sample.ref, sample.valid_day
        window, history = sample.window, sample.history
        forecasts, observed = sample.forecasts, sample.observed
        forecast_signals = tuple(
            Signal.from_values(
                np.asarray([forecasts[day][lead] for day in window], dtype=np.float32),
                spec=_FORECAST_SPECS[lead],
                name=f"forecast_day{lead + 1}",
                time_axis=_DAILY_AXIS,
            )
            for lead in range(3)
        )
        observed_signal = Signal.from_values(
            np.asarray([observed[day] for day in history], dtype=np.float32),
            spec=_OBSERVED_SPEC,
            name="observed_verified",
            time_axis=_DAILY_AXIS,
        )
        record = Record(
            sources=(
                Source(name="ECMWF IFS via Open-Meteo Previous Runs", signals=forecast_signals),
                Source(name="ERA5 via Open-Meteo Archive", signals=(observed_signal,)),
            ),
            metadata={"city": ref.city, "valid_date": valid_day.isoformat()},
        )
        dataset.add_record(record=record)
        record.add_annotations(
            [
                Annotation(key="city", value=ref.city),
                Annotation(key="latitude", value=ref.latitude, unit="degree"),
                Annotation(key="longitude", value=ref.longitude, unit="degree"),
                Annotation(key="valid_date", value=valid_day.isoformat()),
            ]
        )
        issued = forecasts[valid_day][0]
        error = abs(observed[valid_day] - issued)
        failed = error > max(FAILURE_FLOOR_MM, FAILURE_FRACTION * issued)
        split = _split_of(valid_day)
        prompt = (
            f"{WINDOW_DAYS} daily steps end on the valid day. The forecast channels cover every step; "
            f"the observed channel stops {OBS_LAG_DAYS} days before the valid day. "
            "Will the day-1 rainfall forecast for the valid day fail?"
        )
        dataset.add_tasks(
            tasks=[
                ClassificationTask(
                    inputs=(record,),
                    prompt=prompt,
                    targets=("failure" if failed else "reliable",),
                    label_set=_LABELS,
                    split=split,
                ),
                ScalarPredictionTask(
                    inputs=(record,),
                    prompt="How large will the absolute error of the day-1 rainfall forecast be?",
                    targets=(error,),
                    unit="mm",
                    target_name="absolute_error",
                    split=split,
                ),
            ]
        )


CONNECTOR = IndiaRainfallReliabilityConnector
