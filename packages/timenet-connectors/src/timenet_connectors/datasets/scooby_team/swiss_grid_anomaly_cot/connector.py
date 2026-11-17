"""The Swiss Grid Anomaly-CoT connector: weekly Swiss power and weather windows with explained anomalies.

The connector downloads six yearly Swissgrid workbooks (2020-2025, quarter-hourly) and the hourly files
of twelve SwissMetNet stations, and recomputes every label in ``convert``. Nothing in the dataset comes
from a model: a linear baseline fitted on 2020-2023 gives the expected daily consumption, a robust
z-score flags the anomalous days, ordered calendar and weather rules give each one a cause, and fixed
templates write the rationales from the computed facts. See ``_pipeline`` for each step.

Each ISO week becomes one Record with nine hourly signals and three tasks: the cause of the largest
anomaly (classification), the anomalous days (temporal localization) and the net import/export
question (answer). The prompts give the dates but no holiday: a holiday has to be read from the
consumption curve. The holidays stay in the record as annotations and in the rationales.
"""

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

import pandas as pd

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    Split,
    TemporalLocalizationTask,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import Artifact, download_files

from ._pipeline import causes, meteoswiss, swissgrid
from ._pipeline.assemble import hourly_dataset
from ._pipeline.run import annotate
from ._pipeline.stations import STATION_CANTONS


SWISSGRID_PAGE = "https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html"
_SWISSGRID = {
    2020: ("a6deeecb-74e8-4d67-a8ad-8c53c9dfbdd3", "9ccbe5bff13b44a7436555b70c1a21b0a9e3f879b9ea423ec5fb16c7dd5ed7bc"),
    2021: ("26fa6b22-5e81-44eb-abbd-4280c59c3697", "ef940de1ecd58ddd51252f40ac07c80bb7abfee468ea353fefb94931e60ec9b6"),
    2022: ("e3af783a-6c00-40d9-bb3e-a630fb492dfb", "0922fbf418cbaae533d37cfb25f09b7afe637ebdbdb83328bb886c2857368267"),
    2023: ("32576046-3686-4a62-bc32-9b55a89a2142", "b86e0fab5834726a28545bd91dfeba279d751ad34ce2bf33cae0edb52117f0e7"),
    2024: ("4d4ca899-395d-4d3d-9958-5ace16ecbfbf", "3944594e167e6a50762e7c30ad81b3209fc2e9b719a36a9321b477bbaccf8605"),
    2025: ("bbbbadc6-3656-4dfe-b81e-bbb9ad01be79", "48ba9c57086fc0f00c96ba5cf7f449b0f51fd098451783bc4f3c36628a2944de"),
}
"""Year -> (Swissgrid DAM id, SHA-256). Swissgrid re-publishes revised files, so a new digest is a new
dataset version, never a silent change."""
_SMN = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
_STATION_SHA256 = {
    "BAS": "19d5ef8c633b31e586f0090f80f0571d23664ed0afa80b33b92e73ad9104090b",
    "BER": "efb0decf622589e2fdfbc0121c7026ceb18c9e15c7c1d5bb3217ecba15c79c24",
    "BUS": "9b95d63d069ae1ec78a864bba33f17e755a4c85741146529993b682e916e9428",
    "CHU": "cb6a33ce054705ae5a960d606b20336ccbcc0328878cfa9d8d1fcfbde4250cbf",
    "GVE": "0e51543ce10dbbf9561ffce0f0bd1308b54e9b2c483dea7ad2879828e24b3d84",
    "LUG": "ef92692a77655d355625be23066a80a0869a83f161e0cf8b8f9c4dddd9038834",
    "LUZ": "d91c62b800c1c2b727cff59b15e13b5dcf1fdb4d4a6acd0e857b14cffdccbf32",
    "NEU": "7ac287a49d964dcbf3b967e075012c8ffca9407cb562348e404bb58f4e7ba79e",
    "PUY": "13004d355300956a934ab6ecfa44f858209615a9034c48a0cf77f05975785f03",
    "SIO": "e81934655d96dbf74e0db708aee0fdc5ef5c686f51a64e2efec4cf7852b655fe",
    "SMA": "17fee315d2bb8b90815b6c9627825e135a2b5de64c7776756e53c4433f8c8906",
    "STG": "82bf289106cc7e170c1383930cc1108bf454f775e956e183baa496662fa63d7d",
}
"""SHA-256 of each station's ``h_historical_2020-2029`` file. MeteoSwiss rewrites that file once a year."""

HOURLY = RegularAxis.from_rate_hz(Fraction(1, 3600))
US_PER_HOUR = 3_600_000_000
POWER = TimeSeriesSpec(spec_type="electric_power", name="Electric power", unit_value=ureg.megawatt, dtype="float64")
TEMPERATURE = TimeSeriesSpec(
    spec_type="air_temperature", name="Air temperature 2 m", unit_value=ureg.degC, dtype="float64"
)
RADIATION = TimeSeriesSpec(
    spec_type="global_radiation", name="Global radiation", unit_value=ureg.watt / ureg.meter**2, dtype="float64"
)
GRID_SIGNALS = {
    "load_enduser_mw": "End-user consumption",
    "load_total_mw": "Total consumption",
    "production_mw": "Production",
    "net_import_at_mw": "Net import from Austria",
    "net_import_de_mw": "Net import from Germany",
    "net_import_fr_mw": "Net import from France",
    "net_import_it_mw": "Net import from Italy",
}
"""Column -> signal name of the Swissgrid source. Every value is an hourly mean in MW."""
WEATHER_SIGNALS = {"temp_c": ("Temperature", TEMPERATURE), "radiation_wm2": ("Global radiation", RADIATION)}
CAUSE = LabelSet(
    name="anomaly_cause",
    labels=tuple(Label(name=name, description=causes.CAUSE_DESCRIPTIONS[name]) for name in causes.CAUSES),
)
SPLITS = {"train": Split.TRAIN, "validation": Split.VALIDATION, "test": Split.TEST}


@dataclass(frozen=True)
class RawFiles:
    """The paths of the downloaded source files."""

    swissgrid: dict[int, Path]
    stations: dict[str, Path]

    def read_hourly(self) -> pd.DataFrame:
        """Parse every source file into the aligned hourly national table.

        Returns:
            One row per UTC hour, 2020-2025.
        """
        quarters = pd.concat([swissgrid.read_year(path, year) for year, path in sorted(self.swissgrid.items())])
        stations = {code: meteoswiss.read_station(path, code) for code, path in self.stations.items()}
        return hourly_dataset(quarters, stations)


def _signals(window: pd.DataFrame) -> tuple[Source, Source]:
    grid = tuple(
        Signal.from_values(window[column].to_numpy(), spec=POWER, name=name, time_axis=HOURLY)
        for column, name in GRID_SIGNALS.items()
    )
    weather = tuple(
        Signal.from_values(window[column].to_numpy(), spec=spec, name=name, time_axis=HOURLY)
        for column, (name, spec) in WEATHER_SIGNALS.items()
    )
    return (
        Source(name="Swissgrid, Swiss control block", signals=grid),
        Source(name="MeteoSwiss SwissMetNet, population-weighted mean of 12 stations", signals=weather),
    )


def _day_interval(window: pd.DataFrame, date: str, signals: tuple[Signal, ...] | None) -> TimeInterval:
    """The hours of one local day, as an interval of the window's record."""
    hours = (window.local_date == pd.Timestamp(date)).to_numpy().nonzero()[0]
    return TimeInterval.micros(int(hours[0]) * US_PER_HOUR, (int(hours[-1]) + 1) * US_PER_HOUR, time_series=signals)


def add_window(dataset: TimeFDataset, window: pd.DataFrame, item: dict) -> None:
    """Add one week's Record and its three tasks to the dataset.

    Args:
        dataset: The dataset under construction.
        window: The hourly rows of the week, in time order.
        item: The week's facts, prompts and rationales from the pipeline.
    """
    facts = item["facts"]
    grid, weather = _signals(window)
    record = dataset.add_record(
        record=Record(
            sources=(grid, weather),
            start_time=TimeOrigin(window.index[0].to_pydatetime()),
            metadata={"window_id": item["window_id"]},
        )
    )
    load = grid.signals[0]
    anomaly_inputs = (load, *weather.signals)
    exchange_inputs = (*grid.signals[3:], grid.signals[2], grid.signals[1])
    record.add_annotations(
        [
            Annotation(key="window_id", value=item["window_id"]),
            Annotation(key="facts", value=json.dumps(facts, ensure_ascii=False)),
            *(
                Annotation(key="public_holiday", value=day["holiday"], span=_day_interval(window, day["date"], None))
                for day in facts["days"]
                if "holiday" in day
            ),
        ]
    )
    # An anomaly concerns the consumption, so it is attached to that signal.
    for day in facts["anomalous_days"]:
        load.annotate(Annotation(key="anomaly", value=day["cause"], span=_day_interval(window, day["date"], None)))
    whole = len(window) * US_PER_HOUR
    split = SPLITS[item["split"]]
    dataset.add_tasks(
        tasks=[
            ClassificationTask(
                inputs=(record,),
                scope=TimeInterval.micros(0, whole, time_series=anomaly_inputs),
                prompt=item["prompts"]["anomaly_cause"],
                targets=(item["label"],),
                label_set=CAUSE,
                rationale=item["rationales"]["anomaly_cause"],
                split=split,
            ),
            TemporalLocalizationTask(
                inputs=(record,),
                scope=TimeInterval.micros(0, whole, time_series=anomaly_inputs),
                prompt=item["prompts"]["anomaly_days"],
                targets=tuple(_day_interval(window, date, (load,)) for date in item["anomalous_dates"]),
                mode=LocalizationMode.EXHAUSTIVE,
                rationale=item["rationales"]["anomaly_days"],
                split=split,
            ),
            AnswerTask(
                inputs=(record,),
                scope=TimeInterval.micros(0, whole, time_series=exchange_inputs),
                prompt=item["prompts"]["exchange"],
                targets=(item["exchange_answer"],),
                rationale=item["rationales"]["exchange"],
                split=split,
            ),
        ]
    )


def build_dataset(dataset: TimeFDataset, hourly: pd.DataFrame) -> TimeFDataset:
    """Annotate the hourly table and add one Record per ISO week to the dataset.

    Args:
        dataset: An empty dataset carrying the card's metadata.
        hourly: The aligned hourly table from :meth:`RawFiles.read_hourly`.

    Returns:
        The populated dataset.
    """
    result = annotate(hourly)
    for item in result.windows:
        week = pd.date_range(item["facts"]["start_local"], item["facts"]["end_local"], freq="D")
        add_window(dataset, hourly[hourly.local_date.isin(week)], item)
    return dataset


class SwissGridAnomalyCotConnector(BaseConnector[RawFiles]):
    """Connector for the Swiss Grid Anomaly-CoT dataset."""

    async def download_async(self, cache_dir: Path) -> list[RawFiles]:  # ruff: ignore[no-self-use]
        """Download the six Swissgrid workbooks and the twelve station files, checking each digest.

        Args:
            cache_dir: Directory for the downloaded files.

        Returns:
            One reference holding every path.
        """
        grid = {
            year: Artifact(
                f"https://www.swissgrid.ch/dam/jcr:{dam_id}/EnergieUebersichtCH-{year}.xlsx",
                cache_dir / f"EnergieUebersichtCH-{year}.xlsx",
                sha256=digest,
            )
            for year, (dam_id, digest) in _SWISSGRID.items()
        }
        stations = {
            code: Artifact(
                f"{_SMN}/{code.lower()}/ogd-smn_{code.lower()}_h_historical_2020-2029.csv",
                cache_dir / f"ogd-smn_{code.lower()}_h_historical_2020-2029.csv",
                sha256=_STATION_SHA256[code],
            )
            for code in STATION_CANTONS
        }
        paths = await download_files([*grid.values(), *stations.values()])
        n_grid = len(grid)
        return [
            RawFiles(
                swissgrid=dict(zip(grid, paths[:n_grid], strict=True)),
                stations=dict(zip(stations, paths[n_grid:], strict=True)),
            )
        ]

    def convert(self, raw_refs: list[RawFiles]) -> TimeFDataset:
        """Parse the sources, recompute every label and build one Record and three tasks per week.

        Args:
            raw_refs: The reference from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        (files,) = raw_refs
        return build_dataset(TimeFDataset(metadata=self.metadata()), files.read_hourly())


CONNECTOR = SwissGridAnomalyCotConnector
