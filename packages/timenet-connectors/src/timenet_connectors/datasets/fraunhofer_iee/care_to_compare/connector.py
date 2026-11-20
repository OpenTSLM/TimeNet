"""The CARE to Compare connector: wind turbine SCADA data with labeled pre-fault anomaly events.

Each event file of the release becomes one Record. The Record holds one turbine's 10-minute SCADA
signals (one Signal per sensor statistic) and its operating status, over about a year of training
data followed by a prediction period. The event table becomes annotations: the training and
prediction periods, the event window, and the event's label and root-cause note. Each Record gets a
ClassificationTask that asks whether the prediction period holds an anomaly. Each anomaly event also
gets a TemporalLocalizationTask for its window and, for wind farms A and B, a ClassificationTask for
the faulty component.

:mod:`~timenet_connectors.datasets.fraunhofer_iee.care_to_compare.source` parses the files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
from typing import Self

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    MissingReason,
    MissingValues,
    TemporalLocalizationTask,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.source import (
    PERIOD_US,
    EventInfo,
    SensorDescription,
    read_event_info,
    read_event_table,
    read_sensor_descriptions,
    split_feature_column,
)
from timenet_connectors.download import ensure_archive


ZENODO_URL = "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1"
"""Version 6 of the Zenodo record, the release with the corrected labels."""
ARCHIVE_NAME = "CARE_To_Compare.zip"
ARCHIVE_SHA256 = "ca61379e98956d891041ad45c885109bd8a14199fde0688d0184a11c2d4194f1"
"""The SHA-256 hash of the archive, from a download whose MD5 matches the
``2547b58c21ac8c242d13232860cf500c`` that Zenodo publishes."""

PERIOD_S = PERIOD_US / 1_000_000
_FARM_DIRECTORY = re.compile(r"^Wind Farm ([A-Z])$")

EVENT_LABEL = LabelSet(
    [
        Label(
            name="anomaly",
            description="The prediction period holds an anomaly that leads up to a documented turbine fault.",
        ),
        Label(name="normal", description="The prediction period shows normal behavior."),
    ],
    annotations=[Annotation("name", value="event_label")],
)
FAULT_COMPONENT = LabelSet(
    [
        Label(name="hydraulic", description="The hydraulic group."),
        Label(name="gearbox", description="The gearbox or its bearings."),
        Label(name="generator_bearing", description="A generator bearing."),
        Label(name="transformer", description="The transformer or its cell."),
        Label(name="rotor_bearing", description="The rotor (main) bearing."),
    ],
    annotations=[Annotation("name", value="fault_component")],
)

DETECTION_PROMPT = (
    "Does the prediction period of this wind turbine's SCADA data show an anomaly that leads up to a turbine fault?"
)
LOCALIZATION_PROMPT = "Locate the anomaly in the prediction period that leads up to the turbine fault."
COMPONENT_PROMPT = "Which component does the anomaly in the prediction period lead up to a fault of?"
EVENT_WINDOW = (
    "The event of the prediction period, labeled anomaly or normal. For an anomaly event, the window starts "
    "where the authors estimate the anomaly starts and ends where the turbine fault starts."
)

SCADA_MISSING = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description=(
        "The SCADA system recorded no row for this 10-minute step, or the row left the cell empty. "
        "Wind farms B and C were delivered with every missing value replaced by 0, so a run of zeros "
        "there can also be a missing value."
    ),
)
STATUS = TimeSeriesSpec(
    spec_type="care_status_type_id",
    name="Turbine operating status (0 normal, 1 derated, 2 idling, 3 service, 4 downtime, 5 other)",
    unit_value=None,
    dtype="int8",
    missing=MissingValues(
        reason=MissingReason.NOT_RECORDED,
        description="The SCADA system recorded no row for this 10-minute step.",
    ),
)
STATISTIC_NAMES = {
    "average": "10-minute mean",
    "minimum": "10-minute minimum",
    "maximum": "10-minute maximum",
    "std_dev": "10-minute standard deviation",
}


def _load_column(cache_path: Path, index: int) -> pa.Array:
    """Read one feature row of a cached event matrix.

    Returns:
        The values as a float32 Arrow array, with a null for each NaN.
    """
    values = np.ascontiguousarray(np.load(cache_path, mmap_mode="r")[index])
    return pa.array(values, type=pa.float32(), mask=np.isnan(values))


def _column_loader(cache_path: Path, index: int) -> Callable[[], pa.Array]:
    return lambda: _load_column(cache_path, index)


def _interval(first_step: int, last_step: int) -> TimeInterval:
    """The interval from one grid step through another, on the Record timeline.

    Returns:
        The half-open interval that holds both steps.
    """
    return TimeInterval.seconds(first_step * PERIOD_S, (last_step + 1) * PERIOD_S)


@dataclass(frozen=True)
class CareWindFarm:
    """One wind farm directory of the release."""

    name: str
    """The wind farm's letter: ``A``, ``B``, or ``C``."""
    directory: Path

    @classmethod
    def discover(cls, root: Path) -> list[Self]:
        """Find the wind farm directories under an extraction root.

        Returns:
            The wind farms in letter order.

        Raises:
            TimeNetConnectorError: If the root holds no wind farm directory.
        """
        farms = []
        for info in sorted(root.rglob("event_info.csv")):
            match = _FARM_DIRECTORY.match(info.parent.name)
            if match:
                farms.append(cls(name=match.group(1), directory=info.parent))
        if not farms:
            raise TimeNetConnectorError(f"no 'Wind Farm <letter>/event_info.csv' under {root}")
        return sorted(farms, key=lambda farm: farm.name)

    @property
    def event_info_path(self) -> Path:
        """The wind farm's event table."""
        return self.directory / "event_info.csv"

    @property
    def feature_description_path(self) -> Path:
        """The wind farm's sensor descriptions."""
        return self.directory / "feature_description.csv"

    def event_path(self, event_id: int) -> Path:
        """The SCADA file of one event.

        Returns:
            The path of ``datasets/<event_id>.csv``.
        """
        return self.directory / "datasets" / f"{event_id}.csv"


class _FarmConverter:
    """Turns the events of one wind farm into Records and tasks."""

    def __init__(self, farm: CareWindFarm) -> None:
        self.farm = farm
        self.sensors = read_sensor_descriptions(farm.feature_description_path)
        self.events = read_event_info(farm.event_info_path)
        self._specs: dict[str, TimeSeriesSpec] = {}
        self._axis = RegularAxis(period_us=Fraction(PERIOD_US), offset_us=0)
        files = {int(path.stem) for path in (farm.directory / "datasets").glob("*.csv") if path.stem.isdigit()}
        if files != set(self.events):
            raise TimeNetConnectorError(
                f"Wind Farm {farm.name}: event files {sorted(files ^ set(self.events))} do not match event_info.csv"
            )

    def _sensor(self, column: str) -> tuple[SensorDescription, str]:
        # Wind farm A names its energy counters (sensor_44 to sensor_51) without a statistic suffix.
        # Each of them has one statistic in the feature description.
        bare = self.sensors.get(column)
        if bare is not None and len(bare.statistics) == 1:
            return bare, bare.statistics[0]
        sensor_name, statistic = split_feature_column(column)
        sensor = self.sensors.get(sensor_name)
        if sensor is None:
            raise TimeNetConnectorError(f"Wind Farm {self.farm.name}: column {column} has no feature description")
        return sensor, statistic

    def _spec(self, column: str) -> TimeSeriesSpec:
        """The spec of one sensor statistic, shared by every Record of the wind farm."""
        spec = self._specs.get(column)
        if spec is None:
            sensor, statistic = self._sensor(column)
            unit = sensor.unit_expression
            spec = TimeSeriesSpec(
                spec_type=f"care_{self.farm.name.lower()}_{column}",
                name=f"{sensor.description} ({STATISTIC_NAMES[statistic]})",
                unit_value=ureg.parse_units(unit) if unit else None,
                dtype="float32",
                missing=SCADA_MISSING,
            )
            self._specs[column] = spec
        return spec

    def _signal_annotations(self, column: str) -> list[Annotation]:
        sensor, statistic = self._sensor(column)
        annotations = [Annotation("statistic", value=statistic)]
        if sensor.unit and sensor.unit_expression is None:
            annotations.append(Annotation("source_unit", value=sensor.unit))
        if sensor.is_angle:
            annotations.append(Annotation("is_angle", value=True))
        if sensor.is_counter:
            annotations.append(Annotation("is_counter", value=True))
        return annotations

    def add_event(self, dataset: TimeFDataset, event: EventInfo) -> None:
        """Read one event file and add its Record and tasks.

        Raises:
            TimeNetConnectorError: If the file disagrees with its event_info row.
        """
        path = self.farm.event_path(event.event_id)
        table = read_event_table(path)
        where = f"Wind Farm {self.farm.name} event {event.event_id}"
        if table.asset_id != event.asset_id:
            raise TimeNetConnectorError(f"{where}: file asset {table.asset_id} is not {event.asset_id}")
        try:
            window = (table.step_of(event.start_us), table.step_of(event.end_us))
        except TimeNetConnectorError as exc:
            raise TimeNetConnectorError(f"{where}: event window: {exc}") from exc

        # Keep the values on disk, one contiguous row per feature, so each Signal loads only its own.
        cache_path = path.with_suffix(".values.npy")
        np.save(cache_path, table.values)
        signals = [
            Signal.from_loader(
                spec=self._spec(column),
                name=column,
                time_axis=self._axis,
                loader=_column_loader(cache_path, index),
                n_values=table.n_steps,
                annotations=self._signal_annotations(column),
            )
            for index, column in enumerate(table.feature_columns)
        ]
        status = Signal.from_values(
            pa.array(table.status, type=pa.int8(), mask=table.status < 0),
            spec=STATUS,
            name="status_type_id",
            time_axis=self._axis,
        )
        prediction = _interval(*table.prediction_steps)
        facts = [
            Annotation("wind_farm", value=self.farm.name),
            Annotation("event_id", value=event.event_id),
            Annotation("asset_id", value=event.asset_id),
            Annotation("event_label", value=event.label),
            Annotation("training_period", span=_interval(*table.train_steps)),
            Annotation("prediction_period", span=prediction),
        ]
        if event.description is not None:
            facts.append(Annotation("event_description", value=event.description))
        component = event.fault_component(self.farm.name)
        if component is not None:
            facts.append(Annotation("fault_component", value=component))
        record = Record(
            metadata={"source_record_id": f"care-{self.farm.name.lower()}-{event.event_id}"},
            start_time=TimeOrigin(table.start_us),
            sources=[
                Source("SCADA", signals=signals),
                Source("Turbine controller", signals=[status]),
            ],
            annotations=facts,
        )
        dataset.add_record(record)
        event_window = record.annotate(
            Annotation("event_window", value=event.label, description=EVENT_WINDOW, span=_interval(*window))
        )

        detection = ClassificationTask(
            inputs=[record],
            prompt=DETECTION_PROMPT,
            targets=event.label,
            label_set=EVENT_LABEL,
            scope=prediction,
        )
        tasks: list[ClassificationTask | TemporalLocalizationTask] = [detection]
        if event.label == "anomaly":
            tasks.append(
                TemporalLocalizationTask(
                    inputs=[record],
                    prompt=LOCALIZATION_PROMPT,
                    mode=LocalizationMode.SPARSE,
                    scope=prediction,
                    target_annotations=(event_window,),
                    from_tasks=(detection,),
                )
            )
        if component is not None:
            tasks.append(
                ClassificationTask(
                    inputs=[record],
                    prompt=COMPONENT_PROMPT,
                    targets=component,
                    label_set=FAULT_COMPONENT,
                    scope=prediction,
                    from_tasks=(detection,),
                )
            )
        dataset.add_tasks(tasks)


class CareToCompareConnector(BaseConnector[list[CareWindFarm]]):
    """Connector for CARE to Compare, version 6 of the Zenodo record."""

    async def download_async(self, cache_dir: Path) -> list[CareWindFarm]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            The wind farm directories.
        """
        root = await ensure_archive(ZENODO_URL, cache_dir, filename=ARCHIVE_NAME, sha256=ARCHIVE_SHA256)
        return CareWindFarm.discover(Path(root))

    def convert(self, farms: list[CareWindFarm]) -> TimeFDataset:
        """Create one Record and its tasks for each event of each wind farm.

        Args:
            farms: The wind farm directories from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for farm in farms:
            converter = _FarmConverter(farm)
            for event in sorted(converter.events.values(), key=lambda event: event.event_id):
                converter.add_event(dataset, event)
        return dataset


CONNECTOR = CareToCompareConnector
