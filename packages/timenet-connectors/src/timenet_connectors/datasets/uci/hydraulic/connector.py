"""The UCI hydraulic connector: load cycles of a hydraulic test rig with four component conditions.

The UCI Machine Learning Repository publishes the data set as dataset 447, "Condition monitoring of
hydraulic systems". ZeMA gGmbH recorded it on a test rig with a primary working circuit and a
secondary cooling-filtration circuit. The rig repeats a constant load cycle of 60 seconds, while
the condition of four components (cooler, valve, pump, accumulator) is varied.

The archive holds one tab-separated table for each of 17 sensors and ``profile.txt``. In every
file, row ``n`` is cycle ``n``. A sensor row holds the values of one cycle at the rate of that
sensor: 6,000 values at 100 Hz, 600 at 10 Hz, or 60 at 1 Hz. ``profile.txt`` holds five integer
columns per cycle: the four component grades and a flag that says whether the rig had reached
static conditions.

One cycle becomes one Record. Each sensor becomes one Signal on its own regular axis, at the rate
the release states, with no resampling. Each component becomes one ClassificationTask per cycle.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import csv

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive
from timenet_connectors.splitting import Splitter, StratifiedSplitter


HYDRAULIC_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
HYDRAULIC_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
PROFILE_FILE = "profile.txt"
CYCLE_SECONDS = 60
"""Every load cycle lasts 60 seconds, so a sensor row holds 60 seconds of values at its rate."""

_AXES = {rate: RegularAxis.from_rate_hz(rate) for rate in (1, 10, 100)}


@dataclass(frozen=True)
class Sensor:
    """One sensor of the rig: its table in the archive and what its values measure."""

    name: str
    """The name the release gives the sensor. Its table is ``<name>.txt``."""
    spec: TimeSeriesSpec
    rate_hz: int

    @property
    def time_axis(self) -> RegularAxis:
        """The regular axis of the sensor, at the rate the release states."""
        return _AXES[self.rate_hz]

    @property
    def values_per_cycle(self) -> int:
        """The number of values in one row of the table."""
        return self.rate_hz * CYCLE_SECONDS

    @property
    def file_name(self) -> str:
        """The name of the table of the sensor in the archive."""
        return f"{self.name}.txt"


# The physical quantities and units of the attribute table in the release's description.txt.
_PRESSURE = TimeSeriesSpec(spec_type="pressure", name="Pressure", unit_value=ureg.bar)
_MOTOR_POWER = TimeSeriesSpec(spec_type="motor_power", name="Motor power", unit_value=ureg.watt)
_VOLUME_FLOW = TimeSeriesSpec(spec_type="volume_flow", name="Volume flow", unit_value=ureg.liter / ureg.minute)
_TEMPERATURE = TimeSeriesSpec(spec_type="temperature", name="Temperature", unit_value=ureg.degC)
_VIBRATION = TimeSeriesSpec(spec_type="vibration", name="Vibration", unit_value=ureg.millimeter / ureg.second)
_COOLING_EFFICIENCY = TimeSeriesSpec(
    spec_type="cooling_efficiency", name="Cooling efficiency (virtual)", unit_value=ureg.percent
)
_COOLING_POWER = TimeSeriesSpec(spec_type="cooling_power", name="Cooling power (virtual)", unit_value=ureg.kilowatt)
_EFFICIENCY_FACTOR = TimeSeriesSpec(spec_type="efficiency_factor", name="Efficiency factor", unit_value=ureg.percent)

SOURCES: Mapping[str, tuple[Sensor, ...]] = {
    "Pressure": tuple(Sensor(f"PS{number}", _PRESSURE, 100) for number in range(1, 7)),
    "Motor power": (Sensor("EPS1", _MOTOR_POWER, 100),),
    "Volume flow": (Sensor("FS1", _VOLUME_FLOW, 10), Sensor("FS2", _VOLUME_FLOW, 10)),
    "Temperature": tuple(Sensor(f"TS{number}", _TEMPERATURE, 1) for number in range(1, 5)),
    "Vibration": (Sensor("VS1", _VIBRATION, 1),),
    "Efficiency factor": (Sensor("SE", _EFFICIENCY_FACTOR, 1),),
    # The release marks these two as virtual: they are computed from other process values.
    "Virtual": (Sensor("CE", _COOLING_EFFICIENCY, 1), Sensor("CP", _COOLING_POWER, 1)),
}
"""The Sources of a Record, by sensor group, and the Sensors of each."""
SENSORS: tuple[Sensor, ...] = tuple(sensor for sensors in SOURCES.values() for sensor in sensors)


@dataclass(frozen=True)
class Component:
    """One component whose condition ``profile.txt`` states, and the task that asks for it."""

    key: str
    """The annotation key of the grade, and the name of the label set."""
    description: str
    column: int
    """The zero-based column of ``profile.txt``."""
    unit: str | None
    """The unit of the grade, as a unit name of the shared registry."""
    grades: Mapping[int, str]
    """The label of each grade the release writes, from the healthy grade to the worst."""
    label_set: LabelSet

    def annotation(self, grade: int) -> Annotation:
        """The record annotation that holds the grade the release writes."""
        return Annotation(self.key, value=grade, unit=self.unit, description=self.description)

    def task(self, record: Record, grade: int, *, split: Split) -> ClassificationTask:
        """The task that asks for the condition of the component in one cycle."""
        return ClassificationTask(inputs=(record,), targets=self.grades[grade], label_set=self.label_set, split=split)


def _component(  # ruff: ignore[too-many-arguments] -- keep the facts of a component explicit.
    key: str, description: str, *, column: int, unit: str | None, symbol: str, grades: Mapping[int, str]
) -> Component:
    """Describe one component, with labels ordered from the healthy grade to the worst.

    Returns:
        The component and its ordered label set. Each label describes the grade it stands for.
    """
    labels = [Label(name=label, description=f"{description} {grade}{symbol}.") for grade, label in grades.items()]
    label_set = LabelSet(labels, ordered=True, annotations=[Annotation("name", value=key)])
    return Component(key, description, column, unit, grades, label_set)


# The grades and their meanings, as the release's documentation.txt lists them. The release spells
# the cooler's grade 20 "reduced effifiency"; the label corrects the spelling.
COOLER = _component(
    "cooler_condition",
    "Cooler condition",
    column=0,
    unit="percent",
    symbol=" %",
    grades={100: "full efficiency", 20: "reduced efficiency", 3: "close to total failure"},
)
VALVE = _component(
    "valve_condition",
    "Valve condition",
    column=1,
    unit="percent",
    symbol=" %",
    grades={100: "optimal switching behavior", 90: "small lag", 80: "severe lag", 73: "close to total failure"},
)
PUMP = _component(
    "internal_pump_leakage",
    "Internal pump leakage",
    column=2,
    unit=None,
    symbol="",
    grades={0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
)
ACCUMULATOR = _component(
    "hydraulic_accumulator",
    "Hydraulic accumulator pressure",
    column=3,
    unit="bar",
    symbol=" bar",
    grades={
        130: "optimal pressure",
        115: "slightly reduced pressure",
        100: "severely reduced pressure",
        90: "close to total failure",
    },
)
COMPONENTS = (COOLER, VALVE, PUMP, ACCUMULATOR)

STABLE_FLAG_COLUMN = 4
STABLE_FLAGS = {0: "conditions were stable", 1: "static conditions might not have been reached yet"}
_STABLE_FLAG_DESCRIPTION = "Stable flag. " + "; ".join(f"{flag}: {meaning}" for flag, meaning in STABLE_FLAGS.items())


@dataclass(frozen=True)
class HydraulicCycle:
    """One load cycle: its row number and the five values ``profile.txt`` states for it."""

    number: int
    """The one-based row number, which the release uses as the cycle number."""
    grades: tuple[int, int, int, int]
    """The cooler, valve, pump, and accumulator grades, in the column order of ``profile.txt``."""
    stable_flag: int

    @property
    def row(self) -> int:
        """The zero-based row of the cycle in every table."""
        return self.number - 1

    def grade(self, component: Component) -> int:
        """The grade of one component in this cycle."""
        return self.grades[component.column]


def get_condition(cycle: HydraulicCycle) -> Hashable:
    """Group the cycles that share all four grades.

    The rig holds one condition for runs of consecutive cycles, and neighboring cycles under one
    condition are close to copies of each other. All cycles of a condition stay in one partition.

    Returns:
        The cooler, valve, pump, and accumulator grades of the cycle.
    """
    return cycle.grades


def get_accumulator_grade(cycle: HydraulicCycle) -> Hashable:
    """Balance the accumulator grades, the hardest of the four targets, across partitions."""
    return cycle.grade(ACCUMULATOR)


def _read_table(path: Path, *, columns: int, value_type: pa.DataType) -> np.ndarray:
    """Read a headerless tab-separated table with a fixed number of columns.

    Returns:
        The values, one array row per table row.

    Raises:
        TimeNetConnectorError: If the file is missing, a row does not hold ``columns`` values, or a
            cell is empty or not a number.
    """
    if not path.is_file():
        raise TimeNetConnectorError(f"the release holds no {path.name}")
    names = [f"c{column}" for column in range(columns)]
    try:
        table = csv.read_csv(
            path,
            read_options=csv.ReadOptions(column_names=names),
            parse_options=csv.ParseOptions(delimiter="\t"),
            convert_options=csv.ConvertOptions(column_types=dict.fromkeys(names, value_type), null_values=[""]),
        )
    except pa.ArrowInvalid as exc:
        raise TimeNetConnectorError(f"{path.name} is not a table of {columns} numbers per row: {exc}") from exc
    if any(column.null_count for column in table.columns):
        raise TimeNetConnectorError(f"{path.name} has an empty cell")
    return np.column_stack([column.to_numpy() for column in table.columns])


@dataclass(frozen=True)
class HydraulicSource:
    """The directory that holds the extracted tables of the release."""

    root: Path

    def read_cycles(self) -> list[HydraulicCycle]:
        """Read ``profile.txt``, one cycle per row.

        Returns:
            The cycles, in row order.

        Raises:
            TimeNetConnectorError: If the file is missing or malformed, or states a value the
                release does not document.
        """
        table = _read_table(self.root / PROFILE_FILE, columns=5, value_type=pa.int32())
        cycles = []
        for number, values in enumerate(table.tolist(), start=1):
            for component in COMPONENTS:
                if values[component.column] not in component.grades:
                    raise TimeNetConnectorError(
                        f"{PROFILE_FILE} row {number}: {component.key} {values[component.column]} is not one of "
                        f"{sorted(component.grades)}"
                    )
            if values[STABLE_FLAG_COLUMN] not in STABLE_FLAGS:
                raise TimeNetConnectorError(
                    f"{PROFILE_FILE} row {number}: stable flag {values[STABLE_FLAG_COLUMN]} is not one of "
                    f"{sorted(STABLE_FLAGS)}"
                )
            cooler, valve, pump, accumulator, stable_flag = values
            cycles.append(HydraulicCycle(number, (cooler, valve, pump, accumulator), stable_flag))
        return cycles

    def read_sensor(self, sensor: Sensor, *, cycles: int) -> np.ndarray:
        """Read the table of one sensor as float32 values, one row per cycle.

        Every value in the release has at most five significant digits, so float32 holds each one
        without changing its decimal reading.

        Returns:
            The ``(cycles, values_per_cycle)`` array of the sensor.

        Raises:
            TimeNetConnectorError: If the file is missing or malformed, its row count differs from
                the cycle count of ``profile.txt``, or it holds a value that is not finite.
        """
        values = _read_table(self.root / sensor.file_name, columns=sensor.values_per_cycle, value_type=pa.float32())
        if values.shape[0] != cycles:
            raise TimeNetConnectorError(
                f"{sensor.file_name} holds {values.shape[0]} rows, but {PROFILE_FILE} holds {cycles} cycles"
            )
        if not np.isfinite(values).all():
            raise TimeNetConnectorError(f"{sensor.file_name} holds a value that is not finite")
        return values


def _check_train_coverage(train: Sequence[HydraulicCycle]) -> None:
    """Make sure the training partition holds every grade of every component.

    Raises:
        TimeNetConnectorError: If a grade has no training cycle.
    """
    for component in COMPONENTS:
        missing = set(component.grades) - {cycle.grade(component) for cycle in train}
        if missing:
            raise TimeNetConnectorError(f"the train split holds no cycle with {component.key} {sorted(missing)}")


def _make_record(cycle: HydraulicCycle, tables: Mapping[str, np.ndarray]) -> Record:
    """Create the Record of one cycle, with one Source per sensor group.

    Returns:
        The Record, not yet added to a dataset.
    """
    sources = [
        Source(
            group,
            signals=[
                Signal.from_values(
                    tables[sensor.name][cycle.row], spec=sensor.spec, name=sensor.name, time_axis=sensor.time_axis
                )
                for sensor in sensors
            ],
        )
        for group, sensors in SOURCES.items()
    ]
    annotations = [component.annotation(cycle.grade(component)) for component in COMPONENTS]
    annotations.append(Annotation("stable_flag", value=cycle.stable_flag, description=_STABLE_FLAG_DESCRIPTION))
    return Record(sources=sources, annotations=annotations, metadata={"cycle": cycle.number})


class HydraulicConnector(BaseConnector[HydraulicSource]):
    """Connector for UCI dataset 447, "Condition monitoring of hydraulic systems"."""

    def __init__(self) -> None:
        super().__init__()
        self.splitter: Splitter[HydraulicCycle] = StratifiedSplitter(
            train_folds=8,
            validation_folds=1,
            test_folds=1,
            stratify_by=get_accumulator_grade,
            # Consecutive cycles under one condition are near copies, so a random split would put
            # near copies of a test cycle in train. Each condition stays in one partition.
            group_by=get_condition,
            seed=42,
        )

    async def download_async(self, cache_dir: Path) -> HydraulicSource:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted tables.

        Returns:
            A handle to the extracted tables.
        """
        root = await ensure_archive(HYDRAULIC_URL, cache_dir, filename="hydraulic.zip", sha256=HYDRAULIC_SHA256)
        return HydraulicSource(root=root)

    def convert(self, source: HydraulicSource) -> TimeFDataset:
        """Create one Record per cycle and one ClassificationTask per cycle and component.

        Args:
            source: The extracted tables from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeNetConnectorError: If ``profile.txt`` holds no cycle.
        """
        cycles = source.read_cycles()
        if not cycles:
            raise TimeNetConnectorError(f"{PROFILE_FILE} holds no cycle")
        tables = {sensor.name: source.read_sensor(sensor, cycles=len(cycles)) for sensor in SENSORS}
        splits = self.splitter.split(cycles)
        _check_train_coverage(splits[Split.TRAIN])
        split_of = {cycle.number: split for split, members in splits.items() for cycle in members}
        dataset = TimeFDataset(metadata=self.metadata())
        for cycle in cycles:
            record = dataset.add_record(_make_record(cycle, tables))
            dataset.add_tasks(
                component.task(record, cycle.grade(component), split=split_of[cycle.number]) for component in COMPONENTS
            )
        return dataset


CONNECTOR = HydraulicConnector
