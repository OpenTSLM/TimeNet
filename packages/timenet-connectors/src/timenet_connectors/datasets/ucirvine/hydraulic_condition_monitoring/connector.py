"""The UCI hydraulic condition-monitoring connector: one record per load cycle of a test rig.

The rig repeats one 60 s load cycle while four of its components (cooler, valve, pump and
accumulator) are held at documented degradation levels. Seventeen sensors record each cycle at
100, 10 or 1 Hz. A cycle becomes one Record, and each component gives it one
:class:`~timenet.types.ClassificationTask` whose target is the documented level name.

Consecutive cycles share their levels and the rig's thermal state, so the split keeps every cycle of
one level combination in one partition, and the long warm-up runs in train. See
``HydraulicConditionMonitoringConnector._split``.
"""

from collections import Counter
from dataclasses import dataclass
import functools
from pathlib import Path

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDownloadError
from timenet.types import Annotation, ClassificationTask, TimeSeriesSpec, ureg
from timenet.types.splits import Split
from timenet_connectors.download import ensure_archive, find_dir_containing
from timenet_connectors.splitting import Splitter, StratifiedSplitter


_CYCLE_SECONDS = 60
# A run holds 10 cycles, except the three that follow a cooler change: those hold about 200 and are
# flagged unstable throughout, the rig warming up to its new thermal state.
_WARM_UP_RUN_CYCLES = 50

_PRESSURE = TimeSeriesSpec(spec_type="pressure", name="Pressure", unit_value=ureg.bar)
_MOTOR_POWER = TimeSeriesSpec(spec_type="motor_power", name="Motor power", unit_value=ureg.watt)
_VOLUME_FLOW = TimeSeriesSpec(spec_type="volume_flow", name="Volume flow", unit_value=ureg.liter / ureg.minute)
_TEMPERATURE = TimeSeriesSpec(spec_type="temperature", name="Temperature", unit_value=ureg.degC)
_VIBRATION = TimeSeriesSpec(spec_type="vibration", name="Vibration", unit_value=ureg.millimeter / ureg.second)
_COOLING_EFFICIENCY = TimeSeriesSpec(spec_type="cooling_efficiency", name="Cooling efficiency", unit_value=ureg.percent)
_COOLING_POWER = TimeSeriesSpec(spec_type="cooling_power", name="Cooling power", unit_value=ureg.kilowatt)
_EFFICIENCY_FACTOR = TimeSeriesSpec(spec_type="efficiency_factor", name="Efficiency factor", unit_value=ureg.percent)


@dataclass(frozen=True)
class _Sensor:
    name: str
    rate_hz: int
    spec: TimeSeriesSpec
    derived: bool = False

    @property
    def n_values(self) -> int:
        return self.rate_hz * _CYCLE_SECONDS


# Transcribed from description.txt, the only place the release states names, units and rates.
_SENSORS = (
    _Sensor("PS1", 100, _PRESSURE),
    _Sensor("PS2", 100, _PRESSURE),
    _Sensor("PS3", 100, _PRESSURE),
    _Sensor("PS4", 100, _PRESSURE),
    _Sensor("PS5", 100, _PRESSURE),
    _Sensor("PS6", 100, _PRESSURE),
    _Sensor("EPS1", 100, _MOTOR_POWER),
    _Sensor("FS1", 10, _VOLUME_FLOW),
    _Sensor("FS2", 10, _VOLUME_FLOW),
    _Sensor("TS1", 1, _TEMPERATURE),
    _Sensor("TS2", 1, _TEMPERATURE),
    _Sensor("TS3", 1, _TEMPERATURE),
    _Sensor("TS4", 1, _TEMPERATURE),
    _Sensor("VS1", 1, _VIBRATION),
    _Sensor("SE", 1, _EFFICIENCY_FACTOR),
    # description.txt marks these two "(virtual)": computed from other sensors, not measured.
    _Sensor("CE", 1, _COOLING_EFFICIENCY, derived=True),
    _Sensor("CP", 1, _COOLING_POWER, derived=True),
)


# profile.txt column order; each component's levels and their meaning, in description.txt order
# (cooler worst first, the other components best first).
_COMPONENT_LABELS = {
    "cooler": {3: "close to total failure", 20: "reduced efficiency", 100: "full efficiency"},
    "valve": {100: "optimal switching behavior", 90: "small lag", 80: "severe lag", 73: "close to total failure"},
    "pump": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
    "accumulator": {
        130: "optimal pressure",
        115: "slightly reduced pressure",
        100: "severely reduced pressure",
        90: "close to total failure",
    },
}
# The pump's 0, 1 and 2 are ordinal codes with no unit.
_LEVEL_UNITS = {"cooler": "percent", "valve": "percent", "pump": None, "accumulator": "bar"}


@dataclass(frozen=True)
class HydraulicSource:
    """Handle to the extracted release: the directory with ``profile.txt`` and the sensor files."""

    root: Path


@dataclass(frozen=True)
class _Cycle:
    """One row of ``profile.txt``: a cycle's component levels, stable flag and run number."""

    index: int
    levels: dict[str, int]
    stable: bool
    run: int


def _read_profile(path: Path) -> list[_Cycle]:
    """Parse ``profile.txt`` and number the runs of consecutive cycles that share their levels.

    Returns:
        One cycle per row, in file order.

    Raises:
        TimeFFormatError: If a row does not hold four documented levels and a flag.
    """
    rows = np.loadtxt(path, delimiter="\t", dtype=np.int64, ndmin=2).tolist()
    cycles = []
    run = -1
    previous_levels = None
    n_columns = len(_COMPONENT_LABELS) + 1  # the four levels and the stable flag
    for index, row in enumerate(rows):
        if len(row) != n_columns:
            raise TimeFFormatError(f"profile.txt row {index} has {len(row)} columns, expected {n_columns}")
        levels, flag = row[:-1], row[-1]
        for component, level in zip(_COMPONENT_LABELS, levels, strict=True):
            if level not in _COMPONENT_LABELS[component]:
                raise TimeFFormatError(f"profile.txt row {index}: {component} level {level} is not documented")
        if levels != previous_levels:
            run += 1
            previous_levels = levels
        cycles.append(
            _Cycle(
                index=index,
                levels=dict(zip(_COMPONENT_LABELS, levels, strict=True)),
                # The release writes 1 for "static conditions might not have been reached yet".
                stable=flag == 0,
                run=run,
            )
        )
    return cycles


class _SensorMatrices:
    """Parses each sensor file once, on first use, and serves one cycle's row at a time."""

    def __init__(self, root: Path, n_cycles: int) -> None:
        """Check that every sensor file exists, so that a build fails before it starts writing.

        Raises:
            TimeNetDownloadError: If a sensor file is missing.
        """
        for sensor in _SENSORS:
            if not (root / f"{sensor.name}.txt").is_file():
                raise TimeNetDownloadError(f"{root} is missing {sensor.name}.txt")
        self._root = root
        self._n_cycles = n_cycles
        self._matrix = functools.cache(self._load)

    def _load(self, sensor: _Sensor) -> np.ndarray:
        """Parse one sensor file into a (cycle, sample) float32 matrix.

        Returns:
            The C-contiguous matrix, one row per cycle.

        Raises:
            TimeFFormatError: If the file's shape is not (number of cycles, rate x 60 s).
        """
        values = np.loadtxt(self._root / f"{sensor.name}.txt", delimiter="\t", dtype=np.float32, ndmin=2)
        expected = (self._n_cycles, sensor.n_values)
        if values.shape != expected:
            raise TimeFFormatError(f"{sensor.name}.txt has shape {values.shape}, expected {expected}")
        return values

    def row(self, sensor: _Sensor, cycle: int) -> pa.Array:
        """Return one cycle of one sensor as an Arrow array that shares the matrix's memory."""
        return pa.array(self._matrix(sensor)[cycle])


class HydraulicConditionMonitoringConnector(BaseConnector[HydraulicSource]):
    """Connector for UCI dataset 447, condition monitoring of hydraulic systems."""

    ZIP_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
    # The archive has had one release, in April 2018.
    ZIP_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"

    def __init__(self) -> None:
        super().__init__()
        self.splitter: Splitter[_Cycle] = StratifiedSplitter(
            train_folds=8,
            validation_folds=1,
            test_folds=1,
            # The hardest target in the literature, so every partition gets each of its levels.
            stratify_by=lambda cycle: cycle.levels["accumulator"],
            # Cycles with the same levels are near copies, whether in one run or in two runs hours
            # apart, so a level combination is never split: the test set holds unseen combinations.
            group_by=lambda cycle: tuple(cycle.levels.values()),
            seed=42,
        )

    async def download_async(self, cache_dir: Path) -> list[HydraulicSource]:
        """Download and extract the UCI archive once, checking its sha256.

        Returns:
            A list of one :class:`HydraulicSource`.
        """
        root = await ensure_archive(self.ZIP_URL, cache_dir, sha256=self.ZIP_SHA256)
        return [HydraulicSource(find_dir_containing(root, "profile.txt"))]

    def convert(self, raw_refs: list[HydraulicSource]) -> TimeFDataset:
        """Build the dataset from the extracted release.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.

        Raises:
            TimeFValidationError: If ``raw_refs`` does not hold exactly one release.
        """
        if len(raw_refs) != 1:
            raise TimeFValidationError(f"expected one release, got {len(raw_refs)}")
        cycles = _read_profile(raw_refs[0].root / "profile.txt")
        matrices = _SensorMatrices(raw_refs[0].root, len(cycles))

        dataset = TimeFDataset(metadata=self.metadata())
        for component, labels in _COMPONENT_LABELS.items():
            # Each task cites this annotation as its target_schema.
            dataset.annotate(
                Annotation(
                    key=f"{component}_vocabulary",
                    value=list(labels.values()),
                    description=f"The labels a {component!r} task's target takes, in description.txt order.",
                )
            )

        split_of_cycle = self._split(cycles)
        for cycle in cycles:
            record = dataset.add_record(record=self._make_record(cycle, matrices))
            dataset.add_tasks(tasks=self._make_tasks(cycle, record, split_of_cycle[cycle.index]))
        return dataset

    def _split(self, cycles: list[_Cycle]) -> dict[int, Split]:
        """Assign each cycle to a partition: warm-up runs to train, the rest through the splitter.

        A warm-up run would fill a whole validation or test partition with 200 unstable cycles of
        one level combination. Its labels are still true, so it trains rather than being dropped.

        Returns:
            The partition of each cycle, keyed by cycle index.
        """
        run_sizes = Counter(cycle.run for cycle in cycles)
        split_of_cycle = {}
        to_split = []
        for cycle in cycles:
            if run_sizes[cycle.run] >= _WARM_UP_RUN_CYCLES:
                split_of_cycle[cycle.index] = Split.TRAIN
            else:
                to_split.append(cycle)
        for split, members in self.splitter.split(to_split).items():
            for cycle in members:
                split_of_cycle[cycle.index] = split
        return split_of_cycle

    @staticmethod
    def _make_record(cycle: _Cycle, matrices: _SensorMatrices) -> Record:
        """Build a cycle's record: one rig source holding a lazy signal per sensor.

        Returns:
            The record, with its levels, stable flag and run number attached as annotations.
        """
        signals = []
        for sensor in _SENSORS:
            signal = Signal.from_loader(
                spec=sensor.spec,
                time_axis=RegularAxis.from_rate_hz(sensor.rate_hz),
                name=sensor.name,
                loader=functools.partial(matrices.row, sensor, cycle.index),
                n_values=sensor.n_values,
            )
            if sensor.derived:
                signal.annotate(
                    Annotation(key="derived", value=True, description="Computed from other sensors, not measured.")
                )
            signals.append(signal)
        record = Record(
            sources=(Source(name="Hydraulic test rig", signals=tuple(signals)),),
            metadata={"cycle_index": cycle.index},
        )
        # None of these is a task input: the levels are the answers, and the rest describe the
        # recording rather than the machine.
        annotations = [
            Annotation(
                key="stable",
                value=cycle.stable,
                description="Whether the rig had reached static conditions; the release flags whole runs.",
            ),
            Annotation(
                key="run",
                value=cycle.run,
                description="Number of the run of consecutive cycles sharing levels; a grouping key, not an input.",
            ),
        ]
        for component, level in cycle.levels.items():
            annotations.append(
                Annotation(
                    key=f"{component}_level",
                    value=level,
                    unit=_LEVEL_UNITS[component],
                    description=f"The {component} setting as profile.txt states it; the task target names it.",
                )
            )
        record.add_annotations(annotations)
        return record

    @staticmethod
    def _make_tasks(cycle: _Cycle, record: Record, split: Split) -> list[ClassificationTask]:
        """Build the four component tasks of one cycle.

        Returns:
            One task per component, in ``_COMPONENT_LABELS`` order.
        """
        tasks = []
        for component, labels in _COMPONENT_LABELS.items():
            tasks.append(
                ClassificationTask(
                    inputs=(record,),
                    targets=(labels[cycle.levels[component]],),
                    target_schema=f"{component}_vocabulary",
                    split=split,
                )
            )
        return tasks


CONNECTOR = HydraulicConditionMonitoringConnector
