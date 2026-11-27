from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from timenet.connectors import BaseConnector
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import ClassificationTask
from timenet.types.splits import Split
from timenet_connectors.datasets.ucirvine.hydraulic_condition_monitoring.connector import (
    _COMPONENT_LABELS,
    _SENSORS,
    HydraulicConditionMonitoringConnector,
    HydraulicSource,
    _read_profile,
    _SensorMatrices,
)


class Run(NamedTuple):
    cooler: int
    valve: int
    pump: int
    accumulator: int
    cycles: int
    stable: bool


# A synthetic release shaped like the UCI files: one tab-separated row per cycle, one column per sample.
# As in the release, the valve changes nearly every run, the other components rarely, the stable flag
# covers whole runs, and the run after a cooler change is a long unstable warm-up. Eleven short runs,
# because the ten-fold split needs ten level combinations and ten cycles per accumulator level.
RUNS = [
    Run(cooler=3, valve=100, pump=0, accumulator=130, cycles=60, stable=False),  # warm-up
    Run(cooler=3, valve=90, pump=0, accumulator=130, cycles=3, stable=True),
    Run(cooler=3, valve=80, pump=0, accumulator=130, cycles=3, stable=True),
    Run(cooler=3, valve=73, pump=0, accumulator=130, cycles=3, stable=True),
    Run(cooler=3, valve=100, pump=1, accumulator=130, cycles=3, stable=False),
    Run(cooler=3, valve=90, pump=1, accumulator=130, cycles=3, stable=True),
    Run(cooler=20, valve=80, pump=1, accumulator=90, cycles=3, stable=True),
    Run(cooler=20, valve=73, pump=1, accumulator=90, cycles=3, stable=True),
    Run(cooler=20, valve=100, pump=2, accumulator=90, cycles=3, stable=True),
    Run(cooler=20, valve=90, pump=2, accumulator=90, cycles=3, stable=True),
    Run(cooler=20, valve=80, pump=2, accumulator=90, cycles=3, stable=True),
    Run(cooler=20, valve=73, pump=2, accumulator=90, cycles=3, stable=True),
]
N_CYCLES = sum(run.cycles for run in RUNS)


def _run_of_cycle(cycle: int) -> tuple[int, Run]:
    for number, run in enumerate(RUNS):
        if cycle < run.cycles:
            return number, run
        cycle -= run.cycles
    raise ValueError(cycle)


def _write_release(root: Path, runs=RUNS) -> HydraulicSource:
    rows = []
    for run in runs:
        for _ in range(run.cycles):
            rows.append([run.cooler, run.valve, run.pump, run.accumulator, 0 if run.stable else 1])
    np.savetxt(root / "profile.txt", rows, fmt="%d", delimiter="\t")
    # Every value of cycle c is c, so a value identifies the cycle it came from.
    for sensor in _SENSORS:
        with (root / f"{sensor.name}.txt").open("w") as file:
            for cycle in range(len(rows)):
                file.write("\t".join([str(cycle)] * sensor.n_values) + "\n")  # e.g. "7\t7\t7...\t7\n", n_values times
    return HydraulicSource(root)


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> HydraulicSource:
    return _write_release(tmp_path_factory.mktemp("hydraulic_release"))


def _convert(release: HydraulicSource):
    return HydraulicConditionMonitoringConnector().convert([release])


def _record_of_cycle(dataset, cycle: int):
    return next(record for record in dataset.records if record.metadata["cycle_index"] == cycle)


def _signal(record, name: str):
    return next(signal for signal in record.signals if signal.name == name)


def _annotations(owner) -> dict:
    return {annotation.key: annotation.value for annotation in owner.annotations}


def test_is_a_connector():
    assert isinstance(HydraulicConditionMonitoringConnector(), BaseConnector)


def test_metadata():
    metadata = HydraulicConditionMonitoringConnector().metadata()
    assert metadata.dataset_id == "ucirvine/hydraulic-condition-monitoring"
    assert str(metadata.license) == "CC-BY-4.0"


def test_source_is_pinned():
    # A tautology, but it pins the archive and its hash, so moving to another one is a deliberate test edit.
    assert HydraulicConditionMonitoringConnector.ZIP_URL == (
        "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
    )
    assert HydraulicConditionMonitoringConnector.ZIP_SHA256 == (
        "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
    )


def test_one_record_per_cycle_with_one_signal_per_sensor(release):
    dataset = _convert(release)
    assert len(dataset.records) == N_CYCLES
    for record in dataset.records:
        assert len(record.signals) == len(_SENSORS)
        for sensor in _SENSORS:
            signal = _signal(record, sensor.name)
            assert signal.n_values == sensor.n_values
            assert signal.spec is sensor.spec


def test_values_land_on_their_cycle(release):
    dataset = _convert(release)
    for cycle in (0, 5, N_CYCLES - 1):
        record = _record_of_cycle(dataset, cycle)
        values = _signal(record, "PS2").to_numpy()
        assert values.dtype == np.float32
        assert len(values) == 6000
        assert (values == cycle).all()
        assert (_signal(record, "TS1").to_numpy() == cycle).all()


def test_each_cycle_has_one_task_per_component_with_its_documented_label(release):
    dataset = _convert(release)
    for record in dataset.records:
        _, run = _run_of_cycle(record.metadata["cycle_index"])
        expected = {
            "cooler_vocabulary": _COMPONENT_LABELS["cooler"][run.cooler],
            "valve_vocabulary": _COMPONENT_LABELS["valve"][run.valve],
            "pump_vocabulary": _COMPONENT_LABELS["pump"][run.pump],
            "accumulator_vocabulary": _COMPONENT_LABELS["accumulator"][run.accumulator],
        }
        tasks = dataset.tasks_for(record, ClassificationTask)
        assert {task.target_schema: task.targets[0] for task in tasks} == expected


def test_vocabularies_list_every_label(release):
    vocabularies = _annotations(_convert(release))
    assert set(vocabularies) == {f"{component}_vocabulary" for component in _COMPONENT_LABELS}
    assert vocabularies["valve_vocabulary"] == [
        "optimal switching behavior",
        "small lag",
        "severe lag",
        "close to total failure",
    ]


def test_levels_stable_flag_and_run_are_record_annotations(release):
    dataset = _convert(release)
    warm_up, first_stable = _record_of_cycle(dataset, 0), _record_of_cycle(dataset, 60)
    assert _annotations(warm_up) == {
        "stable": False,
        "run": 0,
        "cooler_level": 3,
        "valve_level": 100,
        "pump_level": 0,
        "accumulator_level": 130,
    }
    assert _annotations(first_stable) == {
        "stable": True,
        "run": 1,
        "cooler_level": 3,
        "valve_level": 90,
        "pump_level": 0,
        "accumulator_level": 130,
    }
    units = {annotation.key: annotation.unit for annotation in warm_up.annotations}
    assert units["accumulator_level"] == "bar"
    assert units["pump_level"] is None
    # The flag describes the measurement, not the machine, so an unstable cycle keeps its tasks.
    assert len(dataset.tasks_for(warm_up)) == len(dataset.tasks_for(first_stable)) == len(_COMPONENT_LABELS)


def test_runs_are_numbered_from_any_component_change(release):
    # Between runs 0 and 1 only the valve changes, so one component is enough to start a new run.
    assert RUNS[0]._replace(valve=RUNS[1].valve, cycles=3, stable=True) == RUNS[1]
    dataset = _convert(release)
    runs = {_annotations(record)["run"] for record in dataset.records}
    assert runs == set(range(len(RUNS)))


def test_only_virtual_sensors_are_marked_derived(release):
    record = _record_of_cycle(_convert(release), 0)
    derived = {signal.name for signal in record.signals if "derived" in _annotations(signal)}
    assert derived == {"CE", "CP"}


def test_split_keeps_each_level_combination_in_one_partition(release):
    dataset = _convert(release)
    split_of_levels = {}
    for task in dataset.tasks:
        (record,) = task.inputs
        levels = tuple(value for key, value in _annotations(record).items() if key.endswith("_level"))
        assert split_of_levels.setdefault(levels, task.split) == task.split
    assert set(split_of_levels.values()) == {Split.TRAIN, Split.VALIDATION, Split.TEST}


def test_warm_up_run_trains_only(release):
    dataset = _convert(release)
    warm_up_splits = set()
    for task in dataset.tasks:
        (record,) = task.inputs
        if _annotations(record)["run"] == 0:
            warm_up_splits.add(task.split)
    assert warm_up_splits == {Split.TRAIN}
    assert len(dataset.get_train()) > len(dataset.get_validation()) + len(dataset.get_test())


def test_split_is_repeatable(release):
    first, second = _convert(release), _convert(release)
    assert [task.split for task in first.tasks] == [task.split for task in second.tasks]


def test_too_few_level_combinations_are_rejected(tmp_path):
    with pytest.raises(TimeFValidationError, match=r"10 folds require at least 10 groups; found 4"):
        _convert(_write_release(tmp_path, RUNS[:5]))  # the warm-up run is not split


def test_exactly_one_release_is_required(release):
    with pytest.raises(TimeFValidationError, match="expected one release, got 2"):
        HydraulicConditionMonitoringConnector().convert([release, release])


def _write_profile(path: Path, rows) -> Path:
    np.savetxt(path, rows, fmt="%d", delimiter="\t")
    return path


def test_read_profile_parses_each_row_and_numbers_the_runs(tmp_path):
    # cooler, valve, pump, accumulator, flag; rows 1 and 2 differ only in the pump
    path = _write_profile(tmp_path / "profile.txt", [[3, 100, 0, 130, 1], [3, 100, 0, 130, 0], [3, 100, 1, 130, 0]])
    cycles = _read_profile(path)
    assert [cycle.index for cycle in cycles] == [0, 1, 2]
    assert cycles[0].levels == {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130}
    assert cycles[2].levels == {"cooler": 3, "valve": 100, "pump": 1, "accumulator": 130}
    assert [cycle.stable for cycle in cycles] == [False, True, True]  # the file writes 1 for "not yet stable"
    assert [cycle.run for cycle in cycles] == [0, 0, 1]


def test_read_profile_rejects_an_undocumented_level(tmp_path):
    path = _write_profile(tmp_path / "profile.txt", [[3, 100, 0, 130, 0], [3, 100, 0, 131, 0]])
    with pytest.raises(TimeFFormatError, match=r"profile\.txt row 1: accumulator level 131 is not documented"):
        _read_profile(path)


def test_read_profile_rejects_a_row_without_the_flag(tmp_path):
    path = _write_profile(tmp_path / "profile.txt", [[3, 100, 0, 130]])
    with pytest.raises(TimeFFormatError, match=r"profile\.txt row 0 has 4 columns, expected 5"):
        _read_profile(path)


def _sensor(name: str):
    return next(sensor for sensor in _SENSORS if sensor.name == name)


def test_sensor_matrices_serve_one_cycle_of_one_sensor(release):
    matrices = _SensorMatrices(release.root, N_CYCLES)
    row = matrices.row(_sensor("PS2"), 5).to_numpy()
    assert row.dtype == np.float32
    assert len(row) == 6000
    assert (row == 5).all()
    assert (matrices.row(_sensor("TS1"), N_CYCLES - 1).to_numpy() == N_CYCLES - 1).all()


def test_sensor_matrices_parse_a_file_once(release, monkeypatch):
    calls = []
    loadtxt = np.loadtxt
    monkeypatch.setattr(np, "loadtxt", lambda path, **kwargs: calls.append(path) or loadtxt(path, **kwargs))
    matrices = _SensorMatrices(release.root, N_CYCLES)
    for cycle in range(3):
        matrices.row(_sensor("PS2"), cycle)
    assert calls == [release.root / "PS2.txt"]


def test_sensor_matrices_reject_a_missing_file(tmp_path):
    _write_release(tmp_path)
    (tmp_path / "VS1.txt").unlink()
    with pytest.raises(TimeNetDownloadError, match=r"is missing VS1\.txt"):
        _SensorMatrices(tmp_path, N_CYCLES)


def test_sensor_matrices_reject_a_wrong_sample_count_on_first_read(tmp_path):
    _write_release(tmp_path)
    np.savetxt(tmp_path / "FS1.txt", np.zeros((N_CYCLES, 599)), delimiter="\t")
    matrices = _SensorMatrices(tmp_path, N_CYCLES)
    with pytest.raises(
        TimeFFormatError, match=rf"FS1\.txt has shape \({N_CYCLES}, 599\), expected \({N_CYCLES}, 600\)"
    ):
        matrices.row(_sensor("FS1"), 0)


def test_round_trip_through_the_writer(release, tmp_path):
    dataset = _convert(release)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
    assert len(restored.records) == N_CYCLES
    assert len(restored.tasks) == len(_COMPONENT_LABELS) * N_CYCLES
    assert {task.split for task in restored.tasks} == {Split.TRAIN, Split.VALIDATION, Split.TEST}
    record = _record_of_cycle(restored, 61)
    assert (_signal(record, "PS2").to_numpy() == 61).all()
    assert _annotations(record) == {
        "stable": True,
        "run": 1,
        "cooler_level": 3,
        "valve_level": 90,
        "pump_level": 0,
        "accumulator_level": 130,
    }
