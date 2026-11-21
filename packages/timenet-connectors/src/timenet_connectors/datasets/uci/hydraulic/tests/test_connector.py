from collections import Counter
from collections.abc import Sequence
import itertools
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import RegularAxis, TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split, read_value
from timenet_connectors.datasets.uci.hydraulic.connector import (
    ACCUMULATOR,
    COMPONENTS,
    COOLER,
    PROFILE_FILE,
    SENSORS,
    SOURCES,
    HydraulicConnector,
    HydraulicCycle,
    HydraulicSource,
)
from timenet_connectors.splitting import DatasetSplits, Splitter


# Synthetic tables in the shape of the release, written by the tests, so the repo ships no dataset
# bytes. The first value of each row is the cycle number, so a row that lands on the wrong record
# fails. The other values end in 0.1, which has no exact binary form, so a float64 cast shows.
_TAIL = {
    sensor.name: "\t".join(f"{index % 97 + offset}.1" for index in range(1, sensor.values_per_cycle))
    for offset, sensor in enumerate(SENSORS)
}
_CONDITIONS = list(itertools.product(*(component.grades for component in COMPONENTS)))
# Every condition once, and the first three a second time, so a condition can hold several cycles.
_PROFILE = [(*condition, number % 2) for number, condition in enumerate(_CONDITIONS + _CONDITIONS[:3])]


def _write_release(directory: Path, profile: Sequence[Sequence[int]], **overrides: str) -> HydraulicSource:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROFILE_FILE).write_text("".join("\t".join(map(str, row)) + "\n" for row in profile))
    for sensor in SENSORS:
        text = "".join(f"{number}\t{_TAIL[sensor.name]}\n" for number in range(1, len(profile) + 1))
        (directory / sensor.file_name).write_text(overrides.get(sensor.name, text))
    return HydraulicSource(root=directory)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> HydraulicSource:
    return _write_release(tmp_path_factory.mktemp("release"), _PROFILE)


@pytest.fixture(scope="module")
def dataset(source) -> TimeFDataset:
    return HydraulicConnector().convert(source)


def test_metadata():
    metadata = HydraulicConnector().metadata()
    assert metadata.dataset_id == "uci/hydraulic"
    assert str(metadata.license) == "CC-BY-4.0"
    assert str(metadata.dataset_version) == "1.0.0"


def test_each_cycle_becomes_a_record_of_native_rate_signals(dataset):
    records = dataset.records
    assert len(records) == len(_PROFILE)
    for number, record in enumerate(records, start=1):
        assert record.metadata == {"cycle": number}
        assert sorted(source.name for source in record.sources) == sorted(SOURCES)
        for source in record.sources:
            assert [signal.name for signal in source.signals] == [sensor.name for sensor in SOURCES[source.name]]
        signals = {signal.name: signal for signal in record.signals}
        assert len(signals) == len(SENSORS)
        for sensor in SENSORS:
            signal = signals[sensor.name]
            assert signal.spec == sensor.spec
            assert signal.time_axis == RegularAxis.from_rate_hz(sensor.rate_hz)
            assert signal.n_values == sensor.rate_hz * 60
            expected = np.array([number, *_TAIL[sensor.name].split("\t")], dtype=np.float32)
            np.testing.assert_array_equal(signal.to_numpy(), expected)


def test_signals_keep_the_rates_and_units_of_the_release(dataset):
    signals = {signal.name: signal for signal in dataset.records[0].signals}
    rates = {name: len(signal.to_numpy()) // 60 for name, signal in signals.items()}
    assert rates == {
        **dict.fromkeys(["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"], 100),
        **dict.fromkeys(["FS1", "FS2"], 10),
        **dict.fromkeys(["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"], 1),
    }
    units = {name: str(signal.spec.unit_value) for name, signal in signals.items()}
    assert units["PS1"] == "bar"
    assert units["EPS1"] == "watt"
    assert units["FS1"] == "liter / minute"
    assert units["TS1"] == "degree_Celsius"
    assert units["VS1"] == "millimeter / second"
    assert units["CE"] == units["SE"] == "percent"
    assert units["CP"] == "kilowatt"


def test_each_cycle_has_one_classification_task_per_component(dataset):
    for record, row in zip(dataset.records, _PROFILE, strict=True):
        tasks = dataset.tasks_for(record)
        assert [type(task) for task in tasks] == [ClassificationTask] * len(COMPONENTS)
        assert len({task.split for task in tasks}) == 1
        for task, component in zip(tasks, COMPONENTS, strict=True):
            assert task.label_set == component.label_set
            assert task.targets is not None
            assert read_value(task.targets[0]) == component.grades[row[component.column]]
            assert task.prompt is None


def test_label_sets_are_ordered_from_healthy_to_worst():
    assert COOLER.label_set.names == ("full efficiency", "reduced efficiency", "close to total failure")
    assert COOLER.label_set.ordered
    assert [label.description for label in ACCUMULATOR.label_set.labels] == [
        "Hydraulic accumulator pressure 130 bar.",
        "Hydraulic accumulator pressure 115 bar.",
        "Hydraulic accumulator pressure 100 bar.",
        "Hydraulic accumulator pressure 90 bar.",
    ]


def test_record_annotations_hold_the_profile_values(dataset):
    for record, row in zip(dataset.records, _PROFILE, strict=True):
        values = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
        expected = {component.key: row[component.column] for component in COMPONENTS}
        assert values == {**expected, "stable_flag": row[4]}


def test_split_keeps_each_condition_whole_and_trains_on_every_grade(dataset):
    split_of_condition: dict[tuple[int, ...], set[Split]] = {}
    for record, row in zip(dataset.records, _PROFILE, strict=True):
        (task, *_) = dataset.tasks_for(record)
        assert task.split is not None
        split_of_condition.setdefault(tuple(row[:4]), set()).add(task.split)
    assert all(len(splits) == 1 for splits in split_of_condition.values())
    assert {split for splits in split_of_condition.values() for split in splits} == set(Split)
    train = [
        row for record, row in zip(dataset.records, _PROFILE, strict=True) if _split(dataset, record) == Split.TRAIN
    ]
    for component in COMPONENTS:
        assert {row[component.column] for row in train} == set(component.grades)


def test_split_is_reproducible(source):
    cycles = source.read_cycles()
    first = HydraulicConnector().splitter.split(cycles)
    second = HydraulicConnector().splitter.split(cycles)
    assert {split: [cycle.number for cycle in first[split]] for split in first} == {
        split: [cycle.number for cycle in second[split]] for split in second
    }
    assert sum(first.counts.values()) == len(_PROFILE)


def _split(dataset: TimeFDataset, record) -> Split | None:
    return dataset.tasks_for(record)[0].split


_ROWS_OF_THREE = _PROFILE[:3]
_GOOD_PS1 = "".join(f"{number}\t{_TAIL['PS1']}\n" for number in range(1, 4))


@pytest.mark.parametrize(
    ("profile", "overrides", "message"),
    [
        ([(100, 100, 0, 130, 0), (50, 100, 0, 130, 0)], {}, r"profile\.txt row 2: cooler_condition 50 is not one of"),
        ([(100, 100, 0, 130, 2)], {}, r"profile\.txt row 1: stable flag 2 is not one of"),
        (_ROWS_OF_THREE, {"PS1": _GOOD_PS1.rsplit("\t", 1)[0] + "\n"}, r"PS1\.txt is not a table of 6000 numbers"),
        (_ROWS_OF_THREE, {"TS1": "".join(f"x\t{_TAIL['TS1']}\n" for _ in range(3))}, r"TS1\.txt is not a table"),
        (_ROWS_OF_THREE, {"FS1": "".join(f"\t{_TAIL['FS1']}\n" for _ in range(3))}, r"FS1\.txt has an empty cell"),
        (
            _ROWS_OF_THREE,
            {"VS1": "".join(f"nan\t{_TAIL['VS1']}\n" for _ in range(3))},
            r"VS1\.txt holds a value that is not finite",
        ),
        (
            _ROWS_OF_THREE,
            {"PS1": _GOOD_PS1.split("\n", 1)[1]},
            r"PS1\.txt holds 2 rows, but profile\.txt holds 3 cycles",
        ),
    ],
    ids=[
        "an unknown grade",
        "an unknown stable flag",
        "one value too few",
        "a value that is not a number",
        "an empty cell",
        "a value that is not finite",
        "a missing cycle",
    ],
)
def test_a_malformed_release_raises(tmp_path, profile, overrides, message):
    source = _write_release(tmp_path, profile, **overrides)
    with pytest.raises(TimeNetConnectorError, match=message):
        HydraulicConnector().convert(source)


def test_a_missing_table_raises(tmp_path):
    source = _write_release(tmp_path, _ROWS_OF_THREE)
    (tmp_path / "CP.txt").unlink()
    with pytest.raises(TimeNetConnectorError, match=r"the release holds no CP\.txt"):
        HydraulicConnector().convert(source)


def test_conversion_rejects_too_few_conditions(tmp_path):
    source = _write_release(tmp_path, _ROWS_OF_THREE)
    with pytest.raises(TimeFValidationError, match=r"10 folds require at least 10 groups; found 3"):
        HydraulicConnector().convert(source)


class _EverythingInTest(Splitter[HydraulicCycle]):
    def split(self, samples: Sequence[HydraulicCycle]) -> DatasetSplits[HydraulicCycle]:
        return DatasetSplits({Split.TRAIN: [], Split.TEST: samples})


def test_conversion_rejects_a_train_split_without_every_grade(source):
    connector = HydraulicConnector()
    connector.splitter = _EverythingInTest()
    with pytest.raises(TimeNetConnectorError, match=r"the train split holds no cycle with cooler_condition"):
        connector.convert(source)


def test_stratification_balances_the_accumulator_grades(dataset):
    counts = Counter(
        (_split(dataset, record), row[ACCUMULATOR.column])
        for record, row in zip(dataset.records, _PROFILE, strict=True)
    )
    for split in Split:
        assert {grade for (part, grade) in counts if part == split} == set(ACCUMULATOR.grades)


def test_the_dataset_round_trips_through_storage(tmp_path, dataset):
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
