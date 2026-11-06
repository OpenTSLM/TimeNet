from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
import math

import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.testing import assert_datasets_equal, assert_round_trips, make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    ScalarPredictionTask,
    TimeInterval,
    TimeSeriesSpec,
    Version,
    ureg,
)


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _streamed_dataset() -> TimeFDataset:
    metadata = DatasetMetadata(
        dataset_id="timenet/streamed",
        dataset_version=Version(1, 0, 0),
        name="Streamed",
        description="test",
        license=License.CC_BY_4_0,
    )
    dataset = TimeFDataset(metadata=metadata)
    signal = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[1.0, 2.0])
    record = dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))
    dataset.set_task_stream([AnswerTask], lambda: (AnswerTask(inputs=(record,), targets=(str(i),)) for i in range(3)))
    return dataset


def test_round_trip_returns_the_stored_dataset(tmp_path):
    restored = assert_round_trips(make_dataset(), tmp_path / "registry")
    assert_datasets_equal(make_dataset(), restored)


def _dataset_with_values(values, *, value_shape=()):
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    if value_shape:
        storage = pa.array(values, type=pa.list_(pa.float32(), math.prod(value_shape)))
        values = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.float32(), value_shape), storage)
    signal = Signal.from_values(
        values,
        spec=_SPEC.model_copy(update={"nullable": True, "value_shape": value_shape}),
        name="a",
        time_axis=RegularAxis(period_us=Fraction(1000)),
    )
    dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))
    return dataset


def test_round_trip_accepts_nan_and_null_as_distinct_values(tmp_path):
    restored = assert_round_trips(_dataset_with_values([1.0, math.nan, None, math.inf]), tmp_path / "registry")
    array = restored.records[0].signals[0].to_arrow()
    assert array.null_count == 1
    assert math.isnan(array[1].as_py())


def test_tensor_comparison_accepts_matching_nans_and_nulls():
    values = [[1.0, math.nan], None, [math.inf, -math.inf]]
    assert_datasets_equal(
        _dataset_with_values(values, value_shape=(2,)), _dataset_with_values(values, value_shape=(2,))
    )


@pytest.mark.parametrize("value_shape", [(), (2,)])
@pytest.mark.parametrize("changed", [None, 0.0, 2.0])
def test_signal_comparison_rejects_nan_replaced_by_null_or_a_value(value_shape, changed):
    expected_values = [[1.0, math.nan]] if value_shape else [1.0, math.nan]
    actual_values = [None if changed is None else [1.0, changed]] if value_shape else [1.0, changed]
    expected = _dataset_with_values(expected_values, value_shape=value_shape)
    actual = _dataset_with_values(actual_values, value_shape=value_shape)
    with pytest.raises(AssertionError, match="values differ"):
        assert_datasets_equal(expected, actual)
    with pytest.raises(AssertionError, match="values differ"):
        assert_datasets_equal(actual, expected)


def test_round_trip_compares_streamed_tasks(tmp_path):
    restored = assert_round_trips(_streamed_dataset(), tmp_path / "registry")
    assert [task.targets for task in restored.tasks] == [("0",), ("1",), ("2",)]


def test_round_trip_reports_a_changed_streamed_task(monkeypatch, tmp_path):
    read = TimeFReader.read

    def read_with_changed_target(self, **kwargs):
        restored = read(self, **kwargs)
        changed = replace(restored._tasks[1], targets=("changed",))
        changed._id = restored._tasks[1].id
        restored._tasks[1] = changed
        return restored

    monkeypatch.setattr(TimeFReader, "read", read_with_changed_target)
    with pytest.raises(AssertionError, match="task 2 targets differ"):
        assert_round_trips(_streamed_dataset(), tmp_path / "registry")


def test_round_trip_reports_a_lost_confidence(monkeypatch, tmp_path):
    dataset = make_dataset()
    dataset.records[0].annotate(Annotation(key="grade", value="b", confidence=0.75))
    read = TimeFReader.read

    def read_without_confidence(self, **kwargs):
        restored = read(self, **kwargs)
        record = restored.records[0]
        record.annotations = tuple(
            replace(one, confidence=None) if one.key == "grade" else one for one in record.annotations
        )
        return restored

    monkeypatch.setattr(TimeFReader, "read", read_without_confidence)
    with pytest.raises(AssertionError, match="annotations differ"):
        assert_round_trips(dataset, tmp_path / "registry")


def test_round_trip_runs_verify(monkeypatch, tmp_path):
    def broken_verify(self):
        raise TimeFFormatError("corrupt")

    monkeypatch.setattr(TimeFReader, "verify", broken_verify)
    with pytest.raises(TimeFFormatError, match="corrupt"):
        assert_round_trips(make_dataset(), tmp_path / "registry")


def test_round_trip_reports_a_lost_record(monkeypatch, tmp_path):
    read = TimeFReader.read

    def read_without_last_record(self, **kwargs):
        restored = read(self, **kwargs)
        restored._records.pop()
        return restored

    monkeypatch.setattr(TimeFReader, "read", read_without_last_record)
    with pytest.raises(AssertionError, match="record ids differ"):
        assert_round_trips(make_dataset(), tmp_path / "registry")


class _Key(StrEnum):
    SITE = "site"
    NIGHT = "night"


def test_round_trip_accepts_annotation_keys_built_from_an_enum(tmp_path):
    dataset = make_dataset()
    dataset.annotate(Annotation(key=_Key.SITE, value="clinic"))
    dataset.annotate(Annotation(key="z_site", value="home"))
    record = dataset.records[0]
    record.annotate(Annotation(key=_Key.NIGHT, value=1))
    record.annotate(Annotation(key="z_night", value=2))
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert {one.key for one in restored.annotations} >= {"site", "z_site"}
    assert {one.key for one in restored.records[0].annotations} >= {"night", "z_night"}


@pytest.mark.parametrize("owner", ["dataset", "record"])
def test_round_trip_accepts_enum_and_string_occurrences_of_the_same_key(tmp_path, owner):
    dataset = make_dataset()
    target = dataset if owner == "dataset" else dataset.records[0]
    target.annotate(Annotation(key=_Key.NIGHT, value=1))
    target.annotate(Annotation(key="night", value=2))
    assert_round_trips(dataset, tmp_path / "registry")


def _annotation_owner(dataset, owner):
    return {
        "dataset": dataset,
        "record": dataset.records[0],
        "source": dataset.records[0].sources[0],
        "signal": dataset.records[0].signals[0],
        "task": dataset.tasks[0],
    }[owner]


@pytest.mark.parametrize("owner", ["dataset", "record", "source", "signal", "task"])
@pytest.mark.parametrize("field", ["confidence", "occurrence_metadata", "value"])
def test_round_trip_detects_changed_annotation_fields_on_every_owner(monkeypatch, tmp_path, owner, field):
    dataset = make_dataset()
    _annotation_owner(dataset, owner).annotate(
        Annotation(key="quality", value=1, confidence=0.75, occurrence_metadata={"rater": "A"})
    )
    read = TimeFReader.read

    def changed_read(self, **kwargs):
        restored = read(self, **kwargs)
        target = _annotation_owner(restored, owner)
        changes = {field: {"confidence": None, "occurrence_metadata": {}, "value": 1.0}[field]}
        annotations = tuple(
            replace(annotation, **changes) if annotation.key == "quality" else annotation
            for annotation in target.annotations
        )
        if owner == "dataset":
            target._annotations = list(annotations)
        else:
            object.__setattr__(target, "annotations", annotations)
        return restored

    monkeypatch.setattr(TimeFReader, "read", changed_read)
    with pytest.raises(AssertionError, match="annotations"):
        assert_round_trips(dataset, tmp_path / "registry")


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0), (0.0, -0.0)])
def test_annotation_comparison_preserves_numeric_types_and_signed_zero(left, right):
    expected, actual = make_dataset(), make_dataset()
    expected.annotate(Annotation(key="n", value=left))
    actual.annotate(Annotation(key="n", value=right))
    with pytest.raises(AssertionError, match="dataset annotations differ"):
        assert_datasets_equal(expected, actual)


def test_annotation_comparison_preserves_duplicate_occurrences():
    expected, actual = make_dataset(), make_dataset()
    annotation = Annotation(key="n", value=1)
    expected.annotate(annotation)
    expected.annotate(annotation)
    actual.annotate(annotation)
    with pytest.raises(AssertionError, match="dataset annotations differ"):
        assert_datasets_equal(expected, actual)


def test_round_trip_refuses_a_registry_that_already_holds_the_version(tmp_path):
    dataset = make_dataset()
    assert_round_trips(dataset, tmp_path / "registry")
    dataset.records[0].annotate(Annotation(key="later", value=1))
    with pytest.raises(TimeFValidationError, match="already holds"):
        assert_round_trips(dataset, tmp_path / "registry")


def _two_record_dataset() -> tuple[TimeFDataset, Record, Record]:
    dataset = _streamed_dataset()
    dataset._task_stream = None
    first = dataset.records[0]
    signal = Signal(spec=_SPEC, name="b", time_axis=RegularAxis(period_us=Fraction(1000)), data=[3.0, 4.0])
    second = dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))
    return dataset, first, second


def _patch_read(monkeypatch, change):
    read = TimeFReader.read

    def changed_read(self, **kwargs):
        restored = read(self, **kwargs)
        change(restored)
        return restored

    monkeypatch.setattr(TimeFReader, "read", changed_read)


def test_round_trip_reports_a_lost_record_time_span(monkeypatch, tmp_path):
    dataset, _, _ = _two_record_dataset()
    signal = Signal(spec=_SPEC, name="c", time_axis=RegularAxis(period_us=Fraction(1000)), data=[5.0, 6.0])
    dataset.add_record(
        record=Record(sources=(Source(name="device", signals=(signal,)),), time_span=TimeInterval.seconds(0.0, 0.003))
    )
    _patch_read(monkeypatch, lambda restored: setattr(restored.records[2], "time_span", None))
    with pytest.raises(AssertionError, match="time_span differs"):
        assert_round_trips(dataset, tmp_path / "registry")


def test_round_trip_compares_numeric_targets_by_type_and_bits(monkeypatch, tmp_path):
    dataset, first, _ = _two_record_dataset()
    dataset.add_task(task=ScalarPredictionTask(inputs=(first,), targets=(math.nan,)))
    dataset.add_task(task=ScalarPredictionTask(inputs=(first,), targets=(0.0,)))
    assert_round_trips(dataset, tmp_path / "registry")

    def flip_zero(restored):
        changed = replace(restored._tasks[1], targets=(-0.0,))
        changed._id = restored._tasks[1].id
        restored._tasks[1] = changed

    _patch_read(monkeypatch, flip_zero)
    with pytest.raises(AssertionError, match="targets differ"):
        assert_round_trips(dataset, tmp_path / "registry-2")


def test_round_trip_reports_a_task_annotation_on_another_owner(monkeypatch, tmp_path):
    dataset, first, second = _two_record_dataset()
    on_first = first.annotate(Annotation(key="finding", value="normal"))
    second.annotate(Annotation(key="finding", value="normal"))
    dataset.add_task(task=AnswerTask(inputs=(first, second), target_annotations=(on_first,)))

    def point_to_second(restored):
        task = restored._tasks[0]
        other = next(one for one in restored.records[1].annotations if one.key == "finding")
        changed = replace(task, target_annotations=(other,))
        changed._id = task.id
        restored._tasks[0] = changed

    _patch_read(monkeypatch, point_to_second)
    with pytest.raises(AssertionError, match="point to other owners"):
        assert_round_trips(dataset, tmp_path / "registry")


def test_round_trip_finds_annotation_owners_when_a_release_stores_signals_in_another_order(tmp_path):
    dataset, _, _ = _two_record_dataset()
    later = Signal(spec=_SPEC, name="z", time_axis=RegularAxis(period_us=Fraction(1000)), data=[7.0, 8.0])
    earlier = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[9.0, 1.0])
    record = dataset.add_record(record=Record(sources=(Source(name="pair", signals=(later, earlier)),)))
    on_later = later.annotate(Annotation(key="finding", value="z"))
    earlier.annotate(Annotation(key="finding", value="a"))
    dataset.add_task(task=AnswerTask(inputs=(record,), targets=("x",), input_annotations=(on_later,)))
    assert_round_trips(dataset, tmp_path / "registry")


def test_dataset_comparison_reports_an_extra_dataset_annotation():
    expected, actual = make_dataset(), make_dataset()
    actual.annotate(Annotation(key="extra", value=1))
    with pytest.raises(AssertionError, match="dataset annotations differ"):
        assert_datasets_equal(expected, actual)


def test_dataset_comparison_reports_a_lost_lineage():
    expected, actual = make_dataset(), make_dataset()
    expected._import_state.lineage["org/parent@1.0.0"] = expected.metadata  # ty: ignore[invalid-assignment]
    with pytest.raises(AssertionError, match="lineage differs"):
        assert_datasets_equal(expected, actual)


def test_dataset_comparison_reports_reordered_record_annotations():
    expected, actual = make_dataset(), make_dataset()
    record = next(one for one in actual.records if len(one.annotations) > 1)
    record.annotations = tuple(reversed(record.annotations))
    with pytest.raises(AssertionError, match="annotations differ"):
        assert_datasets_equal(expected, actual)
