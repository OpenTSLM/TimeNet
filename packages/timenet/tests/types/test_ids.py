from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.dataset.edit import edit_version
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    ObjectKind,
    TimeSeriesSpec,
    Version,
    check_id,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _dataset():
    metadata = DatasetMetadata(
        dataset_id="timenet/ids",
        dataset_version=Version(1, 0, 0),
        name="Ids",
        description="test",
        license=License.CC_BY_4_0,
    )
    return TimeFDataset(metadata=metadata)


def _signal(name="a", time_axis=None):
    axis = time_axis or RegularAxis(period_us=Fraction(1000))
    return Signal(spec=SPEC, name=name, time_axis=axis, data=[1.0, 2.0])


def _record(*signals, source=None):
    return Record(sources=(source or Source(name="device", signals=signals or (_signal(),)),))


@pytest.mark.parametrize(
    "read_id",
    [lambda: Record().id, lambda: _signal().id, lambda: AnswerTask().id],
    ids=["record", "signal", "task"],
)
def test_reading_an_id_before_the_object_is_in_a_dataset_raises(read_id):
    with pytest.raises(TimeFValidationError, match="has no id yet; add it to a dataset"):
        read_id()


def test_add_record_numbers_records_and_signals_on_their_own_counters():
    dataset = _dataset()
    first = dataset.add_record(record=_record(_signal("a"), _signal("b")))
    second = dataset.add_record(record=_record())

    assert (first.id, second.id) == (1, 2)
    assert [signal.id for record in (first, second) for signal in record.signals] == [1, 2, 3]


def test_add_task_numbers_tasks_on_their_own_counter():
    dataset = _dataset()
    record = dataset.add_record(record=_record())
    tasks = dataset.add_tasks(tasks=[AnswerTask(inputs=(record,), targets=(str(i),)) for i in range(2)])

    assert [task.id for task in tasks] == [1, 2]


def test_a_rejected_task_batch_leaves_its_tasks_without_ids():
    dataset = _dataset()
    record = dataset.add_record(record=_record())
    valid = AnswerTask(inputs=(record,), targets=("yes",))
    invalid = AnswerTask(inputs=(record,))

    with pytest.raises(TimeFValidationError, match="needs an answer"):
        dataset.add_tasks(tasks=[valid, invalid])

    with pytest.raises(TimeFValidationError, match="has no id yet"):
        _ = valid.id
    assert dataset.add_task(task=valid).id == 1


def _shared_signal_record(dataset):
    return _record(dataset.records[0].signals[0])


def _shared_source_record(dataset):
    return _record(source=dataset.records[0].sources[0])


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda dataset: dataset.records[0], "record 1 is already in this dataset"),
        (_shared_signal_record, "each Signal has one owner"),
        (_shared_source_record, "each Source has one owner"),
    ],
    ids=["record", "signal", "source"],
)
def test_a_record_that_reuses_an_object_of_another_record_is_rejected(build, match):
    dataset = _dataset()
    dataset.add_record(record=_record())

    with pytest.raises(TimeFValidationError, match=match):
        dataset.add_record(record=build(dataset))
    assert len(dataset.records) == 1
    assert dataset.records[0].signals[0].id == 1


def test_streamed_tasks_get_stable_ids_by_stream_position(tmp_path):
    dataset = _dataset()
    record = dataset.add_record(record=_record())
    dataset.set_task_stream([AnswerTask], lambda: (AnswerTask(inputs=(record,), targets=(str(i),)) for i in range(3)))

    restored = TimeFDataset.open(path=dataset.write(path=tmp_path))

    assert {task.id: task.targets for task in restored.tasks} == {1: ("0",), 2: ("1",), 3: ("2",)}
    assert restored.last_ids[ObjectKind.TASK] == 3


def test_an_edit_continues_the_stored_counters_and_never_reuses_an_id(tmp_path):
    dataset = _dataset()
    for _ in range(3):
        dataset.add_record(record=_record())
    base_dir = dataset.write(path=tmp_path)

    edited_dir = edit_version(base_dir, dataset_version=Version(1, 1, 0), remove_record_ids=[3])
    edited = TimeFDataset.open(path=edited_dir)
    added = edited.add_record(record=_record())

    assert [record.id for record in edited.records] == [1, 2, 4]
    assert added.signals[0].id == 4
    assert edited.last_ids[ObjectKind.RECORD] == 4


def _storage_ids(version_dir):
    with TimeFReader(open_local_version(version_dir)) as reader:
        table = reader.read_annotation_table().to_pylist()
    return {(row["object_id"], row["key"]): (row["content_id"], row["occurrence_id"]) for row in table}


def test_the_writer_shares_annotation_content_by_value(tmp_path):
    dataset = _dataset()
    for value in ("A", "A", "B"):
        dataset.add_record(record=_record()).annotate(Annotation(key="cohort", value=value))

    ids = _storage_ids(dataset.write(path=tmp_path))

    assert ids[1, "cohort"][0] == ids[2, "cohort"][0] != ids[3, "cohort"][0]
    assert len({occurrence_id for _, occurrence_id in ids.values()}) == 3


def test_an_edit_numbers_storage_rows_from_one_again(tmp_path):
    dataset = _dataset()
    for value in ("A", "B", "C"):
        dataset.add_record(record=_record()).annotate(Annotation(key="cohort", value=value))
    base_dir = dataset.write(path=tmp_path)

    edited_dir = edit_version(base_dir, dataset_version=Version(1, 1, 0), remove_record_ids=[1])

    assert sorted(_storage_ids(edited_dir).values()) == [(1, 1), (2, 2)]


@pytest.mark.parametrize("value", [1, 42, 2**63 - 1, np.int64(7), np.uint32(7)])
def test_check_id_accepts_a_positive_int64(value):
    result = check_id(value, "record_id")
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "17", 1.0, 2**63])
def test_check_id_rejects_anything_else(value):
    with pytest.raises(TimeFValidationError, match="record_id must be a positive int64 id"):
        check_id(value, "record_id")
