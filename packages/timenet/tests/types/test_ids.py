from fractions import Fraction

import numpy as np
import pytest

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.dataset.edit import edit_version
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    IdKind,
    License,
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


def _signal(name="a", time_axis=None, **kwargs):
    axis = time_axis or RegularAxis(period_us=Fraction(1000))
    return Signal(spec=SPEC, name=name, time_axis=axis, data=[1.0, 2.0], **kwargs)


def _record(*signals, **kwargs):
    return Record(sources=(Source(name="device", signals=signals or (_signal(),)),), **kwargs)


@pytest.mark.parametrize(
    "build",
    [Record, lambda: Source(name="s"), _signal, OrdinalAxis, lambda: Annotation(key="k", value=1), AnswerTask],
)
def test_an_object_has_no_id_until_it_is_added(build):
    assert build().id is None


def test_add_record_numbers_each_kind_on_its_own():
    dataset = _dataset()
    shared_axis = RegularAxis(period_us=Fraction(1000))
    first = _record(_signal("a", time_axis=shared_axis), _signal("b", time_axis=shared_axis))
    first.annotate(Annotation(key="age", value=1))
    dataset.add_record(record=first)
    second = dataset.add_record(record=_record())

    assert (first.id, second.id) == (1, 2)
    assert [source.id for record in (first, second) for source in record.sources] == [1, 2]
    assert [signal.id for record in (first, second) for signal in record.signals] == [1, 2, 3]
    assert shared_axis.id == 1
    assert (first.annotations[0].id, first.annotations[0].occurrence_id) == (1, 1)


def test_tasks_and_dataset_annotations_get_ids_when_added():
    dataset = _dataset()
    record = dataset.add_record(record=_record())
    task = AnswerTask(inputs=(record,), targets=("yes",))
    task.annotate(Annotation(key="source", value="rater"))
    dataset.add_task(task=task)
    on_record = record.annotate(Annotation(key="age", value=1))
    on_dataset = dataset.annotate(Annotation(key="site", value="clinic"))

    assert task.id == 1
    assert [annotation.occurrence_id for annotation in (task.annotations[0], on_record, on_dataset)] == [1, 2, 3]


def test_one_annotation_on_two_records_shares_its_content_id():
    dataset = _dataset()
    first, second = _record(), _record()
    cohort = Annotation(key="cohort", value="A")
    occurrences = (first.annotate(cohort), second.annotate(cohort))
    dataset.add_record(record=first)
    dataset.add_record(record=second)

    assert occurrences[0].id == occurrences[1].id == cohort.id
    assert occurrences[0].occurrence_id != occurrences[1].occurrence_id


def test_an_explicit_id_is_kept_and_later_ids_continue_above_it():
    dataset = _dataset()
    explicit = dataset.add_record(record=_record(id=40))
    generated = dataset.add_record(record=_record())

    assert (explicit.id, generated.id) == (40, 41)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: _record(id=1), "record id 1 is already registered"),
        (lambda: _record(_signal(id=1)), "each Signal has one owner"),
        (lambda: Record(sources=(Source(name="device", id=1, signals=(_signal(),)),)), "each Source has one owner"),
    ],
)
def test_an_explicit_id_that_collides_within_its_kind_is_rejected(build, match):
    dataset = _dataset()
    dataset.add_record(record=_record(_signal(id=1), id=1))
    record = build()

    with pytest.raises(TimeFValidationError, match=match):
        dataset.add_record(record=record)
    assert len(dataset.records) == 1


def test_a_rejected_task_batch_leaves_its_ids_unset():
    dataset = _dataset()
    record = dataset.add_record(record=_record())
    valid = AnswerTask(inputs=(record,), targets=("yes",))
    invalid = AnswerTask(inputs=(record,))

    with pytest.raises(TimeFValidationError, match="needs an answer"):
        dataset.add_tasks(tasks=[valid, invalid])

    assert valid.id is None
    assert dataset.add_task(task=valid).id == 1


def test_an_edit_continues_the_stored_counters_and_never_reuses_an_id(tmp_path):
    dataset = _dataset()
    for _ in range(3):
        dataset.add_record(record=_record())
    base_dir = dataset.write(path=tmp_path)

    edited_dir = edit_version(base_dir, tmp_path, dataset_version=Version(1, 1, 0), remove_record_ids=[3])
    edited = TimeFDataset.open(path=edited_dir)
    added = edited.add_record(record=_record())

    assert [record.id for record in edited.records] == [1, 2, 4]
    assert added.signals[0].id == 4
    assert edited.last_ids[IdKind.RECORD] == 4


ENTITIES = [
    lambda id: Record(id=id),
    lambda id: Source(name="s", id=id),
    lambda id: _signal(id=id),
    lambda id: RegularAxis(period_us=Fraction(1000), id=id),
    lambda id: IrregularAxis(first_us=0, last_us=10, id=id),
    lambda id: OrdinalAxis(id=id),
    lambda id: Annotation(key="k", value=1, id=id),
    lambda id: AnswerTask(id=id),
]


@pytest.mark.parametrize("build", ENTITIES)
def test_every_entity_keeps_a_valid_explicit_id_and_rejects_an_invalid_one(build):
    assert build(42).id == 42
    for bad in ("42", 0, True):
        with pytest.raises(TimeFValidationError, match="positive int64 id"):
            build(bad)


@pytest.mark.parametrize("value", [1, 42, 2**63 - 1, np.int64(7), np.uint32(7)])
def test_check_id_accepts_a_positive_int64(value):
    result = check_id(value, "record_id")
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [True, False, 0, -1, "17", 1.0, 2**63])
def test_check_id_rejects_anything_else(value):
    with pytest.raises(TimeFValidationError, match="record_id must be a positive int64 id"):
        check_id(value, "record_id")


def test_check_id_names_an_unset_id():
    with pytest.raises(TimeFValidationError, match="record_id is not set; add the object to a dataset"):
        check_id(None, "record_id")
