from collections.abc import Iterator
from pathlib import Path

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import CONTROL_FILE, connect_control
from timenet.reader import LoadPlan, TimeFReader
from timenet.types import (
    Annotation,
    AnswerTask,
    DatasetMetadata,
    License,
    TimePoint,
    TimeSeriesSpec,
    Version,
    read_value,
    ureg,
)
from timenet.types.text import LazyText, LazyTextList


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
_KINDS = ("count", "trend", "count", "range", "trend")


def _dataset() -> TimeFDataset:
    return TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="timenet/shared-annotations",
            dataset_version=Version(1, 0, 0),
            name="Shared annotations",
            description="Shared annotation contract fixture",
            license=License.CC_BY_4_0,
        )
    )


def _add_record(dataset: TimeFDataset) -> Record:
    signal = Signal.from_values([1.0, 2.0], spec=_SPEC, name="s", time_axis=RegularAxis.from_rate_hz(1))
    return dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))


def _streamed_dataset() -> TimeFDataset:
    dataset = _dataset()
    record = _add_record(dataset)

    def tasks() -> Iterator[AnswerTask]:
        for kind in _KINDS:
            annotation = dataset.annotate(Annotation(key="question_type", value=kind))
            yield AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(annotation,))

    dataset.set_task_stream([AnswerTask], tasks)
    return dataset


def _annotation_rows(release: Path) -> list[list[tuple]]:
    """Return the stored order of shared values, their occurrences, and the task references to them."""
    with connect_control(release / CONTROL_FILE, read_only=True) as connection:
        return [
            connection.execute(query).fetchall()
            for query in (
                "SELECT content_id, name, text_value FROM annotation_contents ORDER BY content_id",
                "SELECT occurrence_id, content_id, object_type FROM annotation_occurrences ORDER BY occurrence_id",
                "SELECT task_id, field, position, occurrence_id FROM task_annotation_refs"
                " ORDER BY task_id, field, position",
            )
        ]


def test_annotate_returns_the_first_occurrence_for_equal_annotations():
    dataset = _dataset()
    given = Annotation(key="question_type", value="trend")
    first = dataset.annotate(given)
    assert first is not given
    assert dataset.annotate(Annotation(key="question_type", value="trend")) is first
    other = dataset.annotate(Annotation(key="question_type", value="count"))
    assert dataset.annotate(Annotation(key="question_type", value="count")) is other
    assert [read_value(one.value) for one in dataset.annotations] == ["trend", "count"]


@pytest.mark.parametrize(
    "variant",
    [
        Annotation(key="n", value=True),
        Annotation(key="n", value=1.0),
        Annotation(key="n", value=1, unit="year"),
        Annotation(key="n", value=1, description="one"),
        Annotation(key="n", value=1, metadata={"source": "csv"}),
        Annotation(key="n", value=1, metadata={"details": {}}),
        Annotation(key="n", value=1, occurrence_metadata={"rater": "a"}),
        Annotation(key="n", value=1, occurrence_metadata={"details": {}}),
        Annotation(key="n", value=1, source="model"),
        Annotation(key="n", value=1, confidence=0.5),
        Annotation(key="m", value=1),
    ],
)
def test_sharing_preserves_every_stored_field_and_value_type(variant):
    dataset = _dataset()
    base = dataset.annotate(Annotation(key="n", value=1))
    assert dataset.annotate(variant) is not base
    assert len(dataset.annotations) == 2


def test_sharing_distinguishes_signed_zero_and_reuses_nan():
    dataset = _dataset()
    positive = dataset.annotate(Annotation(key="n", value=0.0))
    assert dataset.annotate(Annotation(key="n", value=-0.0)) is not positive
    nan = dataset.annotate(Annotation(key="n", value=float("nan")))
    assert dataset.annotate(Annotation(key="n", value=float("nan"))) is nan


def test_sharing_matches_metadata_independently_of_mapping_order():
    dataset = _dataset()
    first = dataset.annotate(Annotation(key="n", value=1, metadata={"a": [1], "b": {"c": True}}))
    assert dataset.annotate(Annotation(key="n", value=1, metadata={"b": {"c": True}, "a": [1]})) is first


def test_shared_rejects_spans_even_when_the_value_matches():
    dataset = _dataset()
    dataset.annotate(Annotation(key="n", value=1))
    with pytest.raises(TimeFValidationError, match="cannot have a time span"):
        dataset.annotate(Annotation(key="n", value=1, span=TimePoint.seconds(1)))
    assert len(dataset.annotations) == 1


def test_sharing_does_not_load_unrelated_text_or_a_different_value_type():
    def refuse():
        raise AssertionError("unrelated lazy text must stay unread")

    dataset = _dataset()
    dataset.annotate(Annotation(key="notes", value=LazyText(refuse)))
    dataset.annotate(Annotation(key="question_type", value=LazyTextList(refuse)))
    first = dataset.annotate(Annotation(key="question_type", value="trend"))
    assert dataset.annotate(Annotation(key="question_type", value="trend")) is first


def test_streamed_sharing_preserves_values_and_references_through_storage(tmp_path):
    dataset = _streamed_dataset()
    release = dataset.write(path=tmp_path)
    with connect_control(release / CONTROL_FILE, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    with TimeFReader.open(release) as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert [read_value(one.value) for one in restored.annotations] == ["count", "trend", "range"]
    assert [read_value(task.input_annotations[0].value) for task in restored.tasks] == list(_KINDS)
    assert restored.tasks[0].input_annotations[0] is restored.tasks[2].input_annotations[0]


def test_inspection_and_repeated_writes_preserve_storage_order_and_ids(tmp_path):
    dataset = _streamed_dataset()
    expected = _annotation_rows(dataset.write(path=tmp_path / "fresh"))
    assert _annotation_rows(dataset.write(path=tmp_path / "again")) == expected
    inspected = _streamed_dataset()
    tasks = inspected.iter_tasks()
    next(tasks)
    next(tasks)
    assert _annotation_rows(inspected.write(path=tmp_path / "inspected")) == expected


def test_a_stream_factory_can_attach_shared_input_and_target_annotations(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)

    def tasks():
        annotation = dataset.annotate(Annotation(key="label", value="a"))
        return iter([AnswerTask(inputs=(record,), input_annotations=(annotation,), target_annotations=(annotation,))])

    dataset.set_task_stream([AnswerTask], tasks)
    assert dataset.annotations == ()
    with TimeFReader.open(dataset.write(path=tmp_path)) as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    (stored,) = restored.annotations
    assert read_value(stored.value) == "a"
    task = restored.tasks[0]
    assert task.input_annotations == (stored,)
    assert task.target_annotations == (stored,)
    assert task.input_annotations[0] is task.target_annotations[0] is stored


def test_a_failed_stream_keeps_occurrences_for_retry_but_publishes_nothing(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    fail = True

    def tasks():
        for kind in _KINDS:
            annotation = dataset.annotate(Annotation(key="question_type", value=kind))
            yield AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(annotation,))
            if fail:
                raise RuntimeError("source unavailable")

    dataset.set_task_stream([AnswerTask], tasks)
    with pytest.raises(RuntimeError, match="source unavailable"):
        dataset.write(path=tmp_path)
    assert not (tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)).exists()
    (first,) = dataset.annotations
    fail = False
    release = dataset.write(path=tmp_path)
    assert dataset.annotations[0] is first
    assert _annotation_rows(release) == _annotation_rows(_streamed_dataset().write(path=tmp_path / "fresh"))


def test_sharing_does_not_make_a_loose_equal_annotation_a_valid_reference(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    annotation = Annotation(key="question_type", value="trend")
    dataset.annotate(annotation)
    dataset.set_task_stream(
        [AnswerTask], lambda: iter([AnswerTask(inputs=(record,), targets=("x",), input_annotations=(annotation,))])
    )
    with pytest.raises(TimeFValidationError, match="not attached to its dataset"):
        dataset.write(path=tmp_path)
