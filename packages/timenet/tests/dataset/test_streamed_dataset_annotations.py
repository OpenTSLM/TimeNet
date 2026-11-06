"""A task stream that attaches the dataset annotations its tasks refer to."""

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.control_audit import audit_control_database
from timenet.format.duckdb import connect_control
from timenet.reader import LoadPlan, TimeFReader
from timenet.testing import assert_datasets_equal, assert_round_trips
from timenet.types import Annotation, AnswerTask, DatasetMetadata, License, TimeSeriesSpec, Version, read_value, ureg


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)
_ROWS = ("a", "b", "a", "c", "b")


def _dataset() -> tuple[TimeFDataset, Record]:
    metadata = DatasetMetadata(
        dataset_id="timenet/streamed-annotations",
        dataset_version=Version(1, 0, 0),
        name="Streamed annotations",
        description="test",
        license=License.CC_BY_4_0,
    )
    dataset = TimeFDataset(metadata=metadata)
    signal = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[1.0, 2.0])
    record = dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))
    return dataset, record


def _attach_once(dataset: TimeFDataset) -> Callable[[str], Annotation]:
    """Return a function that attaches each value on its first use and returns that occurrence after."""
    occurrences: dict[str, Annotation] = {}

    def attach(value: str) -> Annotation:
        if value not in occurrences:
            occurrences[value] = dataset.annotate(Annotation(key="question_type", value=value))
        return occurrences[value]

    return attach


def _streaming(rows: tuple[str, ...] = _ROWS) -> TimeFDataset:
    """Return a dataset whose stream attaches one annotation when it starts and one per new row value.

    The tasks of even rows refer to the value as an input, and the others as their target.
    """
    dataset, record = _dataset()
    attach = _attach_once(dataset)

    def stream() -> Iterator[AnswerTask]:
        header = attach("created")
        return (
            AnswerTask(inputs=(record,), targets=("x",), input_annotations=(header, attach(value)))
            if position % 2 == 0
            else AnswerTask(inputs=(record,), input_annotations=(header,), target_annotations=(attach(value),))
            for position, value in enumerate(rows)
        )

    dataset.set_task_stream([AnswerTask], stream)
    return dataset


def _values(annotations: tuple[Annotation, ...]) -> list[object]:
    return [read_value(annotation.value) for annotation in annotations]


def _annotation_rows(release: Path) -> list[list[tuple]]:
    with connect_control(release / "control.duckdb", read_only=True) as connection:
        return [
            connection.execute(query).fetchall()
            for query in (
                "SELECT * FROM annotation_contents ORDER BY content_id",
                "SELECT * FROM annotation_occurrences ORDER BY occurrence_id",
                "SELECT * FROM task_annotation_refs ORDER BY task_id, field, position",
            )
        ]


def test_a_stream_can_attach_the_dataset_annotations_its_tasks_use(tmp_path):
    dataset = _streaming()
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert _values(dataset.annotations) == ["created", "a", "b", "c"]
    assert _values(restored.annotations) == ["created", "a", "b", "c"]
    tasks = restored.tasks
    assert [_values(task.input_annotations) for task in tasks[::2]] == [
        ["created", "a"],
        ["created", "a"],
        ["created", "b"],
    ]
    assert [_values(task.target_annotations) for task in tasks[1::2]] == [["b"], ["c"]]
    # Every task refers to the one occurrence of its value, not to a copy.
    assert tasks[0].input_annotations[1] is tasks[2].input_annotations[1]


def test_the_release_passes_the_relational_audit_and_reads_back_whole(tmp_path):
    dataset = _streaming()
    release = dataset.write(path=tmp_path)
    with connect_control(release / "control.duckdb", read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    with TimeFReader.open(release) as reader:
        reader.verify()
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert_datasets_equal(dataset, restored)


def test_repeated_writes_and_fresh_datasets_store_the_same_annotations(tmp_path):
    dataset = _streaming()
    first = _annotation_rows(dataset.write(path=tmp_path / "first"))
    assert _annotation_rows(dataset.write(path=tmp_path / "again")) == first
    assert _annotation_rows(_streaming().write(path=tmp_path / "fresh")) == first


def test_a_partly_read_stream_then_writes_every_annotation(tmp_path):
    dataset = _streaming()
    tasks = dataset.iter_tasks()
    next(tasks)
    next(tasks)
    assert _values(dataset.annotations) == ["created", "a", "b"]
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert _values(restored.annotations) == ["created", "a", "b", "c"]


def test_a_stream_that_raises_publishes_nothing_and_keeps_what_it_attached(tmp_path):
    dataset, record = _dataset()
    attach = _attach_once(dataset)

    def stream() -> Iterator[AnswerTask]:
        yield AnswerTask(inputs=(record,), targets=("x",), input_annotations=(attach("a"),))
        attach("b")
        raise RuntimeError("the source broke")

    dataset.set_task_stream([AnswerTask], stream)
    with pytest.raises(RuntimeError, match="the source broke"):
        dataset.write(path=tmp_path)
    assert not (tmp_path / dataset.metadata.dataset_id / "1.0.0").exists()
    # As after any annotate call, the occurrences stay on the dataset.
    assert _values(dataset.annotations) == ["a", "b"]


def test_a_stream_of_many_tasks_attaches_one_occurrence_per_value(tmp_path):
    rows = tuple("abc"[position % 3] for position in range(300))
    dataset = _streaming(rows)
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert len(dataset.annotations) == len(restored.annotations) == 4
    assert len(restored.tasks) == 300


def test_a_streamed_task_still_cannot_use_an_annotation_that_nothing_holds(tmp_path):
    dataset, record = _dataset()
    loose = Annotation(key="question_type", value="a")
    dataset.set_task_stream(
        [AnswerTask], lambda: iter((AnswerTask(inputs=(record,), targets=("x",), input_annotations=(loose,)),))
    )
    with pytest.raises(TimeFValidationError, match="not attached to its dataset"):
        dataset.write(path=tmp_path)


def _attaching_once(device: str) -> TimeFDataset:
    """Return a dataset whose stream attaches one ``device`` annotation on its first pass, and no task uses it."""
    dataset, record = _dataset()
    attach = _attach_once(dataset)

    def stream() -> Iterator[AnswerTask]:
        if not dataset.annotations:
            dataset.annotate(Annotation(key="device", value=device))
        attach("a")
        return iter([AnswerTask(inputs=(record,), targets=("x",))])

    dataset.set_task_stream([AnswerTask], stream)
    return dataset


def test_dataset_comparison_sees_annotations_that_the_streams_attach():
    with pytest.raises(AssertionError, match="dataset annotations differ"):
        assert_datasets_equal(_attaching_once("wrist"), _attaching_once("ankle"))


def test_a_fresh_streamed_dataset_equals_its_release(tmp_path):
    release = _attaching_once("wrist").write(path=tmp_path)
    with TimeFReader.open(release) as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert_datasets_equal(_attaching_once("wrist"), restored)
