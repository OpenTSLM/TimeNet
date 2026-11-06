from collections import UserList
from fractions import Fraction

import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import LoadPlan, TimeFReader
from timenet.testing import assert_round_trips
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


_SPEC = TimeSeriesSpec(spec_type="s", name="S", unit_value=ureg.dimensionless)


def _dataset() -> TimeFDataset:
    metadata = DatasetMetadata(
        dataset_id="timenet/annotations",
        dataset_version=Version(1, 0, 0),
        name="Annotations",
        description="test",
        license=License.CC_BY_4_0,
    )
    return TimeFDataset(metadata=metadata)


def _add_record(dataset: TimeFDataset) -> Record:
    signal = Signal(spec=_SPEC, name="a", time_axis=RegularAxis(period_us=Fraction(1000)), data=[1.0, 2.0])
    return dataset.add_record(record=Record(sources=(Source(name="device", signals=(signal,)),)))


def test_add_annotations_attaches_new_occurrences_in_order():
    dataset = _dataset()
    given = (Annotation(key="a", value=1), Annotation(key="b", value="x"))
    attached = dataset.add_annotations(given)
    assert [(one.key, one.value) for one in attached] == [("a", 1), ("b", "x")]
    assert all(new is not old for new, old in zip(attached, given, strict=True))
    assert dataset.annotations == attached


def test_add_annotations_attaches_nothing_if_one_fails():
    dataset = _dataset()
    dataset.annotate(Annotation(key="kept", value=1))
    batch = [Annotation(key="a", value=1), Annotation(key="b", value=2, span=TimePoint.seconds(1.0))]
    with pytest.raises(TimeFValidationError, match="cannot have a time span"):
        dataset.add_annotations(batch)
    assert [one.key for one in dataset.annotations] == ["kept"]


def test_add_annotations_takes_a_mapping_of_keys_to_values():
    dataset = _dataset()
    attached = dataset.add_annotations({"sampling_rate_hz": 50, "device": "wrist", "calibrated": True})
    assert [(one.key, read_value(one.value)) for one in attached] == [
        ("sampling_rate_hz", 50),
        ("device", "wrist"),
        ("calibrated", True),
    ]
    assert dataset.annotations == attached


def test_record_add_annotations_takes_a_mapping_of_keys_to_values():
    record = _add_record(_dataset())
    attached = record.add_annotations({"sensor_location": "wrist", "age": 54})
    assert [(one.key, read_value(one.value)) for one in record.annotations] == [
        ("sensor_location", "wrist"),
        ("age", 54),
    ]
    assert record.annotations == attached


@pytest.mark.parametrize(
    ("mapping", "message"),
    [({"a": 1, "b": {"nested": 1}}, "unsupported annotation value type"), ({"a": 1, 3: "x"}, "key must be text")],
)
def test_add_annotations_attaches_nothing_from_a_bad_mapping(mapping, message):
    dataset = _dataset()
    record = _add_record(dataset)
    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_annotations(mapping)
    with pytest.raises(TimeFValidationError, match=message):
        record.add_annotations(mapping)
    assert dataset.annotations == ()
    assert record.annotations == ()


@pytest.mark.parametrize(
    ("batch", "message"),
    [("ab", "not str"), ([Annotation(key="a", value=1), "b"], "must be an Annotation; got str")],
)
def test_add_annotations_rejects_a_batch_of_other_objects(batch, message):
    dataset = _dataset()
    record = _add_record(dataset)
    with pytest.raises(TimeFValidationError, match=message):
        dataset.add_annotations(batch)
    with pytest.raises(TimeFValidationError, match=message):
        record.add_annotations(batch)
    assert dataset.annotations == ()
    assert record.annotations == ()


def test_shared_attaches_each_content_once():
    dataset = _dataset()
    first = dataset.annotate_shared(Annotation(key="question_type", value="trend"))
    again = dataset.annotate_shared(Annotation(key="question_type", value="trend"))
    other = dataset.annotate_shared(Annotation(key="question_type", value="count"))
    assert again is first
    assert other is not first
    assert dataset.annotations == (first, other)


@pytest.mark.parametrize(
    "variant",
    [
        Annotation(key="n", value=True),
        Annotation(key="n", value=1.0),
        Annotation(key="n", value=1, unit="year"),
        Annotation(key="n", value=1, description="one"),
        Annotation(key="n", value=1, metadata={"source": "csv"}),
        Annotation(key="n", value=1, occurrence_metadata={"rater": "a"}),
        Annotation(key="n", value=1, source="model"),
        Annotation(key="n", value=1, confidence=0.5),
        Annotation(key="m", value=1),
    ],
    ids=["bool", "float", "unit", "description", "metadata", "occurrence-metadata", "source", "confidence", "key"],
)
def test_shared_tells_content_apart(variant):
    dataset = _dataset()
    base = dataset.annotate_shared(Annotation(key="n", value=1))
    assert dataset.annotate_shared(variant) is not base
    assert len(dataset.annotations) == 2


def test_shared_finds_annotations_attached_another_way():
    dataset = _dataset()
    annotated = dataset.annotate(Annotation(key="a", value=1))
    (added,) = dataset.add_annotations([Annotation(key="b", value=2)])
    assert dataset.annotate_shared(Annotation(key="a", value=1)) is annotated
    dataset.annotate(Annotation(key="c", value=3))
    assert dataset.annotate_shared(Annotation(key="b", value=2)) is added
    assert dataset.annotate_shared(Annotation(key="c", value=3)) is dataset.annotations[2]
    assert len(dataset.annotations) == 3


def test_shared_rejects_a_time_span():
    with pytest.raises(TimeFValidationError, match="cannot have a time span"):
        _dataset().annotate_shared(Annotation(key="a", value=1, span=TimePoint.seconds(1.0)))


def test_shared_rejects_a_time_span_for_content_it_already_has():
    dataset = _dataset()
    dataset.annotate_shared(Annotation(key="a", value=1))
    with pytest.raises(TimeFValidationError, match="cannot have a time span"):
        dataset.annotate_shared(Annotation(key="a", value=1, span=TimePoint.seconds(1.0)))


def test_shared_finds_long_text_after_a_read(tmp_path):
    long_text = "options: " + "a, " * 5000
    dataset = _dataset()
    record = _add_record(dataset)
    options = dataset.annotate_shared(Annotation(key="answer_options", value=long_text))
    dataset.add_task(task=AnswerTask(inputs=(record,), targets=("a",), input_annotations=(options,)))
    with TimeFReader.open(dataset.write(path=tmp_path / "out")) as reader:
        reread = reader.read(load=LoadPlan.ALL_EAGER)
    assert reread.annotate_shared(Annotation(key="answer_options", value=long_text)) is reread.annotations[0]
    assert len(reread.annotations) == 1


def test_shared_annotations_are_stored_once_and_found_after_a_read(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    for kind in ("trend", "count", "trend", "trend"):
        question_type = dataset.annotate_shared(Annotation(key="question_type", value=kind))
        dataset.add_task(task=AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(question_type,)))

    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert len(restored.annotations) == 2
    assert {read_value(task.input_annotations[0].value) for task in restored.tasks} == {"trend", "count"}

    with TimeFReader.open(dataset.write(path=tmp_path / "out")) as reader:
        reread = reader.read(load=LoadPlan.ALL_EAGER)
    stored = {read_value(one.value): one for one in reread.annotations}
    assert reread.annotate_shared(Annotation(key="question_type", value="count")) is stored["count"]
    assert len(reread.annotations) == 2


def test_shared_annotations_are_stored_in_the_order_of_first_use(tmp_path):
    kinds = ["count", "trend", "count", "range", "trend", "count"]
    dataset = _dataset()
    record = _add_record(dataset)
    for kind in kinds:
        question_type = dataset.annotate_shared(Annotation(key="question_type", value=kind))
        dataset.add_task(task=AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(question_type,)))
    with TimeFReader.open(dataset.write(path=tmp_path / "out")) as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert [read_value(one.value) for one in restored.annotations] == ["count", "trend", "range"]
    assert [read_value(task.input_annotations[0].value) for task in restored.tasks] == kinds


def _streamed_dataset(kinds_before_stream, kinds_in_stream):
    dataset = _dataset()
    record = _add_record(dataset)
    for kind in kinds_before_stream:
        dataset.annotate_shared(Annotation(key="question_type", value=kind))

    def tasks():
        for kind in kinds_in_stream:
            question_type = dataset.annotate_shared(Annotation(key="question_type", value=kind))
            yield AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(question_type,))

    dataset.set_task_stream([AnswerTask], tasks)
    return dataset


def test_a_task_stream_can_look_up_shared_annotations_attached_before_it(tmp_path):
    dataset = _streamed_dataset(["trend", "count"], ["trend", "count", "trend"])
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert len(restored.annotations) == 2
    assert [read_value(task.input_annotations[0].value) for task in restored.tasks] == ["trend", "count", "trend"]


def test_a_task_stream_can_attach_new_shared_annotations_once(tmp_path):
    dataset = _streamed_dataset(["trend"], ["trend", "count", "count"])
    # The round trip reads the stream twice: once to write it, once to compare it.
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert [read_value(one.value) for one in dataset.annotations] == ["trend", "count"]
    assert [read_value(one.value) for one in restored.annotations] == ["trend", "count"]
    assert [read_value(task.input_annotations[0].value) for task in restored.tasks] == ["trend", "count", "count"]


def test_shared_attaches_new_content_while_a_stream_iterator_is_open():
    dataset = _streamed_dataset(["trend"], ["trend", "trend"])
    tasks = dataset.iter_tasks()
    next(tasks)
    assert read_value(dataset.annotate_shared(Annotation(key="question_type", value="count")).value) == "count"
    assert len(dataset.annotations) == 2


def test_a_stream_run_again_after_an_error_finds_what_the_first_run_attached(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    fail = [True]

    def tasks():
        for kind in ("trend", "count"):
            question_type = dataset.annotate_shared(Annotation(key="question_type", value=kind))
            yield AnswerTask(inputs=(record,), targets=(kind,), input_annotations=(question_type,))
            if fail[0]:
                raise RuntimeError("the source broke")

    dataset.set_task_stream([AnswerTask], tasks)
    with pytest.raises(RuntimeError, match="the source broke"):
        dataset.write(path=tmp_path / "first")
    assert [read_value(one.value) for one in dataset.annotations] == ["trend"]
    fail[0] = False
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert [read_value(one.value) for one in restored.annotations] == ["trend", "count"]


def test_add_annotations_attaches_inside_a_stream_factory(tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    attached: list[Annotation] = []

    def tasks():
        if not attached:
            attached.extend(dataset.add_annotations({"source_kind": "survey", "wave": 2}))
        return iter([AnswerTask(inputs=(record,), targets=("x",), input_annotations=tuple(attached))])

    dataset.set_task_stream([AnswerTask], tasks)
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert [read_value(one.value) for one in restored.tasks[0].input_annotations] == ["survey", 2]


def test_add_annotations_takes_lists_and_tuples_of_text_but_not_other_sequences():
    dataset = _dataset()
    attached = dataset.add_annotations({"tags": ["a", "b"], "pair": ("c", "d")})
    assert [read_value(one.value) for one in attached] == [["a", "b"], ["c", "d"]]
    with pytest.raises(TimeFValidationError, match="unsupported annotation value type: UserList"):
        dataset.add_annotations({"tags": UserList(["a", "b"])})  # ty: ignore[invalid-argument-type]
    assert len(dataset.annotations) == 2


def _attach_in_a_generator(dataset, record):
    def tasks():
        question_type = dataset.annotate_shared(Annotation(key="question_type", value="made in the stream"))
        yield AnswerTask(inputs=(record,), targets=("x",), input_annotations=(question_type,))

    return tasks


def _attach_in_a_plain_function(dataset, record):
    def tasks():
        question_type = dataset.annotate_shared(Annotation(key="question_type", value="made in the factory"))
        return iter([AnswerTask(inputs=(record,), targets=("x",), input_annotations=(question_type,))])

    return tasks


@pytest.mark.parametrize("factory", [_attach_in_a_generator, _attach_in_a_plain_function])
def test_a_stream_factory_can_attach_shared_content(factory, tmp_path):
    dataset = _dataset()
    record = _add_record(dataset)
    dataset.set_task_stream([AnswerTask], factory(dataset, record))
    list(dataset.iter_tasks())
    restored = assert_round_trips(dataset, tmp_path / "registry")
    assert len(dataset.annotations) == len(restored.annotations) == 1
