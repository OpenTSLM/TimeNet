"""Public inspection relationships, pagination, and annotation ownership."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pytest

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping
from timenet.testing import make_dataset
from timenet.types import Annotation, AnswerTask, TimeInterval, TimePoint, TimeSeriesSpec
from timenet.viewer.schemas import AnnotationQuery, AnnotationWindowQuery, RecordDetailQuery, TaskDetailQuery


def test_task_references_use_public_ids(reader):
    inspection = reader.inspect()
    detail = inspection.task_detail(TaskDetailQuery(task_id="task-answer-0"))
    assert detail is not None
    assert detail["input_record_ids"] == ["record-0"]
    assert detail["parent_task_ids"] == ["task-cls-0"]
    assert [(item["kind"], item["value"]) for item in detail["targets"]] == [("text", "Normal.")]
    assert [(item["object_type"], item["object_id"], item["name"]) for item in detail["annotation_refs"]] == [
        ("Record", "record-0", "cohort")
    ]
    localized = inspection.task_detail(TaskDetailQuery(task_id="task-localize-0"))
    assert localized is not None
    assert [(item["kind"], item["start"], item["end"], item["signal_ids"]) for item in localized["targets"]] == [
        ("time_point", "500000", None, []),
        ("time_interval", "0", "250000", ["ts-shared"]),
    ]


def test_task_target_pages_preserve_order(open_dataset):
    dataset = make_dataset()
    dataset.add_task(task=AnswerTask(id="many", inputs=(dataset.records[0],), targets=("first", "second", "third")))
    inspection = open_dataset(dataset).inspect()
    first = inspection.task_detail(TaskDetailQuery(task_id="many", limit=2))
    assert first is not None
    assert [item["value"] for item in first["targets"]] == ["first", "second"]
    assert first["next_after"] == 2
    last = inspection.task_detail(TaskDetailQuery(task_id="many", limit=2, after=first["next_after"]))
    assert last is not None
    assert [item["value"] for item in last["targets"]] == ["third"]
    assert last["next_after"] is None


def test_record_pages_do_not_repeat_or_omit_ids(reader):
    inspection = reader.inspect()
    first = inspection.records_page(limit=2)
    second = inspection.records_page(limit=2, after=first.next_after)
    assert [item.record_id for item in first.items + second.items] == ["record-0", "record-1", "record-2"]
    assert second.next_after is None


def test_source_pages_continue_past_the_default_limit(open_dataset):
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(
        record=Record(
            record_id="record", sources=tuple(Source(id=f"source-{i:03}", name=f"Source {i}") for i in range(51))
        )
    )
    inspection = open_dataset(dataset).inspect()
    first = inspection.sources_page("record", None)
    second = inspection.sources_page("record", None, after=first.next_after)
    assert [item.source_id for item in first.items + second.items] == [f"source-{i:03}" for i in range(51)]
    assert second.next_after is None


def test_annotation_filters_select_owner_and_value(open_dataset):
    # Reuse an ID across owner types so the type filter cannot hide behind the ID filter.
    source = Source(id="record-0", name="Sensor")
    source.annotate(Annotation(key="wrong-type", value="A"))
    record = Record(record_id="record-0", sources=(source,), time_span=TimeInterval.seconds(0, 2))
    record.add_annotations(
        [
            Annotation(key="cohort", value="A"),
            Annotation(key="wrong-value", value="B"),
            Annotation(key="wrong-span", value="A", span=TimePoint.seconds(1)),
        ]
    )
    other = Record(record_id="record-1", sources=(Source(id="other-source", name="Other sensor"),))
    other.annotate(Annotation(key="wrong-owner", value="A"))
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(record=record)
    dataset.add_record(record=other)
    page = (
        open_dataset(dataset)
        .inspect()
        .browse_annotations(
            AnnotationQuery(object_type="Record", object_id="record-0", span_type="static", value_query="A")
        )
    )
    assert [
        (item["object_type"], item["object_id"], item["span_type"], item["name"], item["value"])
        for item in page["items"]
    ] == [("Record", "record-0", "static", "cohort", "A")]


def test_annotation_window_filters_before_paging_and_includes_ancestors(open_dataset):
    numeric = Signal(
        id="signal",
        name="Signal",
        spec=TimeSeriesSpec(spec_type="numeric", name="Numeric", unit_value=None),
        time_axis=RegularAxis.from_rate_hz(1),
        data=pa.array([float(i) for i in range(200)], type=pa.float32()),
    )
    for second in range(101):
        numeric.annotate(Annotation(key=f"event-{second}", span=TimePoint.seconds(second)))
    source = Source(id="parent", name="Parent", sources=(Source(id="child", name="Child", signals=(numeric,)),))
    source.annotate(Annotation(key="source-event", span=TimePoint.seconds(100)))
    record = Record(record_id="record", sources=(source,))
    record.annotate(Annotation(key="static", value=True))
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    dataset.add_record(record=record)
    inspection = open_dataset(dataset).inspect()
    query = AnnotationWindowQuery(record_id="record", signal_id="signal", start=100, stop=101, limit=2)
    first = inspection.window_annotations(query)
    assert len(first["items"]) == 2
    assert first["next_cursor"] is not None
    second = inspection.window_annotations(query.model_copy(update={"after": first["next_cursor"]}))
    assert len(second["items"]) == 1
    assert len({item["occurrence_id"] for item in first["items"] + second["items"]}) == 3
    assert {(item["object_type"], item["name"]) for item in first["items"] + second["items"]} == {
        ("Signal", "event-100"),
        ("Source", "source-event"),
        ("Record", "static"),
    }
    assert second["next_cursor"] is None
    context = inspection.record_detail(RecordDetailQuery(record_id="record", source_id="parent"))
    assert context["source_path"] == ["parent", "child"]
    assert context["signal"]["signal_id"] == "signal"


def test_large_task_fields_are_explicit_previews(open_dataset):
    dataset = make_dataset()
    prompt = "Prompt: " + "p" * 600
    target = "Answer: " + "x" * 5000
    metadata: JsonMapping = {"text": "Metadata: " + "m" * 20000}
    dataset.add_task(
        task=AnswerTask(
            id="large",
            inputs=(dataset.records[0],),
            prompt=prompt,
            targets=(target,),
            metadata=metadata,
        )
    )
    detail = open_dataset(dataset).inspect().task_detail(TaskDetailQuery(task_id="large"))
    assert detail is not None
    assert detail["prompt_truncated"]
    assert detail["prompt"] == prompt[:512]
    assert detail["metadata_truncated"]
    assert detail["metadata"] == json.dumps(metadata, separators=(",", ":"))[:16384]
    assert detail["targets"][0]["value_truncated"]
    assert detail["targets"][0]["value"] == target[:4096]


@pytest.mark.parametrize("limit", [0, 201])
def test_public_inspection_rejects_invalid_pagination_without_http(reader, limit):
    query = SimpleNamespace(task_id="task-answer-0", limit=limit, after=0)
    with pytest.raises(TimeFValidationError, match="limit"):
        reader.inspect().task_detail(query)
