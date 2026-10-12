from contextlib import ExitStack
from dataclasses import replace

import pytest

from timenet.composition import BuildContext
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.format.duckdb import connect_control
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion, LocalRegistry
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DatasetMetadata,
    DatasetRef,
    ForecastingTask,
    StepInterval,
    StepPoint,
    TimeInterval,
    TSCorrespondenceTask,
    TSEditingTask,
    TSGenerationTask,
    Version,
)
from timenet.writer import TimeFWriter


def _parent(dataset_id):
    original = make_dataset()
    parent = TimeFDataset(
        metadata=DatasetMetadata.model_validate({**original.metadata.model_dump(), "dataset_id": dataset_id})
    )
    for record in original.records:
        record.task_ids = ()
        parent.add_record(record=record)
    parent.add_tasks(tasks=original.tasks)
    vocabulary = Annotation(
        id="vocabulary",
        key="label-vocabulary",
        value=[dataset_id, "normal"],
        description="Labels",
        metadata={"description": "metadata", "nested": ["vocabulary"]},
        source="annotator",
        confidence=0.8,
        span=TimeInterval.seconds(0, 1),
        occurrence_id="authored-vocabulary-occurrence",
        occurrence_metadata={"note": "authored"},
    )
    parent.register_annotations((vocabulary,))
    parent.register_annotations(
        (
            replace(
                parent.records[0].annotations[0],
                id="registered-age",
                source="registered-author",
                confidence=0.2,
            ),
        )
    )
    parent.tasks_of(ClassificationTask)[0].target_schema = vocabulary.id
    note = parent.annotate(Annotation(id="note", key="dataset-note", value=dataset_id))
    signal = Signal(
        id="ordinal-signal",
        name="ordinal",
        spec=parent.records[0].signals[0].spec,
        time_axis=OrdinalAxis(axis_id="ordinal-axis"),
        data=[1, 2, 3],
    )
    ordinal = parent.add_record(
        record=Record(
            record_id="ordinal-record", sources=(Source(id="ordinal-source", name="ordinal", signals=(signal,)),)
        )
    )
    parent.add_tasks(
        tasks=(
            TSCorrespondenceTask(
                id="correspondence",
                inputs=(parent.records[0],),
                candidate_records=parent.records[1:3],
                targets=(parent.records[1],),
            ),
            TSEditingTask(id="editing", inputs=(parent.records[0],), targets=(parent.records[0].signals[0],)),
            TSGenerationTask(id="generation", prompt="Generate a signal", targets=(parent.records[0].signals[0],)),
            ForecastingTask(
                id="steps",
                inputs=(ordinal,),
                scope=StepInterval(time_series_id=signal.id, start=0, stop=2),
                targets=(StepPoint(time_series_id=signal.id, start=2),),
            ),
            AnswerTask(id="no-input", input_annotations=(note,), targets=("record-0",)),
            AnswerTask(
                id="annotation-answer",
                inputs=(parent.records[0],),
                target_annotations=(parent.records[0].signals[0].annotations[0],),
            ),
            ClassificationTask(
                id="external-schema",
                inputs=(parent.records[0],),
                targets=("normal",),
                target_schema="external-name",
            ),
        )
    )
    for task in parent.tasks:
        task.annotate(Annotation(id="task-tag", key="task-tag", value="record-0"))
        task.metadata["nested"] = [task.id]
    return parent


def _child(parents):
    return TimeFDataset(
        metadata=DatasetMetadata.model_validate(
            {
                **parents[0].metadata.model_dump(),
                "dataset_id": "test/aggregate",
                "parents": tuple(
                    DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
                    for parent in parents
                ),
            }
        )
    )


def _store(root, dataset):
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _assert_parent_references(restored, parent):
    records = {record.id: record for record in restored.records}
    tasks = {task.id: task for task in restored.tasks}
    prefix = f"{parent.metadata.dataset_id}@1.0.0::"
    record = records[prefix + "record-0"]
    assert all(identifier.startswith(prefix) for identifier in record.task_ids)
    classification = tasks[prefix + "task-cls-0"]
    assert isinstance(classification, ClassificationTask)
    assert classification.inputs[0] is record
    assert classification.target_schema == prefix + "vocabulary"
    external = tasks[prefix + "external-schema"]
    assert isinstance(external, ClassificationTask)
    assert external.target_schema == "external-name"
    assert tasks[prefix + "task-answer-0"].from_tasks == (classification,)
    assert tasks[prefix + "task-answer-0"].from_tasks[0] is classification
    assert tasks[prefix + "editing"].targets == (record.signals[0],)
    assert tasks[prefix + "generation"].targets == (record.signals[0],)
    assert tasks[prefix + "annotation-answer"].target_annotations[0] is record.signals[0].annotations[0]
    correspondence = tasks[prefix + "correspondence"]
    assert isinstance(correspondence, TSCorrespondenceTask)
    assert correspondence.targets == (records[prefix + "record-1"],)
    assert correspondence.candidate_records == (
        records[prefix + "record-1"],
        records[prefix + "record-2"],
    )
    steps = tasks[prefix + "steps"]
    assert steps.scope == StepInterval(time_series_id=prefix + "ordinal-signal", start=0, stop=2)
    assert steps.targets == (StepPoint(time_series_id=prefix + "ordinal-signal", start=2),)
    assert tasks[prefix + "task-cls-2"].scope == TimeInterval.seconds(
        0, 0.25, time_series_ids=(prefix + "ts-window-2",)
    )
    localization = tasks[prefix + "task-localize-0"]
    assert localization.targets is not None
    localized = localization.targets[1]
    assert isinstance(localized, TimeInterval)
    assert localized.time_series_ids == (prefix + "ts-shared",)
    no_input = tasks[prefix + "no-input"]
    assert no_input.targets == ("record-0",)
    assert no_input.input_annotations[0] == next(a for a in restored.annotations if a.id == prefix + "note")
    assert no_input.input_annotations[0].occurrence_id in {a.occurrence_id for a in restored.annotations}
    assert no_input.metadata["nested"] == ["no-input"]
    assert no_input.annotations[0].id == prefix + "task-tag"
    definition = next(a for a in restored.registered_annotations if a.id == prefix + "vocabulary")
    original = parent.registered_annotations[0]
    assert definition == replace(original, id=prefix + original.id)
    assert original.occurrence_id is not None
    assert definition.occurrence_id == prefix + original.occurrence_id
    assert definition.occurrence_metadata == original.occurrence_metadata
    assert definition.confidence == original.confidence
    assert next(a for a in restored.registered_annotations if a.key == "age").source == "registered-author"
    assert record.annotations[0].source is None
    assert record.signals[0].to_arrow().equals(parent.records[0].signals[0].to_arrow())


def test_bulk_import_remaps_all_task_references_and_vocabulary_content(tmp_path, monkeypatch):
    parents = [_parent("first/tasks"), _parent("second/tasks")]
    for parent in parents:
        _store(tmp_path, parent)
    child = _child(parents)

    def unexpected_values(*args, **kwargs):
        raise AssertionError("importing structure must not load values or offsets")

    with monkeypatch.context() as patch:
        patch.setattr(TimeFReader, "_load_signal", unexpected_values)
        patch.setattr(TimeFReader, "_load_offsets", unexpected_values)
        with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
            for view in context.parents.values():
                view.import_into(child, include_tasks=True)
            context.verify_imports(child)
            child.set_dependencies(context.dependency_lock())
            directory = _store(tmp_path, child)
    for parent in parents:
        assert parent.records[0].id == "record-0"
        assert parent.tasks[0].id == "task-cls-0"
        assert parent.tasks_of(ClassificationTask)[0].target_schema == "vocabulary"
    assert len(child.tasks) == sum(len(parent.tasks) for parent in parents)
    with LocalRegistry(tmp_path).open_reader("test/aggregate") as reader:
        with monkeypatch.context() as patch:
            patch.setattr(TimeFReader, "_load_signal", unexpected_values)
            restored = reader.read()
        records = {record.id: record for record in restored.records}
        tasks = {task.id: task for task in restored.tasks}
        definitions = {annotation.id: annotation for annotation in restored.registered_annotations}
        for parent in parents:
            _assert_parent_references(restored, parent)
        assert reader._manifest.counts.signal_chunks == 0
        assert reader.task_table(("first/tasks@1.0.0::record-0",)).num_rows > 0
        copy = _store(tmp_path / "copy", restored)
    with ExitStack() as stack:
        readers = {
            parent.metadata.dataset_id: stack.enter_context(
                LocalRegistry(tmp_path).open_reader(parent.metadata.dataset_id)
            )
            for parent in parents
        }
        reread = stack.enter_context(TimeFReader(DatasetVersion.open_local(copy), parents=readers)).read()
        assert {task.id for task in reread.tasks} == set(tasks)
        assert {record.id for record in reread.records} == set(records)
        assert {annotation.id for annotation in reread.registered_annotations} == set(definitions)
    with connect_control(directory / "control.duckdb", read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM annotation_occurrences WHERE occurrence_id LIKE '%vocabulary%'"
        ).fetchone() == (0,)


def test_bulk_import_defaults_to_no_tasks(tmp_path):
    parent = _parent("first/tasks")
    _store(tmp_path, parent)
    child = _child((parent,))
    with BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context:
        view = context.parent("first/tasks")
        assert view.registered_annotations[0].id == "vocabulary"
        assert view.annotations[0].id == "note"
        view.import_into(child)
    assert len(child.records) == len(parent.records)
    assert not child.tasks
    assert all(not record.task_ids for record in child.records)
    assert child.registered_annotations[0].id == "first/tasks@1.0.0::vocabulary"
    assert child.annotations[0].id == "first/tasks@1.0.0::note"
    assert child.annotations[0].occurrence_id == f"first/tasks@1.0.0::{parent.annotations[0].occurrence_id}"


def test_bulk_import_requires_the_exact_parent_release(tmp_path):
    parent = _parent("first/tasks")
    _store(tmp_path, parent)
    child = _child((parent,))
    wrong = TimeFDataset(
        metadata=child.metadata.model_copy(
            update={"parents": (DatasetRef(dataset_id="first/tasks", version=Version(2, 0, 0)),)}
        )
    )
    with (
        BuildContext.open(child.metadata, LocalRegistry(tmp_path)) as context,
        pytest.raises(TimeFValidationError, match="no parent release"),
    ):
        context.parent("first/tasks").import_into(wrong, include_tasks=True)
    assert not wrong.records
    assert not wrong.tasks
