import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.payloads.reader import StoredPayloadLoader
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnnotationContentRef, ClassificationTask, TimeInterval
from timenet.writer import TimeFWriter


def test_multiple_parents_copy_editable_metadata_and_remap_task_graphs(tmp_path, monkeypatch):
    parent = make_dataset()
    vocabulary = parent.annotate(Annotation(id="vocabulary", key="classes", value=["normal", "other"]))
    parent.add_task(
        task=ClassificationTask(
            id="vocabulary-task",
            inputs=(parent.records[0],),
            input_annotations=(vocabulary,),
            target_schema=AnnotationContentRef(id=vocabulary.id),
            targets=("normal",),
        )
    )
    path = parent.write(path=tmp_path)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/composed"}))

    def forbidden(self):
        pytest.fail("copying metadata read a payload body")

    with TimeFReader(DatasetVersion.open_local(path)) as reader, monkeypatch.context() as guard:
        guard.setattr(StoredPayloadLoader, "__call__", forbidden)
        first = child.copy_from(reader, namespace="first", include_tasks=True)
        second = child.copy_from(reader, namespace="second", include_tasks=True)
        assert first.records["record-0"].id != second.records["record-0"].id
        first.records["record-0"].metadata["reviewed"] = True
        assert "reviewed" not in second.records["record-0"].metadata
        assert "reviewed" not in next(reader.iter_records(["record-0"])).metadata
        assert first.signals["ts-shared"].values.reference == second.signals["ts-shared"].values.reference
        mapped_tasks = {task.id: task for task in child.iter_tasks()}
        for imported in (first, second):
            answer = mapped_tasks[imported.task_ref("task-answer-0").id]
            assert answer.from_tasks == (imported.task_ref("task-cls-0"),)
            assert answer.inputs[0] is imported.records["record-0"]
            scoped = mapped_tasks[imported.task_ref("task-cls-2").id]
            assert isinstance(scoped.scope, TimeInterval)
            assert scoped.scope.time_series_ids == (imported.signals["ts-window-2"].id,)
            vocab_task = mapped_tasks[imported.task_ref("vocabulary-task").id]
            assert isinstance(vocab_task, ClassificationTask)
            assert vocab_task.target_schema == AnnotationContentRef(id=imported.content_ids["vocabulary"])
            assert vocabulary.occurrence_id is not None
            assert vocab_task.input_annotations[0] is imported.annotations[vocabulary.occurrence_id]
        with TimeFWriter(tmp_path, child) as writer:
            writer.write()
    child_path = tmp_path / child.metadata.dataset_id / "1.0.0"
    with TimeFReader(DatasetVersion.open_local(child_path)) as reader:
        records = list(reader.iter_records())
        tasks = list(reader.iter_tasks())
        context = reader.annotations
    assert len(records) == 2 * len(parent.records)
    assert len(tasks) == 2 * len(parent.tasks)
    for record in records:
        assert record.origin is not None
        assert record.origin.dataset.dataset_id == parent.metadata.dataset_id
        for signal in record.signals:
            assert signal.time_axis.origin is not None
            assert signal.time_axis.origin.kind == "axis"
    for task in tasks:
        assert task.origin is not None
        assert task.origin.dataset.dataset_id == parent.metadata.dataset_id
    for annotation in context:
        assert annotation.content_origin is not None
        assert annotation.content_origin.id == "vocabulary"


def test_subset_import_strips_tasks_and_duplicate_namespace_is_atomic():
    parent = make_dataset()
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/subset"}))
    subset = child.copy_from(parent, namespace="subset", record_ids=["record-0"])
    assert tuple(subset.records) == ("record-0",)
    assert child.records[0].task_ids == ()
    assert list(child.iter_tasks()) == []
    with pytest.raises(TimeFValidationError, match="unique"):
        child.copy_from(parent, namespace="subset")
    assert len(child.records) == 1
    with pytest.raises(TimeFValidationError, match="whole parent"):
        child.copy_from(parent, namespace="tasks", record_ids=["record-0"], include_tasks=True)
