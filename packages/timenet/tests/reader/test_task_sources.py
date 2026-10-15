import pytest

from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, AnnotationContentRef, AnswerTask, ClassificationTask, TaskRef
from timenet.writer import TimeFWriter


def test_task_sources_keep_local_tasks_and_typed_content_references(tmp_path):
    dataset = make_dataset()
    vocabulary = dataset.annotate(Annotation(id="vocabulary", key="classes", value=["normal", "abnormal"]))
    classified = ClassificationTask(
        id="streamed-classification",
        inputs=(dataset.records[0],),
        targets=("normal",),
        target_schema=AnnotationContentRef(id=vocabulary.content_id),
    )
    dataset.add_task_stream([ClassificationTask], lambda: iter((classified,)))
    answer = dataset.add_task(
        task=AnswerTask(
            id="local-derived", inputs=(dataset.records[0],), targets=("answer",), from_tasks=(classified.ref,)
        )
    )
    original_count = len(list(dataset.iter_tasks()))
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    path = tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        tasks = {task.id: task for task in reader.iter_tasks()}
    assert len(tasks) == original_count
    assert tasks[answer.id].from_tasks == (TaskRef(id=classified.id),)
    hydrated = tasks[classified.id]
    assert isinstance(hydrated, ClassificationTask)
    assert hydrated.target_schema == AnnotationContentRef(id="vocabulary")


def test_streamed_derivation_cycle_rejects_the_release(tmp_path):
    dataset = make_dataset()
    first = AnswerTask(id="first", inputs=(dataset.records[0],), targets=("a",), from_tasks=(TaskRef(id="second"),))
    second = AnswerTask(id="second", inputs=(dataset.records[0],), targets=("b",), from_tasks=(first.ref,))
    dataset.add_task_stream([AnswerTask], lambda: iter((first, second)))
    with pytest.raises(TimeFValidationError, match="cycle"), TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
