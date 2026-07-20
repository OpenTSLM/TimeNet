import pytest

from timenet.types import (
    TASKS,
    CaptioningTask,
    ClassificationTask,
    ForecastingTask,
    LabelingTask,
    QATask,
    ReasoningTask,
    Task,
    TaskType,
)
from timenet.types.tasks import _build_task_registry


def test_task_types():
    assert ClassificationTask.task_type is TaskType.CLASSIFICATION
    assert LabelingTask.task_type is TaskType.LABELING
    assert CaptioningTask.task_type is TaskType.CAPTIONING
    assert QATask.task_type is TaskType.QUESTION_AND_ANSWER
    assert ForecastingTask.task_type is TaskType.FORECASTING
    assert ReasoningTask.task_type is TaskType.REASONING


def test_tasks_registry_is_derived_and_complete():
    # Derived by walking the task hierarchy — every concrete task is registered by its task_type.
    assert set(TASKS) == set(TaskType)
    assert all(cls.task_type is key for key, cls in TASKS.items())
    assert TASKS[TaskType.CLASSIFICATION] is ClassificationTask


def test_classification_payload():
    t = ClassificationTask(label="afib")
    assert t.label == "afib"
    assert t.label_schema is None


def test_labeling_payload():
    t = LabelingTask(label="walking", time_series_ids=("s1", "s2"), windows_s=((0.0, 5.0),))
    assert t.time_series_ids == ("s1", "s2")
    assert t.windows_s == ((0.0, 5.0),)


def test_qa_and_forecasting_payloads():
    assert QATask(question="q?", answer="a").question == "q?"
    f = ForecastingTask(context_sample_ids=("c1",), target_sample_id="t1")
    assert f.target_sample_id == "t1"


def test_auto_id_unique():
    assert ClassificationTask(label="a").id != ClassificationTask(label="a").id


def test_from_task_ids_property():
    base = ClassificationTask(label="a")
    derived = ReasoningTask(question="q", answer="a", from_tasks=(base,))
    assert derived.from_task_ids == (base.id,)


def test_task_is_mutable_for_post_construction_linking():
    # add_task() populates sample_ids after construction, so Task must be mutable.
    t = ClassificationTask(label="a")
    t.sample_ids = ("sample-0",)
    assert t.sample_ids == ("sample-0",)


def test_base_task_has_no_task_type():
    with pytest.raises(AttributeError):
        _ = Task.task_type


def test_registry_rejects_task_type_collision():
    # Two classes claiming the same task_type would otherwise silently drop one from the registry.
    with pytest.raises(ValueError, match="task_type"):
        _build_task_registry([ClassificationTask, ClassificationTask])
