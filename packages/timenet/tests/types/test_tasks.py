import pytest

from timenet.types import (
    TASKS,
    CaptioningTask,
    ClassificationTask,
    ForecastingTask,
    LabelingTask,
    QATask,
    ReasoningTask,
    TargetTask,
    Task,
    TaskType,
)


def test_task_types():
    assert ClassificationTask.task_type is TaskType.CLASSIFICATION
    assert LabelingTask.task_type is TaskType.LABELING
    assert CaptioningTask.task_type is TaskType.CAPTIONING
    assert QATask.task_type is TaskType.QUESTION_AND_ANSWER
    assert ForecastingTask.task_type is TaskType.FORECASTING
    assert ReasoningTask.task_type is TaskType.REASONING


def test_tasks_registry_is_derived_and_complete():
    # Derived by walking the task hierarchy — every concrete task is registered by its task_type, no
    # drift, and the intermediate TargetTask base is skipped.
    assert set(TASKS) == set(TaskType)
    assert all(cls.task_type is key for key, cls in TASKS.items())
    assert TASKS[TaskType.CLASSIFICATION] is ClassificationTask
    assert TargetTask not in TASKS.values()


def test_target_bearing_tasks_share_target_base():
    for cls in (ClassificationTask, LabelingTask, CaptioningTask, QATask, ReasoningTask):
        assert issubclass(cls, TargetTask)
    assert not issubclass(ForecastingTask, TargetTask)  # forecasting has no scalar target


def test_classification_payload():
    t = ClassificationTask(target="afib")
    assert t.target == "afib"
    assert t.target_schema is None


def test_labeling_payload():
    t = LabelingTask(target="walking", time_series_ids=("s1", "s2"), windows_s=((0.0, 5.0),))
    assert t.time_series_ids == ("s1", "s2")
    assert t.windows_s == ((0.0, 5.0),)


def test_qa_and_forecasting_payloads():
    assert QATask(question="q?", target="a").question == "q?"
    f = ForecastingTask(context_sample_ids=("c1",), target_sample_id="t1")
    assert f.target_sample_id == "t1"


def test_auto_id_unique():
    assert ClassificationTask(target="a").id != ClassificationTask(target="a").id


def test_from_task_ids_property():
    base = ClassificationTask(target="a")
    derived = ReasoningTask(question="q", target="a", from_tasks=(base,))
    assert derived.from_task_ids == (base.id,)


def test_task_is_mutable_for_post_construction_linking():
    # add_task() populates sample_ids after construction, so Task must be mutable.
    t = ClassificationTask(target="a")
    t.sample_ids = ("sample-0",)
    assert t.sample_ids == ("sample-0",)


def test_base_task_has_no_task_type():
    with pytest.raises(AttributeError):
        _ = Task.task_type
