"""VeriTime's tasks, one per published row, each built as the TimeNet task its answer makes it.

Every task has the same shape apart from its target. The prompt is the authors' question, with its
series marker. The rationale is the published ``cot``, the chain of thought and answer the authors
fine-tune on. The process labels of steps 1, 2, 4 and 6, which the authors' RL rewards score, are task
annotations, never model input.
"""

from typing import Any, TypeVar

from timenet.dataset import Record
from timenet.types import Annotation, AnswerTask, ClassificationTask, ScalarPredictionTask, Task
from timenet_connectors.datasets.hayleyzhou1113.veritime.keys import AnnotationKey, class_vocabulary
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import SourceTask
from timenet_connectors.datasets.hayleyzhou1113.veritime.rows import Row


_STEP_LABELS = (
    (AnnotationKey.STEP1_LABEL, "step1_label", "The process label of step 1, the task intent, as published."),
    (
        AnnotationKey.STEP2_LABEL,
        "step2_label",
        "The process label of step 2, the key patterns separated by semicolons, as published.",
    ),
    (AnnotationKey.STEP4_LABEL, "step4_label", "The process label of step 4, the preliminary answer, as published."),
    (AnnotationKey.STEP6_LABEL, "step6_label", "The process label of step 6, the final answer, as published."),
)
_SOURCE_ANSWER = "The ChatTS answer the authors extracted the label from, as published."

_T = TypeVar("_T", bound=Task)


def _fields(row: Row, record: Record) -> dict[str, Any]:
    """Return the fields every VeriTime task shares, on the Record of the row's series.

    The split is the authors' split of the row's file.
    """
    return {
        "inputs": (record,),
        "prompt": row.question,
        "rationale": row.cot,
        "split": row.ref.part.timenet_split,
        "metadata": {"source_task": row.task.value, "part": row.ref.part.value, "row": row.ref.row},
    }


def _annotate(task: _T, row: Row) -> _T:
    """Attach the row's process labels and, on a scenario row, its ChatTS answer.

    A process label published as null has no annotation, since an annotation must say something.

    Returns:
        The same task.
    """
    for key, field, description in _STEP_LABELS:
        value = getattr(row, field)
        if value is not None:
            task.annotate(Annotation(key=key, value=value, description=description))
    if row.output is not None:
        task.annotate(Annotation(key=AnnotationKey.SOURCE_ANSWER, value=row.output, description=_SOURCE_ANSWER))
    return task


def anomaly_detection_task(row: Row, record: Record) -> ClassificationTask:
    """Return a true/false question's task: does the series meet the question's anomaly criterion? Yes or No.

    Returns:
        The classification task, with Yes or No as target.
    """
    schema = class_vocabulary(row.task)
    return _annotate(ClassificationTask(targets=(row.label,), target_schema=schema, **_fields(row, record)), row)


def scenario_attribution_task(row: Row, record: Record) -> AnswerTask:
    """Return a multiple-choice question's task: which of the options it offers best explains the series?

    The options differ by question, so there is no shared vocabulary. The target is the label as
    published: an option's text, sometimes with other capitalization or its letter.

    Returns:
        The answer task, with the chosen option's text as target.
    """
    return _annotate(AnswerTask(targets=(row.label,), **_fields(row, record)), row)


def inferential_calculation_task(row: Row, record: Record) -> ScalarPredictionTask:
    """Return a counting question's task: how many events of the question's kind does the series show?

    Returns:
        The scalar task, with the label's number as target: a count, apart from one amplitude.
    """
    return _annotate(ScalarPredictionTask(targets=(row.quantity,), **_fields(row, record)), row)


def knowledge_task(row: Row, record: Record) -> ClassificationTask:
    """Return a classification question's task on a real-world signal: CTU, ECG, EMG or RCW, each with its classes.

    Returns:
        The classification task, with the class name as target.
    """
    schema = class_vocabulary(row.task)
    return _annotate(ClassificationTask(targets=(row.label,), target_schema=schema, **_fields(row, record)), row)


def row_task(row: Row, record: Record) -> AnswerTask | ClassificationTask | ScalarPredictionTask:
    """Return a row's task, built by the function of its source task.

    Returns:
        The task, on the Record of the row's series.
    """
    match row.task:
        case SourceTask.ANOMALY_DETECTION:
            return anomaly_detection_task(row, record)
        case SourceTask.SCENARIO_ATTRIBUTION:
            return scenario_attribution_task(row, record)
        case SourceTask.INFERENTIAL_CALCULATION:
            return inferential_calculation_task(row, record)
        case SourceTask.CTU | SourceTask.ECG | SourceTask.EMG | SourceTask.RCW:
            return knowledge_task(row, record)
