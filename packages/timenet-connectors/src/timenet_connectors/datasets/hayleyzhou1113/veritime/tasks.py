"""VeriTime's tasks, one per published row, each built as the TimeNet task its answer makes it.

Every task has the same shape apart from its target. The prompt is the authors' question, with its
series marker. The rationale is the published ``cot``, the chain of thought and answer the authors
fine-tune on. The process labels of steps 1, 2, 4 and 6, which the authors' RL rewards score, are task
annotations, never model input.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from timenet.dataset import Record
from timenet.types import Annotation, AnswerTask, ClassificationTask, ScalarPredictionTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
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


@dataclass(frozen=True)
class VeriTimeTask(DatasetTask):
    """One row's task, on the Record of the series it publishes."""

    row: Row
    record: Record

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split of the row's file."""
        return self.row.ref.part.timenet_split

    def _fields(self) -> dict[str, Any]:
        """Return the fields every VeriTime task shares."""
        row = self.row
        return {
            "inputs": (self.record,),
            "prompt": row.question,
            "rationale": row.cot,
            "split": self.split,
            "metadata": {"source_task": row.task.value, "part": row.ref.part.value, "row": row.ref.row},
        }

    def _annotate(self, task: _T) -> _T:
        """Attach the row's process labels and, on a scenario row, its ChatTS answer.

        A process label published as null has no annotation, since an annotation must say something.

        Returns:
            The same task.
        """
        for key, field, description in _STEP_LABELS:
            value = getattr(self.row, field)
            if value is not None:
                task.annotate(Annotation(key=key, value=value, description=description))
        if self.row.output is not None:
            task.annotate(
                Annotation(key=AnnotationKey.SOURCE_ANSWER, value=self.row.output, description=_SOURCE_ANSWER)
            )
        return task


@dataclass(frozen=True)
class AnomalyDetectionTask(VeriTimeTask):
    """A true/false question: does the series meet the question's anomaly criterion? Yes or No."""

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with Yes or No as target."""
        schema = class_vocabulary(self.row.task)
        return self._annotate(ClassificationTask(targets=(self.row.label,), target_schema=schema, **self._fields()))


@dataclass(frozen=True)
class ScenarioAttributionTask(VeriTimeTask):
    """A multiple-choice question: which of the options the question offers best explains the series?

    The options differ by question, so there is no shared vocabulary. The target is the label as
    published: an option's text, sometimes with other capitalization or its letter.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    def to_timenet_task(self) -> AnswerTask:
        """Return the answer task, with the chosen option's text as target."""
        return self._annotate(AnswerTask(targets=(self.row.label,), **self._fields()))


@dataclass(frozen=True)
class InferentialCalculationTask(VeriTimeTask):
    """A counting question: how many events of the question's kind does the series show?"""

    TIMENET_TYPE: ClassVar[type[Task]] = ScalarPredictionTask

    def to_timenet_task(self) -> ScalarPredictionTask:
        """Return the scalar task, with the label's number as target: a count, apart from one amplitude."""
        return self._annotate(ScalarPredictionTask(targets=(self.row.quantity,), **self._fields()))


@dataclass(frozen=True)
class KnowledgeClassificationTask(VeriTimeTask):
    """A classification question on a real-world signal: CTU, ECG, EMG or RCW, each with its classes."""

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the class name as target."""
        schema = class_vocabulary(self.row.task)
        return self._annotate(ClassificationTask(targets=(self.row.label,), target_schema=schema, **self._fields()))


_KINDS: dict[SourceTask, type[VeriTimeTask]] = {
    SourceTask.ANOMALY_DETECTION: AnomalyDetectionTask,
    SourceTask.SCENARIO_ATTRIBUTION: ScenarioAttributionTask,
    SourceTask.INFERENTIAL_CALCULATION: InferentialCalculationTask,
    SourceTask.CTU: KnowledgeClassificationTask,
    SourceTask.ECG: KnowledgeClassificationTask,
    SourceTask.EMG: KnowledgeClassificationTask,
    SourceTask.RCW: KnowledgeClassificationTask,
}


def task_of(row: Row, record: Record) -> VeriTimeTask:
    """Return a row's task, of the kind its source task makes it.

    Returns:
        The task, on the Record of the row's series.
    """
    return _KINDS[row.task](row, record)
