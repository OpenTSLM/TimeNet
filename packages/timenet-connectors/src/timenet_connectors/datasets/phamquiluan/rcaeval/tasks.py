"""RCAEval's task at its two levels: name the root cause service, and the root cause indicator.

The paper scores root cause analysis coarse-grained, against the root cause service, and fine-grained,
against the root cause indicator, a metric of that service (section 4.2). A method ranks candidates
and ``main.py`` scores the ranking with AC@k and Avg@k. Each level becomes one ClassificationTask per
case, whose target is the answer the ranking is scored against.
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record, Signal
from timenet.types import Annotation, ClassificationTask, Split, Task, TimeInterval
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.phamquiluan.rcaeval.cases import Case
from timenet_connectors.datasets.phamquiluan.rcaeval.keys import AnnotationKey


_INPUT_KEYS = frozenset({AnnotationKey.INJECT_TIME, AnnotationKey.SYSTEM})
_ROOT_CAUSE_LOG = "The case's root_cause.txt: the root cause log line, as published. Never model input."


@dataclass(frozen=True)
class RCAEvalTask(DatasetTask):
    """One case's task: its Record, seen through the window ``main.py`` gives a method.

    Every method ``main.py`` runs gets the case's telemetry, its inject time and its system, so the inject
    time and the system are the task's input annotations. The benchmark asks no question in words, so the
    task has no prompt. The paper defines no split.
    """

    case: Case
    record: Record
    scope: TimeInterval
    """The window ``main.py`` gives a method: up to 600 s on each side of the inject time."""

    @property
    def split(self) -> Split | None:
        """Return no split: RCAEval is an evaluation benchmark and the paper splits nothing."""
        return None

    def _inputs(self) -> tuple[Annotation, ...]:
        return tuple(annotation for annotation in self.record.annotations if annotation.key in _INPUT_KEYS)

    def _annotate(self, task: Task) -> None:
        """Add the fault, by which the authors report scores, and the root cause log line, if published.

        Both are task annotations, never input annotations, so they are never model input.
        """
        task.annotate(
            Annotation(key=AnnotationKey.FAULT, value=self.case.fault.value, description=self.case.fault.description)
        )
        if self.case.root_cause_log is not None:
            task.annotate(
                Annotation(
                    key=AnnotationKey.ROOT_CAUSE_LOG, value=self.case.root_cause_log, description=_ROOT_CAUSE_LOG
                )
            )


@dataclass(frozen=True)
class RCAEvalServiceTask(RCAEvalTask):
    """A case's coarse-grained task: name the root cause service.

    ``main.py`` scores a method's ranked metrics by their service, the part before the first ``_`` with a
    ``-db`` suffix removed, against the case's service.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, whose target is the root cause service."""
        task = ClassificationTask(
            inputs=(self.record,),
            scope=self.scope,
            targets=(self.case.service,),
            input_annotations=self._inputs(),
            metadata={"case": self.case.path},
        )
        self._annotate(task)
        return task


@dataclass(frozen=True)
class RCAEvalIndicatorTask(RCAEvalTask):
    """A case's fine-grained task: name the root cause indicator, the metric of the root cause service.

    Only cases whose fault has an indicator in ``main.py`` have one: none of the code-level faults do.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    indicator: Signal
    """The indicator's Signal in the Record's metrics."""
    authors_indicator: str
    """The indicator as ``main.py`` names it, after its rename, such as ``checkoutservice_latency``."""

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, whose target is the indicator's Signal."""
        task = ClassificationTask(
            inputs=(self.record,),
            scope=self.scope,
            targets=(self.indicator,),
            input_annotations=self._inputs(),
            metadata={"case": self.case.path, "authors_indicator": self.authors_indicator},
        )
        self._annotate(task)
        return task
