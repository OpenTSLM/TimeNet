"""RCAEval's task at its two levels: name the root cause service, and the root cause indicator.

The paper scores root cause analysis coarse-grained, against the root cause service, and fine-grained,
against the root cause indicator, a metric of that service (section 4.2). A method ranks candidates
and ``main.py`` scores the ranking with AC@k and Avg@k. Each level becomes one ClassificationTask per
case, whose target is the answer the ranking is scored against.

Every method ``main.py`` runs gets the case's telemetry, its inject time and its system, so the inject
time and the system are the task's input annotations. The benchmark asks no question in words, so a
task has no prompt. RCAEval is an evaluation benchmark and the paper splits nothing, so a task has no
split.
"""

from timenet.dataset import Record, Signal
from timenet.types import Annotation, ClassificationTask, TimeInterval
from timenet_connectors.datasets.phamquiluan.rcaeval.cases import Case
from timenet_connectors.datasets.phamquiluan.rcaeval.keys import AnnotationKey


_INPUT_KEYS = frozenset({AnnotationKey.INJECT_TIME, AnnotationKey.SYSTEM})
_ROOT_CAUSE_LOG = "The case's root_cause.txt: the root cause log line, as published. Never model input."


def service_task(case: Case, record: Record, scope: TimeInterval) -> ClassificationTask:
    """Return a case's coarse-grained task: name the root cause service.

    ``main.py`` scores a method's ranked metrics by their service, the part before the first ``_`` with a
    ``-db`` suffix removed, against the case's service.

    Args:
        case: The case.
        record: The case's Record.
        scope: The window ``main.py`` gives a method: up to 600 s on each side of the inject time.

    Returns:
        The classification task, whose target is the root cause service.
    """
    task = ClassificationTask(
        inputs=(record,),
        scope=scope,
        targets=(case.service,),
        input_annotations=_inputs(record),
        metadata={"case": case.path},
    )
    _annotate(task, case)
    return task


def indicator_task(
    case: Case, record: Record, scope: TimeInterval, indicator: Signal, authors_indicator: str
) -> ClassificationTask:
    """Return a case's fine-grained task: name the root cause indicator, the metric of the root cause service.

    Only cases whose fault has an indicator in ``main.py`` have one: none of the code-level faults do.

    Args:
        case: The case.
        record: The case's Record.
        scope: The window ``main.py`` gives a method: up to 600 s on each side of the inject time.
        indicator: The indicator's Signal in the Record's metrics.
        authors_indicator: The indicator as ``main.py`` names it, after its rename, such as
            ``checkoutservice_latency``.

    Returns:
        The classification task, whose target is the indicator's Signal.
    """
    task = ClassificationTask(
        inputs=(record,),
        scope=scope,
        targets=(indicator,),
        input_annotations=_inputs(record),
        metadata={"case": case.path, "authors_indicator": authors_indicator},
    )
    _annotate(task, case)
    return task


def _inputs(record: Record) -> tuple[Annotation, ...]:
    return tuple(annotation for annotation in record.annotations if annotation.key in _INPUT_KEYS)


def _annotate(task: ClassificationTask, case: Case) -> None:
    """Add the fault, by which the authors report scores, and the root cause log line, if published.

    Both are task annotations, never input annotations, so they are never model input.
    """
    task.annotate(Annotation(key=AnnotationKey.FAULT, value=case.fault.value, description=case.fault.description))
    if case.root_cause_log is not None:
        task.annotate(
            Annotation(key=AnnotationKey.ROOT_CAUSE_LOG, value=case.root_cause_log, description=_ROOT_CAUSE_LOG)
        )
