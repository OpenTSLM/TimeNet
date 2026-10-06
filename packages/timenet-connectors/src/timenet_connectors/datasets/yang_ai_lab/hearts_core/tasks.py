"""Construct task fields shared by the HEARTS children."""

from typing import Any

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, ClassificationTask, ScalarPredictionTask, Split
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import ANSWER_KEY


def task_fields(case: Case, inputs: tuple[Record, ...], prompt: str) -> dict[str, Any]:
    """Build a case's task identity, inputs, prompt, and metadata.

    Returns:
        Fields accepted by every TimeF task type.
    """
    return {
        "id": f"{case.id}-task",
        "prompt": prompt,
        "inputs": inputs,
        "split": Split.TEST,
        "metadata": {"corpus": case.corpus, "task": case.task, "testcase_idx": case.index},
    }


def classification_task(
    case: Case,
    inputs: tuple[Record, ...],
    prompt: str,
    vocabulary: Annotation,
    *,
    label: str | None = None,
) -> ClassificationTask:
    """Build a classification task and check its answer vocabulary.

    Returns:
        The classification task.

    Raises:
        TimeFFormatError: If the answer is not an allowed label.
    """
    label = str(case.payload[ANSWER_KEY]) if label is None else label
    if label not in case.definition.options:
        raise TimeFFormatError(
            f"HEARTS answer {case.payload[ANSWER_KEY]!r} is not one of {list(case.definition.options)}"
        )
    return ClassificationTask(targets=(label,), target_schema=vocabulary.id, **task_fields(case, inputs, prompt))


def scalar_task(
    case: Case, inputs: tuple[Record, ...], prompt: str, *, unit: str | None = None
) -> ScalarPredictionTask:
    """Flatten a scalar answer in the task definition's field order.

    Returns:
        The scalar prediction task.
    """
    answer = case.payload[ANSWER_KEY]
    parts = [answer[field] for field in case.definition.fields] if case.definition.fields else [answer]
    values = tuple(float(value) for part in parts for value in (part if isinstance(part, list) else [part]))
    return ScalarPredictionTask(
        targets=values,
        unit=unit,
        target_name=case.definition.target_name,
        **task_fields(case, inputs, prompt),
    )
