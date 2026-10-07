"""Attach Coswara symptom context to benchmark classification tasks."""

from pydantic import TypeAdapter

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import BooleanValue
from timenet_connectors.datasets.yang_ai_lab.hearts_core.tasks import classification_task
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.release import PROMPTS, SYMPTOM_NAMES, VOCABULARIES


_SYMPTOMS_ADAPTER = TypeAdapter(dict[str, BooleanValue])


def convert_case(case: Case, inputs: tuple[Record, ...]) -> ClassificationTask:
    """Build a Coswara task with symptoms on its input record or on the task itself.

    Returns:
        The classification task.
    """
    flags = []
    values = {}
    if "symptoms" in case.task:
        values["symptoms"] = ", ".join(
            f"{SYMPTOM_NAMES[key]}: {'yes' if value else 'no'}" for key, value in case.payload["symptoms"].items()
        )
        flags = _symptom_flags(case)
    task = classification_task(case, inputs, PROMPTS[case.directory].substitute(values), VOCABULARIES[case.directory])
    if inputs:
        task.input_annotations = tuple(inputs[0].add_annotations(flags))
    else:
        for flag in flags:
            task.annotate(flag)
    return task


def _symptom_flags(case: Case) -> list[Annotation]:
    """Build one boolean annotation per symptom, named as the harness spelt it out.

    Returns:
        The flags in source order.

    """
    symptoms = _SYMPTOMS_ADAPTER.validate_python(case.payload["symptoms"])
    return [Annotation(key=key, value=value, description=SYMPTOM_NAMES[key]) for key, value in symptoms.items()]
