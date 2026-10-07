"""Convert COUGHVID boolean, diagnosis, and MFCC answers."""

from pydantic import TypeAdapter

from timenet.dataset import Record
from timenet.types import ScalarPredictionTask, Task
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import ANSWER_KEY, BooleanValue
from timenet_connectors.datasets.yang_ai_lab.hearts_core.tasks import classification_task, scalar_task
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.release import PROMPTS, VOCABULARIES


_BOOLEAN_ADAPTER = TypeAdapter(BooleanValue)


def convert_case(case: Case, inputs: tuple[Record, ...]) -> Task:
    """Build a task with the release's boolean labels or ordered MFCC fields.

    Returns:
        The task, not yet registered.
    """
    prompt = PROMPTS[case.directory].substitute()
    if case.definition.task_type is ScalarPredictionTask:
        # The pinned release supplies MFCC coefficients without a unit convention.
        return scalar_task(case, inputs, prompt)
    answer = case.payload[ANSWER_KEY]
    if case.task == "diagnosis_classification":
        label = str(answer)
    else:
        label = "true" if _BOOLEAN_ADAPTER.validate_python(answer) else "false"
    return classification_task(case, inputs, prompt, VOCABULARIES[case.directory], label=label)
