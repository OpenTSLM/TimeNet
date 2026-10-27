"""The RATS40K authors' evaluation prompts, and the request and answer they exchange with a model, from a task.

``prompts/<variant>.txt`` holds the authors' prompt as the ``str.format`` template their evaluation
scripts fill with ``our_source`` and ``our_observation``. A task's prompt is that template without the
domain clause and the ``Observation:`` line, which the task's inputs supply. The dataset's
``<variant>_request_template`` annotation holds the full template, so :func:`authors_request` gives the
authors' request back. The stored templates drop the trailing spaces that end 15 univariate and 2
multivariate lines of the authors' text; nothing else differs. :func:`authors_answer` gives the answer
the authors fine-tune on. TimeNet ships no parser or metric for the answer.
"""

from collections.abc import Iterable
import json
from pathlib import Path
import re

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, Task
from timenet_connectors.datasets.time_ra.rats40k.keys import CHANNEL_ORDER, AnnotationKey, request_template
from timenet_connectors.datasets.time_ra.rats40k.records import record_metadata
from timenet_connectors.datasets.time_ra.rats40k.release import Variant


SYSTEM_PROMPT = "You are a helpful assistant."
"""The authors' system message. Their API clients send it; their local Hugging Face models drop it."""

_DOMAIN_CLAUSE = re.compile(r" The oberservation is from (?:the )?domain of \{our_source\}\.")
_OBSERVATION_LINE = "Observation: {our_observation}\n"


def task_prompt(template: str) -> str:
    """Remove the domain clause and the ``Observation:`` line, which the task's inputs supply.

    Returns:
        The template's remaining text with its ``{{}}`` escapes resolved and no final newline.

    Raises:
        TimeFValidationError: If the template lacks either placeholder exactly once.
    """
    text, clauses = _DOMAIN_CLAUSE.subn("", template)
    if clauses != 1 or text.count(_OBSERVATION_LINE) != 1:
        raise TimeFValidationError("prompt template needs one domain clause and one Observation line")
    return text.replace(_OBSERVATION_LINE, "").format().rstrip("\n")


TEMPLATES = {
    variant: (Path(__file__).parent / "prompts" / f"{variant}.txt").read_text(encoding="utf-8") for variant in Variant
}
TASK_PROMPTS = {variant: task_prompt(template) for variant, template in TEMPLATES.items()}
CATEGORY_NAMES = {
    variant: tuple(re.findall(r"^\d+\. ([^:\n]+):", prompt, re.MULTILINE)) for variant, prompt in TASK_PROMPTS.items()
}
"""The category names each prompt defines ("12. Sudden Spike Anomaly: ..."), in the authors' code order."""


def observation_text(record: Record) -> str:
    """Format a Record's series as the authors' scripts print the published observation.

    A univariate series prints as Python's ``str`` of its float list. Multivariate channels print in the
    Source's ``channel_order`` as ``"name": [values]``, joined by ``"; "``. The release spells every value
    as a float, so this equals the published text.

    Returns:
        The observation text to fill into the request template.

    Raises:
        TimeFValidationError: If the Record is not a RATS40K Record.
    """
    variant = record_metadata(record).variant
    order = record.sources[0].metadata.get(CHANNEL_ORDER) if len(record.sources) == 1 else None
    if not isinstance(order, list) or not order:
        raise TimeFValidationError("the Record is not a RATS40K Record")
    names = [name for name in order if isinstance(name, str)]
    if len(names) != len(order):
        raise TimeFValidationError("the Record's channel_order must list channel names")
    values = {signal.name: signal.to_numpy().tolist() for signal in record.signals}
    channels = [(name, values[name]) for name in names]
    if variant is Variant.UNI:
        return str(channels[0][1])
    return "; ".join(f"{json.dumps(name, ensure_ascii=False)}: {series}" for name, series in channels)


def authors_request(task: Task, annotations: Iterable[Annotation]) -> str:
    """Rebuild the authors' user message for a RATS40K task.

    Args:
        task: A task of the dataset, with its one input Record.
        annotations: The dataset's annotations, which hold the request templates.

    Returns:
        The user message the authors' evaluation scripts send for this row.

    Raises:
        TimeFValidationError: If the task, its Record, or the dataset annotations are not RATS40K's.
    """
    if len(task.inputs) != 1:
        raise TimeFValidationError("a RATS40K task needs exactly one input Record")
    (record,) = task.inputs
    key = request_template(record_metadata(record).variant)
    templates = [a.value for a in annotations if a.key == key]
    domains = [a.value for a in task.input_annotations if a.key == AnnotationKey.SOURCE_DOMAIN]
    if len(templates) != 1 or len(domains) != 1:
        raise TimeFValidationError("the task has no RATS40K request template or source_domain")
    return templates[0].format(our_source=domains[0], our_observation=observation_text(record))


def authors_answer(task: Task) -> str:
    r"""Format a task's rationale and target as the answer the authors fine-tune on.

    This is ``train/train_sft_parser.py``'s target at Time-RA ``200b244``: the ``Thought`` in
    ``\boxed1{}`` and the category name in ``\boxed2{}``, the shape the evaluation prompt asks for.

    Returns:
        ``Thought: \boxed1{<rationale>}``, a newline, and ``Action: \boxed2{<target>}``.

    Raises:
        TimeFValidationError: If the task lacks a rationale or a single target name.
    """
    if task.rationale is None or task.targets is None or len(task.targets) != 1:
        raise TimeFValidationError("a RATS40K task needs a rationale and exactly one target")
    (target,) = task.targets
    return f"Thought: \\boxed1{{{task.rationale}}}\nAction: \\boxed2{{{target}}}"
