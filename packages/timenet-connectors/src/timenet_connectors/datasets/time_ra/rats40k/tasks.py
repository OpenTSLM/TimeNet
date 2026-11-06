"""Build tasks with the published prompts, labels and rationales."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

from timenet.dataset import Record
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Task, read_value
from timenet_connectors.datasets.time_ra.rats40k.source import Row
from timenet_connectors.datasets.time_ra.rats40k.vocabulary import (
    ACTION_ID_DESCRIPTION,
    ANSWER_FORMAT,
    CATEGORY_PATTERN,
    DOMAIN_CLAUSE_PATTERN,
    LABEL_DESCRIPTION,
    OBSERVATION_LINE,
    OUTSIDE_PROMPT_DESCRIPTION,
    PROMPT_DIRECTORY,
    AnnotationKey,
    MetadataKey,
    Variant,
)


@dataclass(frozen=True)
class _Prompts:
    """One variant's prompts, read from ``prompts/<variant>.txt``."""

    template: str
    """The authors' full evaluation prompt, a ``str.format`` template with ``{our_source}`` and
    ``{our_observation}``."""
    task_prompt: str
    """The template without the domain clause and the ``Observation:`` line, which a task's input
    annotation and Record supply."""
    categories: tuple[str, ...]
    """The category names the prompt defines, in the authors' code order."""


@cache
def load_prompts(variant: Variant) -> _Prompts:
    """Read a variant's prompt file and derive the task prompt and the category names from it.

    Returns:
        The variant's prompts.

    Raises:
        TimeNetConnectorError: If the prompt defines no category, or does not end with the empty answer
            fields of :data:`ANSWER_FORMAT`.
    """
    template = (PROMPT_DIRECTORY / f"{variant.value}.txt").read_text(encoding="utf-8")
    # ``format`` resolves the template's ``{{}}`` escapes, and fails if a placeholder is left.
    task_prompt = DOMAIN_CLAUSE_PATTERN.sub("", template).replace(OBSERVATION_LINE, "").format().rstrip("\n")
    categories = tuple(CATEGORY_PATTERN.findall(task_prompt))
    if not categories:
        raise TimeNetConnectorError(f"{variant.value} prompt defines no category")
    # authors_answer fills these fields, so the answer has the format that the prompt asks for.
    if not task_prompt.endswith(ANSWER_FORMAT.format(thought="", action="")):
        raise TimeNetConnectorError(f"{variant.value} prompt does not end with the answer fields Thought and Action")
    return _Prompts(template, task_prompt, categories)


def build_label_set(variant: Variant, targets: Iterable[str]) -> LabelSet:
    """Build a variant's label set: its prompt categories, then the other names its targets use.

    The prompt's categories come first, in the authors' code order, so a category's index in the set
    is its code. Some published targets are not prompt categories, such as the multivariate names of
    univariate test rows. They follow in sorted order, so that every row keeps its task.

    Args:
        variant: The variant whose prompt lists the categories.
        targets: The published target names of the variant's rows.

    Returns:
        The label set.
    """
    categories = load_prompts(variant).categories
    others = sorted(set(targets) - set(categories))
    return LabelSet(
        name=variant.label_set_name,
        labels=(
            *(Label(name=name) for name in categories),
            *(Label(name=name, description=OUTSIDE_PROMPT_DESCRIPTION) for name in others),
        ),
    )


def build_task(row: Row, record: Record, label_set: LabelSet) -> ClassificationTask:
    """Build a row's task: name the series' anomaly category, as the authors' prompt asks.

    The authors ask one question per row and score detection, category and reasoning from one answer. So
    one task carries all three: the name as target, the ``Thought`` as rationale, and ``Label`` and
    ``ActionID`` as task annotations, which are never model input.

    Args:
        row: The published row.
        record: The row's Record.
        label_set: The label set of the row's variant.

    Returns:
        The classification task.
    """
    metadata: dict[str, JsonValue] = {
        # False for the univariate test rows labelled with multivariate names.
        MetadataKey.TARGET_IN_VOCABULARY: row.action in load_prompts(row.observation.variant).categories,
        # In univariate test, a .pdf path marks the rows where gpt-4o ranks first, so it stays off the
        # input Record.
        MetadataKey.FIGURE_PATH: row.figure_path,
    }
    if row.rank is not None:
        metadata[MetadataKey.RANK] = list(row.rank)
    task = ClassificationTask(
        inputs=(record,),
        prompt=load_prompts(row.observation.variant).task_prompt,
        targets=(row.action,),
        rationale=row.thought,
        split=row.observation.split,
        input_annotations=tuple(a for a in record.annotations if a.key == AnnotationKey.SOURCE_DOMAIN),
        label_set=label_set,
        metadata=metadata,
    )
    # The authors score detection and category codes against these.
    task.annotate(Annotation(key=AnnotationKey.LABEL, value=row.label.value, description=LABEL_DESCRIPTION))
    task.annotate(Annotation(key=AnnotationKey.ACTION_ID, value=row.action_id, description=ACTION_ID_DESCRIPTION))
    return task


def authors_answer(task: Task) -> str:
    r"""Format a task's rationale and target as the answer that the prompt asks for.

    The prompt ends with the empty fields ``Thought: \boxed1{}`` and ``Action: \boxed2{}``. This fills
    them with the rationale and the category name. It is also the target that the authors fine-tune on
    (``train/train_sft_parser.py`` at Time-RA ``200b244``).

    Args:
        task: A RATS40K task, with its rationale and one target.

    Returns:
        ``Thought: \boxed1{<rationale>}``, a newline, and ``Action: \boxed2{<target>}``.

    Raises:
        TimeFValidationError: If the task has no rationale or not exactly one target.
    """
    if task.rationale is None or task.targets is None or len(task.targets) != 1:
        raise TimeFValidationError("a RATS40K task needs a rationale and exactly one target")
    (target,) = task.targets
    return ANSWER_FORMAT.format(thought=read_value(task.rationale), action=read_value(target))
