"""The text, label and series the authors' loader gives a model for a row.

The authors' configs (``configs/tasks/llms_and_ts/*.yaml`` in behavioral-data/TSandLanguage at
d4e748cb) load every row through ``MultimodalMCQDataset.__getitem__`` (``src/models/tasks/
multimodal.py``) with ``format_abc_mcq``. Its context is the config's ``context_prefix``, then the
row's ``context_columns`` joined by newlines, then one line per option, ``A) ...``. Its label is the
letter of the right option. A task's prompt is that context and its target that letter.
"""

from collections.abc import Sequence
import random

from timenet.types import Task
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import DescriptionRow, QuestionRow


DESCRIPTION_PREFIX = "What description best relates to this time series?"
"""The ``context_prefix`` of ``ts2desc_mcq.yaml``, whose ``context_columns`` is empty."""
QUESTION_PREFIX = "Answer the following question about the time series"
"""The ``context_prefix`` of ``qa_mcq.yaml`` and ``qa_counterfactual.yaml``, whose ``context_columns``
are ``description_tiny``, ``metadata`` and ``question``."""


def letter(index: int) -> str:
    """Return the authors' letter for an option position: ``A`` for 0.

    Returns:
        The letter.
    """
    return chr(ord("A") + index)


def _context(prefix: str, columns: Sequence[str], options: Sequence[str]) -> str:
    # The prefix and the first column meet without a separator, as in the authors' loader.
    text = prefix + "\n".join(columns)
    return text + "\n" + "\n".join(f"{letter(index)}) {option}" for index, option in enumerate(options))


def description_options(row: DescriptionRow) -> list[str]:
    """Shuffle a description row's options once, with a seed of the row's own.

    The authors' loader shuffles them with ``random.shuffle`` and no seed, so each load shows its
    own order. The published order puts the answer first, so a fixed prompt needs one draw: this one
    seeds ``random.Random`` with the row's file and line, so every build gives the row the same order.

    Returns:
        The options in the order the task's prompt shows them.
    """
    options = list(row.options)
    random.Random(f"{row.ref.file.value}:{row.ref.line}").shuffle(options)  # noqa: S311 - a shuffle, not cryptography
    return options


def description_item(row: DescriptionRow) -> tuple[str, str]:
    """Build the authors' context and label for a description row, with its seeded order.

    Returns:
        The context, and the letter of the row's description.
    """
    options = description_options(row)
    # The authors' loader finds the label as options.index(description).
    return _context(DESCRIPTION_PREFIX, (), options), letter(options.index(row.description))


def question_item(row: QuestionRow) -> tuple[str, str]:
    """Build the authors' context and label for a question row, in the published option order.

    Returns:
        The context, and the letter at the row's ``answer_index``.
    """
    # The loader prints each column with str(): the metadata as the dict's repr, in key order.
    columns = (row.description_tiny, str(row.metadata), row.question)
    return _context(QUESTION_PREFIX, columns, row.options), letter(row.answer_index)


def authors_series(task: Task) -> list[int | float]:
    """Return the series the authors' loader gives a model with the prompt.

    A two-series question has two input Records, the series and its changed version. The loader
    passes them as one list, ``series + new_series``, with nothing to mark where one ends.

    Returns:
        The values of the task's input Records, in order, as one list.
    """
    return [value for record in task.inputs for signal in record.signals for value in signal.to_numpy().tolist()]
