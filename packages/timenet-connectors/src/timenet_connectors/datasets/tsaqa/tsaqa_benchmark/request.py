"""A task's prompt, with a placeholder for each series its question prints, and the published question rebuilt.

The authors print a data transformation or temporal relationship question's candidate series inside
the question, as JSON lists (paper Appendix C.2). Each of those series is a Record of the task's
inputs, so the prompt shows ``<series N>`` in its place: the task's N-th input, counted from 1. The
input series come first, so a question's first printed series is ``<series 2>``. The placeholder is the
connector's choice; the authors publish no prompt code. :func:`authors_question` puts the printed text
back and gives the published question exactly.
"""

import re

from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.questions import series_text
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.rows import Row


_PLACEHOLDER = re.compile(r"<series (\d+)>")


def placeholder(position: int) -> str:
    """Return the placeholder of a task's input, counted from 1, such as ``<series 2>``."""
    return f"<series {position}>"


def task_prompt(row: Row) -> str:
    """Return a row's question with a placeholder in place of each series it prints.

    Returns:
        The prompt. A question that prints no series is returned as published.
    """
    first = len(row.input_lengths) + 1
    return row.question.text([placeholder(first + index) for index in range(len(row.question.series_labels))])


def authors_question(task: Task) -> str:
    """Rebuild the published question of a TSAQA task, with each placeholder's series printed back.

    Args:
        task: A task of the dataset, with its input Records.

    Returns:
        The question as the release publishes it.

    Raises:
        TimeFValidationError: If the task has no prompt, or a placeholder names no input Record.
    """
    if task.prompt is None:
        raise TimeFValidationError("a TSAQA task has a prompt")

    def printed(match: re.Match[str]) -> str:
        position = int(match[1])
        if not 1 <= position <= len(task.inputs):
            raise TimeFValidationError(f"{match[0]} names no input of a task with {len(task.inputs)}")
        (signal,) = task.inputs[position - 1].signals
        return series_text(signal.to_numpy().tolist())

    return _PLACEHOLDER.sub(printed, task.prompt.read())
