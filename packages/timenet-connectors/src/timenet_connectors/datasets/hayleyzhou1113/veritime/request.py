"""The VeriTime authors' system prompt, and the user message they send a model, rebuilt from a task.

The authors' ``VeriTime_sft.py``, ``VeriTime_grpo.py`` and ``VeriTime_evaluate.py`` build one request
per row the same way: :data:`SYSTEM_PROMPT`, then the question with its series marker replaced by the
series as text. A task's prompt keeps the marker and its Record holds the series, so
:func:`authors_request` gives the authors' user message back. A task's rationale is the published
``cot``, which is the answer they fine-tune on, so it needs no rebuilding. TimeNet ships no parser or
metric for the answer.
"""

from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.hayleyzhou1113.veritime.records import record_metadata
from timenet_connectors.datasets.hayleyzhou1113.veritime.rows import SERIES_MARKER
from timenet_connectors.datasets.hayleyzhou1113.veritime.series import SeriesKind


SYSTEM_PROMPT = (
    "You are given a time series reasoning problem.\n"
    "Think about the problem and provide your working out.\n"
    "Place it between <THINK> and </THINK>.\n"
    "Then, provide your solution between <ANSWER></ANSWER>"
)
"""The authors' system message, as their scripts format it."""


def series_text(values: list[float], kind: SeriesKind) -> str:
    """Format a series as the authors' scripts paste it into a question.

    A scenario series is ``", ".join(map(str, values))`` of the published floats. A knowledge series is
    pasted as the release's string, whose every value has four decimals, which ``f"{value:.4f}"``
    reproduces for every published row.

    Returns:
        The series as text.
    """
    if kind is SeriesKind.SCENARIO:
        return ", ".join(map(str, values))
    return ", ".join(f"{value:.4f}" for value in values)


def authors_request(task: Task) -> str:
    """Rebuild the authors' user message for a VeriTime task.

    Args:
        task: A task of the dataset, with its one input Record.

    Returns:
        The task's question, with the Record's series in place of its marker.

    Raises:
        TimeFValidationError: If the task or its Record is not VeriTime's.
    """
    if len(task.inputs) != 1 or task.prompt is None:
        raise TimeFValidationError("a VeriTime task needs exactly one input Record and a prompt")
    (record,) = task.inputs
    kind = record_metadata(record).kind
    (signal,) = record.signals
    question = task.prompt.read()
    if question.count(SERIES_MARKER) != 1:
        raise TimeFValidationError(f"a VeriTime prompt holds {SERIES_MARKER!r} exactly once")
    return question.replace(SERIES_MARKER, series_text(signal.to_numpy().tolist(), kind))
