"""The source issues a question row can have, kept as published and named in its task's metadata.

The connector keeps every row. A question task lists its row's issues in its metadata under
``TaskKey.SOURCE_ISSUES``, an empty list when it has none, so :func:`without_source_issues` can leave
them out of a score. The connector itself never calls it. The README describes each issue.
"""

from collections.abc import Collection, Iterable, Iterator
from enum import StrEnum, unique
from typing import TypeVar

from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import TaskKey


@unique
class SourceIssue(StrEnum):
    """A problem in a published question row, which the connector keeps as published."""

    BLANK_QUESTION = "blank_question"
    BLANK_OPTION = "blank_option"
    REPEATED_OPTION = "repeated_option"
    """Two options have the same text."""
    ANSWER_TEXT_REPEATED = "answer_text_repeated"
    """The answer's text is also a wrong option's, so the text alone doesn't tell which one the authors
    count as right; ``answer_index`` does. Every such row also has ``repeated_option``."""


T = TypeVar("T", bound=Task)


def without_source_issues(tasks: Iterable[T], issues: Collection[SourceIssue] = tuple(SourceIssue)) -> Iterator[T]:
    """Leave out the tasks whose row has any of ``issues``.

    A task without the metadata key, such as a description task, has none and is kept.

    Args:
        tasks: The tasks to filter.
        issues: The issues that exclude a task. All of them by default.

    Yields:
        Each task whose row has none of ``issues``, in order.

    Raises:
        TimeFValidationError: If a task's ``source_issues`` isn't a list of issue names.
    """  # noqa: DOC502 - raised by _issues_of
    excluded = {SourceIssue(issue) for issue in issues}
    for task in tasks:
        if excluded.isdisjoint(_issues_of(task)):
            yield task


def _issues_of(task: Task) -> set[SourceIssue]:
    """Read a task's issues from its metadata, none when it has no such key.

    Returns:
        The issues the task's metadata names.

    Raises:
        TimeFValidationError: If the metadata value isn't a list of issue names.
    """
    found = task.metadata.get(TaskKey.SOURCE_ISSUES, [])
    message = f"task metadata {TaskKey.SOURCE_ISSUES.value!r} must list issue names, got {found!r}"
    if not isinstance(found, list):
        raise TimeFValidationError(message)
    try:
        return {SourceIssue(name) for name in found}
    # SourceIssue raises ValueError for a name that isn't an issue.
    except ValueError as exc:
        raise TimeFValidationError(message) from exc
