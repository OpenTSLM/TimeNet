"""The source issues a row can have, kept as published and named in its task's metadata.

The connector keeps every row. A task lists its row's issues in its metadata under
``TaskKey.SOURCE_ISSUES``, an empty list when it has none, so :func:`without_source_issues` can leave
them out of a score. The connector itself never calls it. The README describes each issue and counts it.
"""

from collections.abc import Collection, Iterable, Iterator
from enum import StrEnum, unique
from typing import TypeVar

from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.keys import TaskKey


@unique
class SourceIssue(StrEnum):
    """A problem in a published row, which the connector keeps as published."""

    QUESTION_TYPE_MISMATCH = "question_type_mismatch"
    """The published ``question_type`` names another format than the question has: an ``ordering`` row
    that asks a true-or-false or multiple-choice question."""
    REPEATED_QUESTION = "repeated_question"
    """Another row asks the same question about the same series, and expects the same answer. Its
    context may differ: some UCR series are published under two dataset names."""
    REPEATED_ACROSS_SPLITS = "repeated_across_splits"
    """A copy of the row's question and series is in another split, so a model can train on a test
    question. Every such row also has ``repeated_question`` or ``conflicting_answer``."""
    CONFLICTING_ANSWER = "conflicting_answer"
    """Another row asks the same question about the same series, and expects another answer."""


T = TypeVar("T", bound=Task)


def without_source_issues(tasks: Iterable[T], issues: Collection[SourceIssue] = tuple(SourceIssue)) -> Iterator[T]:
    """Leave out the tasks whose row has any of ``issues``.

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
    """Read a task's issues from its metadata.

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
