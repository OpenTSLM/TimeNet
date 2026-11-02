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
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    """Another answer is as right as the published one, but only the published one scores: two of a
    puzzle's patches are equal, or another choice prints the same series or text as the right one."""


T = TypeVar("T", bound=Task)
_ISSUE_NAMES = [issue.value for issue in SourceIssue]


def without_source_issues(tasks: Iterable[T], issues: Collection[SourceIssue] = tuple(SourceIssue)) -> Iterator[T]:
    """Leave out the tasks whose row has any of ``issues``.

    ``issues`` is checked at the call. Each task is checked when the result reaches it: reading a task
    that has no ``source_issues`` list of issue names raises ``TimeFValidationError``.

    Args:
        tasks: The tasks to filter.
        issues: The issues that exclude a task. All of them by default.

    Returns:
        An iterator over each task whose row has none of ``issues``, in order.

    Raises:
        TimeFValidationError: If ``issues`` isn't a collection of issue names.
    """
    if isinstance(issues, str):
        raise TimeFValidationError(f"issues is a collection of issue names, got the text {issues!r}")
    if not isinstance(issues, Iterable):
        raise TimeFValidationError(f"issues is a collection of issue names, got {issues!r}")
    excluded: set[SourceIssue] = set()
    for issue in issues:
        # Check the type first: an entry such as a list can't be looked up in a set.
        if not isinstance(issue, str) or issue not in set(SourceIssue):
            raise TimeFValidationError(f"unknown source issue {issue!r}, expected one of {_ISSUE_NAMES}")
        excluded.add(SourceIssue(issue))
    return (task for task in tasks if excluded.isdisjoint(_issues_of(task)))


def _issues_of(task: Task) -> set[SourceIssue]:
    """Read a task's issues from its metadata.

    Returns:
        The issues the task's metadata names.

    Raises:
        TimeFValidationError: If the metadata has no such list, so the task's issues are unknown.
    """
    if TaskKey.SOURCE_ISSUES not in task.metadata:
        raise TimeFValidationError(
            f"task metadata has no {TaskKey.SOURCE_ISSUES.value!r} list, so its source issues are unknown"
        )
    found = task.metadata[TaskKey.SOURCE_ISSUES]
    message = f"task metadata {TaskKey.SOURCE_ISSUES.value!r} must list issue names, got {found!r}"
    if not isinstance(found, list):
        raise TimeFValidationError(message)
    try:
        return {SourceIssue(name) for name in found}
    # SourceIssue raises ValueError for a name that isn't an issue.
    except ValueError as exc:
        raise TimeFValidationError(message) from exc
