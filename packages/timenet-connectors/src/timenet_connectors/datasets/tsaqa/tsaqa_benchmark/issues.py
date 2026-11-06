"""Filter TSAQA tasks by the source issues of their rows, for the users of the dataset."""

from collections.abc import Collection, Iterable, Iterator
from typing import TypeVar

from timenet.errors import TimeFValidationError
from timenet.types import Task
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import SOURCE_ISSUES, SourceIssue


_TaskT = TypeVar("_TaskT", bound=Task)


def without_source_issues(
    tasks: Iterable[_TaskT], issues: Collection[SourceIssue] = tuple(SourceIssue)
) -> Iterator[_TaskT]:
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
            raise TimeFValidationError(
                f"unknown source issue {issue!r}, expected one of {[issue.value for issue in SourceIssue]}"
            )
        excluded.add(SourceIssue(issue))
    return (task for task in tasks if excluded.isdisjoint(_task_issues(task)))


def _task_issues(task: Task) -> set[SourceIssue]:
    if SOURCE_ISSUES not in task.metadata:
        raise TimeFValidationError(f"task metadata has no {SOURCE_ISSUES!r} list, so its source issues are unknown")
    found = task.metadata[SOURCE_ISSUES]
    message = f"task metadata {SOURCE_ISSUES!r} must list issue names, got {found!r}"
    if not isinstance(found, list):
        raise TimeFValidationError(message)
    try:
        return {SourceIssue(name) for name in found}
    except ValueError as exc:
        raise TimeFValidationError(message) from exc
