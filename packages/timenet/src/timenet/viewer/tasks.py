"""Task browsing routes using public reader iterators."""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from itertools import islice

from fastapi import FastAPI, Query

from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import BaseRegistry
from timenet.types import Task
from timenet.types.splits import Split
from timenet.viewer.errors import fail, registry_errors, resolve_version
from timenet.viewer.models import TaskDetail, TaskPage, TaskSummary, project_task


@contextmanager
def task_iterator(reader: TimeFReader, split: Split | None = None) -> Iterator[Iterator[Task]]:
    """Close a streaming generator before closing its reader, including on early exit.

    Yields:
        The public task iterator, with only the requested split filter.
    """
    iterator = reader.iter_tasks() if split is None else reader.iter_tasks(split=split)
    try:
        yield iterator
    finally:
        if isinstance(iterator, Generator):
            iterator.close()


def install_task_routes(app: FastAPI, registry: BaseRegistry, initial_manifest: Manifest | None) -> None:
    """Install typed task browsing endpoints within the authenticated app."""

    @app.get("/api/v1/tasks")
    def tasks(
        dataset_id: str,
        version: str,
        split: Split | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
    ) -> TaskPage:
        """Return a task page using one extra item to determine further availability."""
        # Response limits do not bound hydration: skipped tasks and their dependencies
        # are still read, and unfiltered iteration first hydrates every record.
        # TODO: Task summary pages. Add a reusable public reader query that avoids task
        # and record hierarchy hydration, with stable pagination before text/type filters.
        # TODO: Measured resource limits. Measure deep pages, missing-ID lookups,
        # dependency-heavy tasks, and large records before choosing scan/hydration guards
        # or efficient queries. A stopped lookup needs a distinct limit response, not 404.
        version = resolve_version(dataset_id, version, initial_manifest, "tasks")
        with (
            registry_errors("tasks"),
            TimeFReader(registry.open_version(dataset_id, version)) as reader,
            task_iterator(reader, split) as iterator,
        ):
            page = list(islice(iterator, offset, offset + limit + 1))
            return TaskPage(
                items=[
                    TaskSummary(
                        id=task.id,
                        task_type=task.task_type,
                        split=task.split,
                        prompt_preview=None if task.prompt is None else task.prompt[:200],
                    )
                    for task in page[:limit]
                ],
                offset=offset,
                limit=limit,
                has_more=len(page) > limit,
            )

    @app.get("/api/v1/task")
    def task(dataset_id: str, version: str, task_id: str = Query(..., min_length=1)) -> TaskDetail:
        """Scan for an exact task ID independently of the selected list split.

        Returns:
            The selected task projection.

        Raises:
            HTTPException: If the scan exhausts without finding the task.
        """  # noqa: DOC502 - raised by fail
        # TODO: Task-ID lookup. Add public keyed task hydration so opening a task or
        # rejecting a missing ID does not require scanning from the start.
        version = resolve_version(dataset_id, version, initial_manifest, "task")
        with (
            registry_errors("task"),
            TimeFReader(registry.open_version(dataset_id, version)) as reader,
            task_iterator(reader) as iterator,
        ):
            found = next((item for item in iterator if item.id == task_id), None)
            if found is None:
                fail(404, "not_found", "Task not found.", "task")
            return project_task(found)
