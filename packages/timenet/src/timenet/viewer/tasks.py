"""Task browsing routes using projected summary pages and keyed reader lookup."""

from fastapi import FastAPI, Query

from timenet.errors import TimeFNotFoundError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import BaseRegistry
from timenet.types.splits import Split
from timenet.types.tasks import Task
from timenet.viewer.errors import fail, registry_errors, resolve_version
from timenet.viewer.models import TaskDetail, TaskPage, TaskSummary, project_task, public_id


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
        version = resolve_version(dataset_id, version, initial_manifest, "tasks")
        with registry_errors("tasks"), TimeFReader(registry.open_version(dataset_id, version)) as reader:
            page = reader.read_task_table(
                split=split,
                columns=("task_id", "task_type", "split", "prompt"),
                offset=offset,
                limit=limit + 1,
            ).to_pylist()
            return TaskPage(
                items=[
                    TaskSummary(
                        id=public_id(task["task_id"]),
                        task_type=task["task_type"],
                        split=task["split"],
                        prompt_preview=None if task["prompt"] is None else task["prompt"][:200],
                    )
                    for task in page[:limit]
                ],
                offset=offset,
                limit=limit,
                has_more=len(page) > limit,
            )

    @app.get("/api/v1/task")
    def task(dataset_id: str, version: str, task_id: int = Query(..., ge=1)) -> TaskDetail:
        """Return one task by ID, independently of the selected list split.

        Returns:
            The selected task projection.

        Raises:
            HTTPException: If the task ID does not exist.
        """  # noqa: DOC502 - raised by fail
        version = resolve_version(dataset_id, version, initial_manifest, "task")
        with registry_errors("task"), TimeFReader(registry.open_version(dataset_id, version)) as reader:
            try:
                item = reader.get(Task, task_id)
            except TimeFNotFoundError:
                fail(404, "not_found", "Task not found.", "task")
            return project_task(item)
