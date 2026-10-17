"""Bounded inspection projections, independent of HTTP and UI frameworks."""

# Projection functions document the returned facts in their summary.
# ruff: noqa: DOC201
# Only fixed projection SQL and integer budgets are interpolated; caller values are bound.
# ruff: noqa: S608
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

from pydantic import TypeAdapter, ValidationError

from timenet.errors import TimeFFormatError
from timenet.format.query_helpers import OBJECT_IDS_SQL, like_literal
from timenet.reader.inspection.limits import (
    MAX_METADATA,
    MAX_RELATIONSHIPS,
    MAX_TEXT,
    annotation_value_sql,
    bounded_text,
    validate_page,
)
from timenet.reader.inspection.requests import TaskDetailQuery, TaskQuery
from timenet.reader.inspection.results import AnnotationItem, TargetItem, TaskDetail, TaskItem, TaskPage, TaskScope


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader

_PROMPT_LIMIT = 512
_TARGET = TypeAdapter(TargetItem)
_SCOPE = TypeAdapter(TaskScope)


class TaskRow(NamedTuple):
    """Named columns for the selected task's bounded scalar facts."""

    task_id: str
    task_type: str
    prompt: str | None
    rationale: str | None
    split: str | None
    input_modalities: list[str]
    metadata: str
    scope_type: str | None
    scope_start: int | None
    scope_end: int | None
    has_inline_targets: bool
    target_schema: str | None
    unit: str | None
    target_name: str | None
    mode: str | None
    scope_signal_ids: list[str] | None


class AnnotationRow(NamedTuple):
    """One task annotation decoded at the SQL boundary."""

    field: str
    position: int
    occurrence_id: str
    object_type: Literal["Dataset", "Record", "Source", "Signal", "Task"]
    object_id: str
    name: str
    value_kind: str | None
    value: str | None
    span_type: Literal["static", "point", "interval"]
    start_us: int | None
    end_us: int | None


class TargetRow(NamedTuple):
    """One bounded task target decoded at the SQL boundary."""

    kind: str
    value: str | None
    record_id: str | None
    signal_id: str | None
    start: int | None
    end: int | None
    signal_ids: list[str] | None


_OBJECT_IDS_CTE = f"object_ids AS ({OBJECT_IDS_SQL})"


def tasks(reader: TimeFReader, body: TaskQuery) -> TaskPage:
    """Return a bounded, lightweight task page."""
    validate_page(body.limit)
    rows = (
        reader._control_reader()
        .connection.execute(
            f"""SELECT task_id, task_type, substring(prompt, 1, {_PROMPT_LIMIT + 1}), split
                   FROM tasks
                   WHERE task_id > ? AND lower(task_id) LIKE lower(?) ESCAPE '\\'
                     AND (? IS NULL OR coalesce(split, '__unassigned__') = ?)
                     AND (? IS NULL OR task_type = ?)
                   ORDER BY task_id LIMIT ?""",
            [
                body.after or "",
                f"%{like_literal(body.query or '')}%",
                body.split,
                body.split,
                body.task_type,
                body.task_type,
                body.limit + 1,
            ],
        )
        .fetchall()
    )
    items: list[TaskItem] = [
        {"task_id": task_id, "task_type": task_type, "prompt": bounded_text(prompt), "split": split}
        for task_id, task_type, prompt, split in rows[: body.limit]
    ]
    return {"api_version": 1, "items": items, "next_cursor": items[-1]["task_id"] if len(rows) > body.limit else None}


def task_detail(reader: TimeFReader, body: TaskDetailQuery) -> TaskDetail | None:
    """Return bounded task relationships for the selected task."""
    validate_page(body.limit, body.after)
    row = _task_row(reader, body)
    if row is None:
        return None
    task = TaskRow(*row)
    inputs = _record_refs(reader, body, "inputs")
    targets = _targets(reader, body)
    annotation_refs = _task_annotations(reader, body)
    parents = _parents(reader, body)
    candidates = _record_refs(reader, body, "candidate_records")
    more = any(
        len(items) > body.limit
        for items in (inputs, targets, annotation_refs, parents, candidates, task.scope_signal_ids or [])
    )
    return {
        "api_version": 1,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "prompt": task.prompt[:_PROMPT_LIMIT] if task.prompt is not None else None,
        "rationale": task.rationale[:_PROMPT_LIMIT] if task.rationale is not None else None,
        "split": task.split,
        "input_modalities": task.input_modalities,
        "metadata": task.metadata[:MAX_METADATA],
        "metadata_truncated": len(task.metadata) > MAX_METADATA,
        "scope": _scope_detail(task, body.limit),
        "has_inline_targets": task.has_inline_targets,
        "configuration": {
            "target_schema": task.target_schema,
            "unit": task.unit,
            "target_name": task.target_name,
            "mode": task.mode,
        },
        "prompt_truncated": task.prompt is not None and len(task.prompt) > _PROMPT_LIMIT,
        "rationale_truncated": task.rationale is not None and len(task.rationale) > _PROMPT_LIMIT,
        "input_record_ids": [item[0] for item in inputs[: body.limit]],
        "targets": [_target_detail(TargetRow(*item)) for item in targets[: body.limit]],
        "more_targets": len(targets) > body.limit,
        "annotation_refs": [_annotation_detail(AnnotationRow(*item)) for item in annotation_refs[: body.limit]],
        "more_annotation_refs": len(annotation_refs) > body.limit,
        "parent_task_ids": [item[0] for item in parents[: body.limit]],
        "candidate_record_ids": [item[0] for item in candidates[: body.limit]],
        "next_after": body.after + body.limit if more else None,
    }


def _task_annotations(reader: TimeFReader, body: TaskDetailQuery) -> list[tuple]:
    """Return a bounded page of annotations referenced by one task."""
    return (
        reader._control_reader()
        .connection.execute(
            f"""WITH {_OBJECT_IDS_CTE}
               SELECT field, position, occurrence_id, annotation_occurrences.object_type, object_ids.object_id,
                      name, value_kind, {annotation_value_sql()},
                      span_type, start_us, end_us
               FROM task_annotation_refs JOIN tasks USING (task_key)
               JOIN annotation_occurrences USING (occurrence_key)
               JOIN annotation_contents USING (content_key)
               JOIN object_ids USING (object_key, object_type)
               WHERE task_id = ? ORDER BY field, position LIMIT ? OFFSET ?""",
            [body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )


def _annotation_detail(annotation: AnnotationRow) -> AnnotationItem:
    """Serialize all annotation kinds through the shared bounded SQL projection."""
    return {
        "field": annotation.field,
        "occurrence_id": annotation.occurrence_id,
        "object_type": annotation.object_type,
        "object_id": annotation.object_id,
        "name": annotation.name,
        "value_kind": annotation.value_kind,
        "value": bounded_text(annotation.value),
        "value_truncated": annotation.value is not None and len(annotation.value) > MAX_TEXT,
        "span_type": annotation.span_type,
        "start_us": None if annotation.start_us is None else str(annotation.start_us),
        "end_us": None if annotation.end_us is None else str(annotation.end_us),
    }


def _target_detail(target: TargetRow) -> TargetItem:
    """Serialize a named target with explicit text and relationship truncation.

    Raises:
        TimeFFormatError: If required target facts are missing or invalid.
    """
    value: str | bool | None = bounded_text(target.value)
    if target.kind == "boolean" and target.value is not None:
        value = target.value == "true"
    try:
        return _TARGET.validate_python(
            {
                "kind": target.kind,
                "value": value,
                "value_truncated": target.value is not None and len(target.value) > MAX_TEXT,
                "record_id": target.record_id,
                "signal_id": target.signal_id,
                "start": None if target.start is None else str(target.start),
                "end": None if target.end is None else str(target.end),
                "signal_ids": (target.signal_ids or [])[:MAX_RELATIONSHIPS],
                "signal_ids_truncated": len(target.signal_ids or []) > MAX_RELATIONSHIPS,
            }
        )
    except ValidationError as exc:
        raise TimeFFormatError("invalid task target") from exc


def _scope_detail(task: TaskRow, limit: int) -> TaskScope:
    """Validate required temporal facts before returning a scope variant.

    Raises:
        TimeFFormatError: If the persisted scope is incomplete.
    """
    try:
        return _SCOPE.validate_python(
            {
                "kind": task.scope_type,
                "start": None if task.scope_start is None else str(task.scope_start),
                "end": None if task.scope_end is None else str(task.scope_end),
                "signal_ids": (task.scope_signal_ids or [])[:limit],
                "more_signal_ids": len(task.scope_signal_ids or []) > limit,
            }
        )
    except ValidationError as exc:
        raise TimeFFormatError("invalid task scope") from exc


def _task_row(reader: TimeFReader, body: TaskDetailQuery) -> tuple | None:
    """Read bounded scalar task facts and scope references."""
    return (
        reader._control_reader()
        .connection.execute(
            f"""SELECT task_id, task_type, substring(prompt, 1, {_PROMPT_LIMIT + 1}),
                          substring(rationale, 1, {_PROMPT_LIMIT + 1}), split, input_modalities,
                          substring(metadata, 1, {MAX_METADATA + 1}),
                          scope_type, scope_start, scope_end, has_inline_targets,
                          target_schema, unit, target_name, mode,
                          (SELECT list(signal_id ORDER BY signal_id) FROM
                              (SELECT signal_id FROM signals
                               WHERE signal_key IN (SELECT unnest(tasks.scope_signal_keys))
                               ORDER BY signal_id LIMIT ? OFFSET ?))
                   FROM tasks WHERE task_id = ?""",
            [body.limit + 1, body.after, body.task_id],
        )
        .fetchone()
    )


def _record_refs(
    reader: TimeFReader, body: TaskDetailQuery, field: Literal["inputs", "candidate_records"]
) -> list[tuple[str]]:
    """Read one ordered page of task record references."""
    return (
        reader._control_reader()
        .connection.execute(
            """SELECT records.record_id FROM task_record_refs
               JOIN records USING (record_key) JOIN tasks USING (task_key)
               WHERE tasks.task_id = ? AND field = ? ORDER BY position LIMIT ? OFFSET ?""",
            [body.task_id, field, body.limit + 1, body.after],
        )
        .fetchall()
    )


def _targets(reader: TimeFReader, body: TaskDetailQuery) -> list[tuple]:
    """Read a bounded page of targets and their signal references."""
    return (
        reader._control_reader()
        .connection.execute(
            f"""WITH span_signal_ids AS (
                       SELECT task_key, position, list(signal_id ORDER BY ordinal) AS signal_ids
                       FROM (
                           SELECT task_key, position, unnest(list_slice(signal_keys, 1, 201)) AS signal_key,
                                  generate_subscripts(list_slice(signal_keys, 1, 201), 1) AS ordinal
                           FROM task_targets
                           WHERE task_key = (SELECT task_key FROM tasks WHERE task_id = ?)
                             AND position >= ? AND position < ?
                       ) span_keys
                       JOIN signals USING (signal_key)
                       GROUP BY task_key, position
                   )
                   SELECT target_kind, substring(coalesce(text_value, cast(integer_value AS VARCHAR),
                          cast(float_value AS VARCHAR), cast(boolean_value AS VARCHAR)), 1, {MAX_TEXT + 1}),
                          records.record_id, signals.signal_id, span_start, span_end, span_signal_ids.signal_ids
                   FROM task_targets
                   JOIN tasks USING (task_key)
                   LEFT JOIN records ON records.record_key = task_targets.record_key
                   LEFT JOIN signals signals ON signals.signal_key = task_targets.signal_key
                   LEFT JOIN span_signal_ids
                     ON span_signal_ids.task_key = task_targets.task_key
                    AND span_signal_ids.position = task_targets.position
                   WHERE tasks.task_id = ?
                   ORDER BY task_targets.position LIMIT ? OFFSET ?""",
            [body.task_id, body.after, body.after + body.limit + 1, body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )


def _parents(reader: TimeFReader, body: TaskDetailQuery) -> list[tuple[str]]:
    """Read a bounded page of parent task identities."""
    return (
        reader._control_reader()
        .connection.execute(
            """SELECT parent.task_id FROM task_dependencies
                   JOIN tasks child USING (task_key)
                   JOIN tasks parent ON parent.task_key = task_dependencies.parent_task_key
                   WHERE child.task_id = ? ORDER BY task_dependencies.position LIMIT ? OFFSET ?""",
            [body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )
