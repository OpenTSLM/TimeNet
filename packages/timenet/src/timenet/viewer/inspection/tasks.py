"""Bounded inspection projections, independent of HTTP and UI frameworks."""

# Projection functions document the returned facts in their summary.
# ruff: noqa: DOC201
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from timenet.viewer.inspection.requests import TaskDetailQuery, TaskQuery


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader

_MAX_METADATA_TEXT = 16384
_MAX_TEXT = 512

_OBJECT_IDS_CTE = """object_ids AS (
    SELECT dataset_key AS object_key, 'Dataset' AS object_type, dataset_id AS object_id FROM datasets
    UNION ALL SELECT record_key, 'Record', record_id FROM records
    UNION ALL SELECT source_key, 'Source', source_id FROM sources
    UNION ALL SELECT signal_key, 'Signal', signal_id FROM signals
    UNION ALL SELECT task_key, 'Task', task_id FROM tasks
)"""
_DETAIL_LIMIT = 200


def tasks(reader: TimeFReader, body: TaskQuery) -> dict[str, object]:
    """Return a bounded, lightweight task page."""
    rows = (
        reader._control_reader()
        .connection.execute(
            """SELECT task_id, task_type, prompt, split
                   FROM tasks
                   WHERE task_id > ? AND lower(task_id) LIKE lower(?) ESCAPE '\\'
                     AND (? IS NULL OR coalesce(split, '__unassigned__') = ?)
                     AND (? IS NULL OR task_type = ?)
                   ORDER BY task_id LIMIT ?""",
            [
                body.after or "",
                f"%{_like_literal(body.query or '')}%",
                body.split,
                body.split,
                body.task_type,
                body.task_type,
                body.limit + 1,
            ],
        )
        .fetchall()
    )
    items = [
        {"task_id": task_id, "task_type": task_type, "prompt": _bounded_text(prompt), "split": split}
        for task_id, task_type, prompt, split in rows[: body.limit]
    ]
    return {"api_version": 1, "items": items, "next_cursor": items[-1]["task_id"] if len(rows) > body.limit else None}


def task_detail(reader: TimeFReader, body: TaskDetailQuery) -> dict[str, object] | None:
    """Return bounded task relationships for the selected task."""
    row = (
        reader._control_reader()
        .connection.execute(
            """SELECT task_id, task_type, prompt, rationale, split, input_modalities, metadata,
                          scope_type, scope_start, scope_end, has_inline_targets,
                          target_schema, unit, target_name, mode,
                          (SELECT list(signal_id ORDER BY signal_id) FROM signal_refs
                           WHERE signal_key IN (SELECT unnest(tasks.scope_signal_keys)))
                   FROM tasks WHERE task_id = ?""",
            [body.task_id],
        )
        .fetchone()
    )
    if row is None:
        return None
    inputs = (
        reader._control_reader()
        .connection.execute(
            """SELECT records.record_id FROM task_record_refs
               JOIN records USING (record_key) JOIN tasks USING (task_key)
               WHERE tasks.task_id = ? AND field = 'inputs' ORDER BY position LIMIT ? OFFSET ?""",
            [body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )
    targets = (
        reader._control_reader()
        .connection.execute(
            """WITH span_signal_ids AS (
                       SELECT task_key, position, list(signal_id ORDER BY ordinal) AS signal_ids
                       FROM (
                           SELECT task_key, position, unnest(signal_keys) AS signal_key,
                                  generate_subscripts(signal_keys, 1) AS ordinal
                           FROM task_targets
                           WHERE task_key = (SELECT task_key FROM tasks WHERE task_id = ?)
                             AND position >= ? AND position < ?
                       ) span_keys
                       JOIN signal_refs USING (signal_key)
                       GROUP BY task_key, position
                   )
                   SELECT target_kind, text_value, integer_value, float_value, boolean_value,
                          records.record_id, signals.signal_id, span_start, span_end, span_signal_ids.signal_ids
                   FROM task_targets
                   JOIN tasks USING (task_key)
                   LEFT JOIN records ON records.record_key = task_targets.record_key
                   LEFT JOIN signal_refs signals ON signals.signal_key = task_targets.signal_key
                   LEFT JOIN span_signal_ids
                     ON span_signal_ids.task_key = task_targets.task_key
                    AND span_signal_ids.position = task_targets.position
                   WHERE tasks.task_id = ?
                   ORDER BY task_targets.position LIMIT ? OFFSET ?""",
            [body.task_id, body.after, body.after + body.limit + 1, body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )
    annotation_refs = _task_annotations(reader, body)
    parents = (
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
    candidates = (
        reader._control_reader()
        .connection.execute(
            """SELECT record_id FROM task_record_refs JOIN records USING (record_key)
               JOIN tasks USING (task_key) WHERE task_id = ? AND field = 'candidate_records'
               ORDER BY position LIMIT ? OFFSET ?""",
            [body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )
    more = any(len(items) > body.limit for items in (inputs, targets, annotation_refs, parents, candidates))
    return {
        "api_version": 1,
        "task_id": row[0],
        "task_type": row[1],
        "prompt": _bounded_text(row[2]),
        "rationale": _bounded_text(row[3]),
        "split": row[4],
        "input_modalities": row[5],
        "metadata": json.loads(row[6]),
        "scope": {
            "kind": row[7],
            "start": None if row[8] is None else str(row[8]),
            "end": None if row[9] is None else str(row[9]),
            "signal_ids": row[15],
        },
        "has_inline_targets": row[10],
        "configuration": {"target_schema": row[11], "unit": row[12], "target_name": row[13], "mode": row[14]},
        "prompt_truncated": row[2] is not None and len(row[2]) > _MAX_TEXT,
        "rationale_truncated": row[3] is not None and len(row[3]) > _MAX_TEXT,
        "input_record_ids": [item[0] for item in inputs[: body.limit]],
        "targets": [_target_detail(item) for item in targets[: body.limit]],
        "more_targets": len(targets) > body.limit,
        "annotation_refs": [_annotation_detail(item) for item in annotation_refs[: body.limit]],
        "more_annotation_refs": len(annotation_refs) > body.limit,
        "parent_task_ids": [item[0] for item in parents[: body.limit]],
        "candidate_record_ids": [item[0] for item in candidates[: body.limit]],
        "next_after": body.after + body.limit if more else None,
    }


def _task_annotations(reader: TimeFReader, body: TaskDetailQuery) -> list[tuple]:
    """Resolve a bounded relationship page through imported occurrence identities."""
    refs = (
        reader._control_reader()
        .connection.execute(
            """SELECT field, position, annotation_refs.occurrence_id, records.record_id
           FROM task_annotation_refs JOIN tasks USING (task_key) JOIN annotation_refs USING (occurrence_key)
           LEFT JOIN imported_annotations USING (occurrence_key) LEFT JOIN records USING (record_key)
           WHERE task_id = ? ORDER BY field, position LIMIT ? OFFSET ?""",
            [body.task_id, body.limit + 1, body.after],
        )
        .fetchall()
    )
    result = []
    for field, position, occurrence_id, record_id in refs:
        candidates = [reader]
        node = reader
        if record_id is not None:
            while (
                row := node._control_reader()
                .connection.execute(
                    """SELECT parent_dataset_id FROM record_imports JOIN records USING (record_key)
                   WHERE record_id = ?""",
                    [record_id],
                )
                .fetchone()
            ):
                node = node._parents[row[0]]
                candidates.append(node)
        for node in candidates:
            row = (
                node._control_reader()
                .connection.execute(
                    f"""WITH {_OBJECT_IDS_CTE}
                   SELECT occurrence_id, annotation_occurrences.object_type, object_ids.object_id, name,
                          value_kind, text_value, integer_value, float_value, boolean_value, span_type, start_us, end_us
                   FROM annotation_occurrences JOIN annotation_contents USING (content_key)
                   JOIN object_ids USING (object_key, object_type) WHERE occurrence_id = ?""",  # noqa: S608
                    [occurrence_id],
                )
                .fetchone()
            )
            if row is not None:
                result.append((field, position, *row))
                break
    return result


def _like_literal(value: str) -> str:
    """Escape a user string for DuckDB's LIKE expression."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _bounded_text(value: str | None) -> str | None:
    """Prevent a task prompt from becoming an unexpectedly large API response."""
    if value is None:
        return None
    return value[:_MAX_TEXT]


def _annotation_value(values: tuple[object, ...]) -> str | None:
    """Return the one typed annotation payload as a safe display string."""
    for value in values[1:]:
        if value is not None:
            return str(value)
    return None


def _annotation_detail(row: tuple[object, ...]) -> dict[str, object]:
    """Serialize one task annotation reference without exposing storage keys."""
    return {
        "field": row[0],
        "occurrence_id": row[2],
        "object_type": row[3],
        "object_id": row[4],
        "name": row[5],
        "value": _annotation_value(row[6:11]),
        "span_type": row[11],
        "start_us": None if row[12] is None else str(row[12]),
        "end_us": None if row[13] is None else str(row[13]),
    }


def _target_detail(row: tuple[object, ...]) -> dict[str, object]:
    """Serialize one target as its public IDs, value, and optional span."""
    value = next((item for item in row[1:5] if item is not None), None)
    if isinstance(value, int | float) and not isinstance(value, bool):
        value = str(value)
    return {
        "kind": row[0],
        "value": value,
        "record_id": row[5],
        "signal_id": row[6],
        "start": None if row[7] is None else str(row[7]),
        "end": None if row[8] is None else str(row[8]),
        "signal_ids": row[9] or [],
    }
