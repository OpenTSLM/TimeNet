"""Qualified, paginated annotation overlays across imported record layers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from timenet.errors import TimeFValidationError
from timenet.viewer.inspection import ViewerInspection


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


_CURSOR_FIELDS = 3
_MAX_TEXT = 16384


def window_annotations(reader: TimeFReader, query: Any) -> dict[str, object]:
    """Return record, ancestor-source and signal occurrences visible in a window.

    Returns:
        A bounded page with qualified occurrence identities and an opaque continuation.

    Raises:
        TimeFValidationError: If the continuation is malformed.
    """
    after = ("", "", "")
    if query.after:
        try:
            decoded = json.loads(query.after)
            if (
                not isinstance(decoded, list)
                or len(decoded) != _CURSOR_FIELDS
                or not all(isinstance(x, str) for x in decoded)
            ):
                raise TimeFValidationError("invalid annotation cursor")
            after = tuple(decoded)
        except (TypeError, ValueError) as exc:
            raise TimeFValidationError("invalid annotation cursor") from exc
    ViewerInspection(reader).signal(query.record_id, query.signal_id)
    items = []
    node = reader
    while True:
        identity = (node.metadata.dataset_id, str(node.metadata.dataset_version))
        control = node._control_reader().connection
        if identity >= after[:2]:
            rows = control.execute(
                """WITH RECURSIVE ancestors AS (
                       SELECT sources.source_key, parent_source_key, source_id FROM signals
                       JOIN sources USING (source_key) JOIN records USING (record_key)
                       WHERE record_id = ? AND signal_id = ?
                       UNION ALL
                       SELECT sources.source_key, sources.parent_source_key, sources.source_id
                       FROM sources JOIN ancestors ON sources.source_key = ancestors.parent_source_key
                   ), owners AS (
                       SELECT 'Record' AS object_type, record_key AS object_key, record_id AS object_id
                       FROM records WHERE record_id = ?
                       UNION ALL SELECT 'Source', source_key, source_id FROM ancestors
                       UNION ALL SELECT 'Signal', signal_key, signal_id FROM signals
                       JOIN sources USING (source_key) JOIN records USING (record_key)
                       WHERE record_id = ? AND signal_id = ?
                   )
                   SELECT occurrence_id, name, span_type, start_us, end_us, owners.object_type, object_id,
                          value_kind, text_value, integer_value, float_value, boolean_value, text_list_value
                   FROM annotation_occurrences JOIN annotation_contents USING (content_key)
                   JOIN owners USING (object_key, object_type)
                   WHERE occurrence_id > ? AND (
                     span_type = 'static' OR
                     (span_type = 'point' AND (? IS NULL OR start_us >= ?)
                         AND (? IS NULL OR start_us < ?)) OR
                     (span_type = 'interval' AND (? IS NULL OR end_us > ?)
                         AND (? IS NULL OR start_us < ?)))
                   ORDER BY occurrence_id LIMIT ?""",
                [
                    query.record_id,
                    query.signal_id,
                    query.record_id,
                    query.record_id,
                    query.signal_id,
                    after[2] if identity == after[:2] else "",
                    query.start_us,
                    query.start_us,
                    query.end_us,
                    query.end_us,
                    query.start_us,
                    query.start_us,
                    query.end_us,
                    query.end_us,
                    query.limit + 1,
                ],
            ).fetchall()
            for row in rows:
                items.append(
                    {
                        "dataset_id": identity[0],
                        "version": identity[1],
                        "occurrence_id": row[0],
                        "name": row[1],
                        "span_type": row[2],
                        "start_us": None if row[3] is None else str(row[3]),
                        "end_us": None if row[4] is None else str(row[4]),
                        "object_type": row[5],
                        "object_id": row[6],
                        "value_kind": row[7],
                        "value": next((str(value) for value in row[8:] if value is not None), None),
                    }
                )
            items.sort(key=lambda item: (item["dataset_id"], item["version"], item["occurrence_id"]))
            items = items[: query.limit + 1]
        imported = control.execute(
            """SELECT parent_dataset_id FROM record_imports JOIN records USING (record_key)
               WHERE record_id = ?""",
            [query.record_id],
        ).fetchone()
        if imported is None:
            break
        node = node._parents[imported[0]]
    more = len(items) > query.limit
    page = items[: query.limit]
    cursor = json.dumps([page[-1][key] for key in ("dataset_id", "version", "occurrence_id")]) if more else None
    return {"api_version": 1, "items": page, "more": more, "next_cursor": cursor}


def _visible(root: TimeFReader, node: TimeFReader, record_id: str | None) -> bool:
    if root is node:
        return True
    if record_id is None:
        return False
    while root is not node:
        row = (
            root._control_reader()
            .connection.execute(
                """SELECT parent_dataset_id FROM record_imports JOIN records USING (record_key)
               WHERE record_id = ?""",
                [record_id],
            )
            .fetchone()
        )
        if row is None:
            return False
        root = root._parents[row[0]]
    return True


def browse_annotations(reader: TimeFReader, query: Any) -> dict[str, object]:
    """Return a bounded, qualified page of root-visible inherited occurrences.

    Returns:
        Annotation values and owners with a continuation that preserves dataset identity.

    Raises:
        TimeFValidationError: If the continuation cannot be decoded.
    """
    try:
        after = tuple(json.loads(query.after)) if query.after else ("", "", "")
        if len(after) != _CURSOR_FIELDS or not all(isinstance(item, str) for item in after):
            raise TimeFValidationError("invalid annotation cursor")
    except (ValueError, TypeError) as exc:
        raise TimeFValidationError("invalid annotation cursor") from exc
    items: list[dict[str, Any]] = []
    for node in sorted(
        ViewerInspection(reader).dependencies(),
        key=lambda item: (item.metadata.dataset_id, str(item.metadata.dataset_version)),
    ):
        identity = (node.metadata.dataset_id, str(node.metadata.dataset_version))
        if identity < after[:2]:
            continue
        cursor = after[2] if identity == after[:2] else ""
        while len(items) <= query.limit:
            rows = (
                node._control_reader()
                .connection.execute(
                    """WITH owners AS (
                   SELECT 'Dataset' AS object_type, dataset_key AS object_key, dataset_id AS object_id,
                          NULL::VARCHAR AS record_id FROM datasets
                   UNION ALL SELECT 'Record', record_key, record_id, record_id FROM records
                   UNION ALL SELECT 'Source', source_key, source_id, record_id FROM sources JOIN records USING (record_key)
                   UNION ALL SELECT 'Signal', signal_key, signal_id, record_id FROM signals
                       JOIN sources USING (source_key) JOIN records USING (record_key)
                   UNION ALL SELECT 'Task', task_key, task_id, NULL FROM tasks)
                   SELECT occurrence_id, owners.object_type, object_id, content_id, name, value_kind,
                          text_value, integer_value, float_value, boolean_value, text_list_value,
                          span_type, start_us, end_us, record_id
                   FROM annotation_occurrences JOIN annotation_contents USING (content_key)
                   JOIN owners USING (object_type, object_key)
                   WHERE occurrence_id > ? AND (? IS NULL OR owners.object_type = ?)
                     AND (? IS NULL OR object_id = ?) AND (? IS NULL OR span_type = ?)
                     AND contains(lower(coalesce(text_value, cast(integer_value AS VARCHAR),
                         cast(float_value AS VARCHAR), cast(boolean_value AS VARCHAR),
                         cast(text_list_value AS VARCHAR), '')), lower(?))
                   ORDER BY occurrence_id LIMIT 200""",
                    [
                        cursor,
                        query.object_type,
                        query.object_type,
                        query.object_id,
                        query.object_id,
                        query.span_type,
                        query.span_type,
                        query.value_query or "",
                    ],
                )
                .fetchall()
            )
            if not rows:
                break
            for row in rows:
                cursor = row[0]
                if _visible(reader, node, row[14]):
                    value = next((str(item) for item in row[6:11] if item is not None), None)
                    items.append(
                        {
                            "dataset_id": identity[0],
                            "version": identity[1],
                            "occurrence_id": row[0],
                            "object_type": row[1],
                            "object_id": row[2],
                            "content_id": row[3],
                            "name": row[4],
                            "value_kind": row[5],
                            "value": value[:_MAX_TEXT] if value else value,
                            "value_truncated": value is not None and len(value) > _MAX_TEXT,
                            "span_type": row[11],
                            "start_us": None if row[12] is None else str(row[12]),
                            "end_us": None if row[13] is None else str(row[13]),
                            "record_id": row[14],
                        }
                    )
                if len(items) > query.limit:
                    break
        if len(items) > query.limit:
            break
    page = items[: query.limit]
    cursor = (
        json.dumps([page[-1][key] for key in ("dataset_id", "version", "occurrence_id")])
        if len(items) > query.limit
        else None
    )
    return {"api_version": 1, "items": page, "next_cursor": cursor}
