"""Bounded annotation projections for one TimeF dataset."""

# Only fixed projection SQL and integer budgets are interpolated; caller values are bound.
# ruff: noqa: S608
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from timenet.dataset import OrdinalAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.format.query_helpers import OBJECT_IDS_SQL
from timenet.reader.inspection.limits import MAX_TEXT, annotation_value_sql, bounded_text, validate_page
from timenet.reader.inspection.queries import ReaderQueries
from timenet.reader.inspection.requests import AnnotationQuery, AnnotationWindowQuery
from timenet.reader.inspection.results import AnnotationItem, AnnotationPage
from timenet.reader.inspection.window_bounds import window_bounds, x_coordinates


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


_CURSOR_FIELDS = 3


def window_annotations(reader: TimeFReader, query: AnnotationWindowQuery) -> AnnotationPage:
    """Return record, source, and signal occurrences visible in a window.

    Returns:
        A bounded page with qualified occurrence identities and an opaque continuation.

    Raises:
        TimeFValidationError: If the continuation is malformed.
    """
    validate_page(query.limit)
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
    inspection = ReaderQueries(reader)
    signal = inspection.signal(query.record_id, query.signal_id)
    start_us, end_us = query.start_us, query.end_us
    if start_us is not None and end_us is not None and start_us > end_us:
        raise TimeFValidationError("annotation window bounds must be ordered")
    has_steps = query.start is not None or query.stop is not None or query.full
    if has_steps and start_us is None and end_us is None:
        start, stop = window_bounds(inspection, signal, _OverlayBounds(query))
        if not isinstance(signal.time_axis, OrdinalAxis):
            if start == stop:
                return {"api_version": 1, "items": [], "next_cursor": None, "more": False}
            start_us = x_coordinates(inspection, query.record_id, query.signal_id, signal.time_axis, start, start + 1)[
                0
            ]
            end_us = (
                x_coordinates(inspection, query.record_id, query.signal_id, signal.time_axis, stop, stop + 1)[0]
                if stop < signal.n_values
                else x_coordinates(inspection, query.record_id, query.signal_id, signal.time_axis, stop - 1, stop)[0]
                + 1
            )
            if isinstance(signal.time_axis, RegularAxis):
                end_us = signal.time_axis.time_offset_us(stop)
    identity = (reader.metadata.dataset_id, str(reader.metadata.dataset_version))
    items: list[AnnotationItem] = []
    if identity >= after[:2]:
        rows = (
            reader._control_reader()
            .connection.execute(
                f"""WITH RECURSIVE ancestors AS (
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
                          value_kind, {annotation_value_sql()}
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
                    start_us,
                    start_us,
                    end_us,
                    end_us,
                    start_us,
                    start_us,
                    end_us,
                    end_us,
                    query.limit + 1,
                ],
            )
            .fetchall()
        )
        items = [
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
                "value": bounded_text(row[8]),
                "value_truncated": row[8] is not None and len(row[8]) > MAX_TEXT,
            }
            for row in rows
        ]
    more = len(items) > query.limit
    page = items[: query.limit]
    cursor = json.dumps([page[-1][key] for key in ("dataset_id", "version", "occurrence_id")]) if more else None
    return {"api_version": 1, "items": page, "more": more, "next_cursor": cursor}


def browse_annotations(reader: TimeFReader, query: AnnotationQuery) -> AnnotationPage:
    """Return a bounded, qualified page of annotation occurrences.

    Returns:
        Annotation values and owners with a continuation that preserves dataset identity.

    Raises:
        TimeFValidationError: If the continuation cannot be decoded.
    """
    validate_page(query.limit)
    try:
        decoded = json.loads(query.after) if query.after else ["", "", ""]
        if not isinstance(decoded, list):
            raise TimeFValidationError("invalid annotation cursor")
        after = tuple(decoded)
        if len(after) != _CURSOR_FIELDS or not all(isinstance(item, str) for item in after):
            raise TimeFValidationError("invalid annotation cursor")
    except (ValueError, TypeError) as exc:
        raise TimeFValidationError("invalid annotation cursor") from exc
    identity = (reader.metadata.dataset_id, str(reader.metadata.dataset_version))
    if identity < after[:2]:
        return {"api_version": 1, "items": [], "next_cursor": None}
    cursor = after[2] if identity == after[:2] else ""
    rows = (
        reader._control_reader()
        .connection.execute(
            f"""WITH owners AS ({OBJECT_IDS_SQL})
                   SELECT occurrence_id, owners.object_type, object_id, content_id, name, value_kind,
                          {annotation_value_sql()},
                          span_type, start_us, end_us
                   FROM annotation_occurrences JOIN annotation_contents USING (content_key)
                   JOIN owners USING (object_type, object_key)
                   WHERE occurrence_id > ? AND (? IS NULL OR owners.object_type = ?)
                     AND (? IS NULL OR object_id = ?) AND (? IS NULL OR span_type = ?)
                     AND contains(lower(coalesce(text_value, cast(integer_value AS VARCHAR),
                         cast(float_value AS VARCHAR), cast(boolean_value AS VARCHAR),
                         cast(text_list_value AS VARCHAR), '')), lower(?))
                   ORDER BY occurrence_id LIMIT ?""",
            [
                cursor,
                query.object_type,
                query.object_type,
                query.object_id,
                query.object_id,
                query.span_type,
                query.span_type,
                query.value_query or "",
                query.limit + 1,
            ],
        )
        .fetchall()
    )
    items: list[AnnotationItem] = []
    for row in rows:
        value = row[6]
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
                "value": value[:MAX_TEXT] if value else value,
                "value_truncated": value is not None and len(value) > MAX_TEXT,
                "span_type": row[7],
                "start_us": None if row[8] is None else str(row[8]),
                "end_us": None if row[9] is None else str(row[9]),
            }
        )
    page = items[: query.limit]
    cursor = (
        json.dumps([page[-1][key] for key in ("dataset_id", "version", "occurrence_id")])
        if len(items) > query.limit
        else None
    )
    return {"api_version": 1, "items": page, "next_cursor": cursor}


class _OverlayBounds:
    """Adapt optional overlay bounds to the shared signal window resolver."""

    def __init__(self, query: AnnotationWindowQuery) -> None:
        self.record_id = query.record_id
        self.signal_id = query.signal_id
        self.start = query.start or 0
        self.stop = query.stop
        self.start_us = query.start_us
        self.end_us = query.end_us
        self.full = query.full
