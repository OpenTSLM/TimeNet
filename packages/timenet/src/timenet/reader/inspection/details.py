"""Bounded inspection projections, independent of HTTP and UI frameworks."""

# Projection functions document the returned facts in their summary.
# ruff: noqa: DOC201
# Only fixed projection SQL and integer budgets are interpolated; caller values are bound.
# ruff: noqa: S608
from __future__ import annotations

from typing import TYPE_CHECKING

from timenet.errors import TimeFValidationError
from timenet.reader.inspection.limits import MAX_METADATA
from timenet.reader.inspection.queries import ReaderQueries
from timenet.reader.inspection.requests import (
    OwnerRecordQuery,
    RecordDetailQuery,
)
from timenet.reader.inspection.results import Overview, OwnerRecord, RecordDetail


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader


def overview(reader: TimeFReader) -> Overview:
    """Return manifest-backed overview facts without reading signal values."""
    manifest = reader._manifest
    metadata = manifest.metadata
    counts = reader._manifest.counts
    return {
        "api_version": 1,
        "identity": {
            "dataset_id": metadata.dataset_id,
            "version": str(metadata.dataset_version),
            "name": metadata.name,
            "description": metadata.description,
            "license": str(metadata.license),
            "access": metadata.access.value,
            "domains": [domain.value for domain in metadata.domains],
            "tags": list(metadata.tags),
            "source_url": metadata.source_url,
        },
        "format": {"timef_version": manifest.timef_format_version, "values_backend": manifest.values_backend.value},
        "counts": {
            "records": str(counts.records),
            "sources": str(counts.sources),
            "signals": str(counts.signals),
            "tasks": str(sum(counts.tasks.values())),
            "annotation_occurrences": str(counts.annotation_occurrences),
        },
        "schema": {
            "spec_types": [spec.spec_type for spec in manifest.dataset_schema.time_series_specs],
            "task_types": [str(task.task_type) for task in manifest.dataset_schema.tasks],
            "tensor_specs": [
                {
                    "spec_type": spec.spec_type,
                    "value_shape": list(spec.value_shape),
                    "dimension_names": list(spec.dimension_names),
                }
                for spec in manifest.dataset_schema.time_series_specs
                if spec.value_shape
            ],
        },
        "dependencies": [str(dependency.dataset) for dependency in getattr(manifest, "dependencies", ())],
    }


def record_detail(reader: TimeFReader, body: RecordDetailQuery) -> RecordDetail:
    """Return persisted clock, span provenance and bounded metadata.

    Raises:
        TimeFValidationError: If metadata is unavailable.
    """
    ReaderQueries(reader).require_record(body.record_id)
    connection = reader._control_reader().connection
    row = connection.execute(
        f"""SELECT start_time_us, time_span_start_us, time_span_end_us, substring(metadata, 1, {MAX_METADATA + 1})
               FROM records LEFT JOIN clocks USING (clock_id) WHERE record_id = ?""",
        [body.record_id],
    ).fetchone()
    if row is None:
        raise TimeFValidationError("record metadata not found")
    result: RecordDetail = {
        "record_id": body.record_id,
        "dataset_id": reader.metadata.dataset_id,
        "version": str(reader.metadata.dataset_version),
        "start_time_us": None if row[0] is None else str(row[0]),
        "span": None if row[1] is None else {"start_us": str(row[1]), "end_us": str(row[2])},
        "span_provenance": "declared" if row[1] is not None else "not_declared",
        "metadata": (row[3] or "{}")[:MAX_METADATA],
        "metadata_truncated": len(row[3] or "") > MAX_METADATA,
    }
    signal_id = body.signal_id
    if body.source_id is not None:
        source_path, descendant = _source_context(reader, body.record_id, body.source_id)
        result["source_path"] = source_path
        signal_id = signal_id or descendant
    if signal_id is not None:
        signal = ReaderQueries(reader).signal(body.record_id, signal_id)
        stored = connection.execute(
            f"SELECT substring(metadata, 1, {MAX_METADATA + 1}) FROM signals WHERE signal_id = ?",
            [signal_id],
        ).fetchone()
        if stored is None:
            raise TimeFValidationError("signal metadata not found")
        metadata = stored[0]
        result["signal"] = {
            "signal_id": signal.id,
            "name": signal.name,
            "axis_kind": signal.time_axis.axis_type.value,
            "n_values": signal.n_values,
            "spec": signal.spec.model_dump(mode="json"),
            "metadata": metadata[:MAX_METADATA],
            "metadata_truncated": len(metadata) > MAX_METADATA,
        }
    return result


def _source_context(reader: TimeFReader, record_id: str, source_id: str) -> tuple[list[str], str | None]:
    """Resolve an owning source's path and a signal within its subtree.

    Raises:
        TimeFValidationError: If the source is outside the requested record.
    """
    connection = reader._control_reader().connection
    path = connection.execute(
        """WITH RECURSIVE ancestors AS (
            SELECT s.source_key, s.source_id, s.parent_source_key, 0 AS depth
            FROM sources s JOIN records USING (record_key)
            WHERE record_id = ? AND s.source_id = ?
            UNION ALL
            SELECT s.source_key, s.source_id, s.parent_source_key, a.depth + 1
            FROM sources s JOIN ancestors a ON s.source_key = a.parent_source_key
        ) SELECT source_id FROM ancestors ORDER BY depth DESC""",
        [record_id, source_id],
    ).fetchall()
    if not path:
        raise TimeFValidationError("source does not belong to this record")
    row = connection.execute(
        """WITH RECURSIVE descendants AS (
            SELECT source_key, [source_id] AS path FROM sources WHERE source_id = ?
            UNION ALL
            SELECT s.source_key, list_append(d.path, s.source_id)
            FROM sources s JOIN descendants d ON s.parent_source_key = d.source_key
        ) SELECT signal_id, path FROM signals JOIN descendants USING (source_key) ORDER BY signal_id LIMIT 1""",
        [source_id],
    ).fetchone()
    ancestors = [item[0] for item in path]
    return (ancestors, None) if row is None else (ancestors[:-1] + row[1], row[0])


def owner_record(reader: TimeFReader, body: OwnerRecordQuery) -> OwnerRecord | None:
    """Resolve a Source or Signal owner to its containing Record for viewer navigation."""
    query = (
        """SELECT records.record_id FROM sources
               JOIN records USING (record_key) WHERE sources.source_id = ?"""
        if body.object_type == "Source"
        else """SELECT records.record_id FROM signals
                    JOIN sources USING (source_key) JOIN records USING (record_key)
                    WHERE signals.signal_id = ?"""
    )
    row = reader._control_reader().connection.execute(query, [body.object_id]).fetchone()
    if row is None:
        return None
    return {
        "api_version": 1,
        "object_type": body.object_type,
        "record_id": row[0],
        "dataset_id": reader.metadata.dataset_id,
    }
