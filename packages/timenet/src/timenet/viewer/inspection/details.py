"""Bounded inspection projections, independent of HTTP and UI frameworks."""

# Projection functions document the returned facts in their summary.
# ruff: noqa: DOC201
from __future__ import annotations

from typing import TYPE_CHECKING

from timenet.errors import TimeFValidationError
from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.inspection.requests import (
    OwnerRecordQuery,
    RecordDetailQuery,
)


if TYPE_CHECKING:
    from timenet.reader.reader import TimeFReader

_MAX_METADATA_TEXT = 16384
_MAX_TEXT = 512


def overview(reader: TimeFReader) -> dict[str, object]:
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
        "dependencies": [str(dependency.dataset) for dependency in manifest.dependencies],
    }


def record_detail(reader: TimeFReader, body: RecordDetailQuery) -> dict[str, object]:
    """Return persisted clock, span provenance and bounded metadata.

    Raises:
        TimeFValidationError: If metadata is unavailable.
    """
    owner = ViewerInspection(reader).record_owner(body.record_id)
    connection = owner._control_reader().connection
    row = connection.execute(
        """SELECT start_time_us, time_span_start_us, time_span_end_us, metadata
               FROM records LEFT JOIN clocks USING (clock_id) WHERE record_id = ?""",
        [body.record_id],
    ).fetchone()
    if row is None:
        raise TimeFValidationError("record metadata not found")
    result = {
        "record_id": body.record_id,
        "dataset_id": owner.metadata.dataset_id,
        "version": str(owner.metadata.dataset_version),
        "start_time_us": None if row[0] is None else str(row[0]),
        "span": None if row[1] is None else {"start_us": str(row[1]), "end_us": str(row[2])},
        "span_provenance": "declared" if row[1] is not None else "not_declared",
        "metadata": (row[3] or "{}")[:_MAX_METADATA_TEXT],
        "metadata_truncated": len(row[3] or "") > _MAX_METADATA_TEXT,
        "subject_ids": None,
        "subject_ids_status": "not_persisted_in_control_format",
    }
    if body.signal_id is not None:
        signal = ViewerInspection(reader).signal(body.record_id, body.signal_id)
        stored = connection.execute("SELECT metadata FROM signals WHERE signal_id = ?", [body.signal_id]).fetchone()
        if stored is None:
            raise TimeFValidationError("signal metadata not found")
        metadata = stored[0]
        result["signal"] = {
            "signal_id": signal.id,
            "spec": signal.spec.model_dump(mode="json"),
            "metadata": metadata[:_MAX_METADATA_TEXT],
            "metadata_truncated": len(metadata) > _MAX_METADATA_TEXT,
        }
    return result


def owner_record(reader: TimeFReader, body: OwnerRecordQuery) -> dict[str, object] | None:
    """Resolve a Source or Signal owner to its containing Record for viewer navigation."""
    query = (
        """SELECT records.record_id FROM sources
               JOIN records USING (record_key) WHERE sources.source_id = ?"""
        if body.object_type == "Source"
        else """SELECT records.record_id FROM signals
                    JOIN sources USING (source_key) JOIN records USING (record_key)
                    WHERE signals.signal_id = ?"""
    )
    for node in ViewerInspection(reader).dependencies():
        row = node._control_reader().connection.execute(query, [body.object_id]).fetchone()
        if row is None:
            continue
        try:
            owner = ViewerInspection(reader).record_owner(row[0])
        except TimeFValidationError:
            continue
        if owner is node:
            return {
                "api_version": 1,
                "object_type": body.object_type,
                "record_id": row[0],
                "dataset_id": node.metadata.dataset_id,
            }
    return None
