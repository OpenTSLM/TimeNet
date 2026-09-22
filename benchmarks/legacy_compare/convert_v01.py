"""Load a v0.1 TimeF version (Parquet control tables) into this stack's object model.

The converter reads ``records``, ``annotations``, ``tasks``, and ``time_series_index`` with pyarrow
and builds a :class:`~timenet.dataset.TimeFDataset`. Signal values stay lazy: each loader pulls its
chunks from the v0.1 value shards by row group and row, so the new writer streams them through.

Differences the conversion has to bridge, and how it does:

- v0.1 records hold a flat ``time_series`` list with a ``source_id`` per series. One Source per
  distinct ``source_id`` is created under the record, named after it and given a record-scoped id,
  since v0.1 source ids repeat across records and may be empty.
- v0.1 has no shared axis ids. Each signal gets its own axis, named after the signal.
- v0.1 annotation values are JSON. Map values are not supported any more, so they are converted to
  their JSON text; other values keep their type.
- Annotations that no record carries are attached to the Dataset so tasks can reference them.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.manifest.manifest import _metadata_from_dict  # noqa: PLC2701 - benchmark harness reuses the codec
from timenet.types import (
    TASKS,
    Annotation,
    Task,
    TaskType,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)


class _ShardCache:
    """Row-group reads from the v0.1 value shards, keeping the most recent few in memory."""

    def __init__(self, root: Path, capacity: int = 8) -> None:
        self.root = root
        self.capacity = capacity
        self._groups: OrderedDict[tuple[str, int], pa.Table] = OrderedDict()
        self._files: dict[str, pq.ParquetFile] = {}

    def cell(self, chunk_file: str, major: int, minor: int, column: str) -> pa.Array:
        """Return one list cell of a value shard as a flat Arrow array, without a Python round trip."""
        key = (chunk_file, major)
        table = self._groups.get(key)
        if table is None:
            parquet = self._files.get(chunk_file)
            if parquet is None:
                parquet = self._files[chunk_file] = pq.ParquetFile(self.root / chunk_file)
            table = parquet.read_row_group(major, columns=["values", "time_offsets_us"])
            self._groups[key] = table
            if len(self._groups) > self.capacity:
                self._groups.popitem(last=False)
        else:
            self._groups.move_to_end(key)
        scalar = table.column(column)[minor]
        return scalar.values if scalar.is_valid else pa.array([], type=table.schema.field(column).type.value_type)


def _span(struct: dict[str, Any] | None, rename: Any = None) -> TimeInterval | TimePoint | None:
    if struct is None or struct.get("start_us") is None:
        return None
    ids = struct.get("time_series_ids")
    scope = None if ids is None else tuple(rename(item) if rename else item for item in ids)
    if struct.get("end_us") is None:
        return TimePoint(start_us=struct["start_us"], time_series_ids=scope)
    return TimeInterval(start_us=struct["start_us"], end_us=struct["end_us"], time_series_ids=scope)


def _interval(struct: dict[str, Any] | None, rename: Any = None) -> TimeInterval | None:
    span = _span(struct, rename)
    return span if isinstance(span, TimeInterval) else None


def _annotation_value(text: str | None) -> Any:
    if text is None:
        return None
    value = json.loads(text)
    if isinstance(value, dict):
        return text
    if isinstance(value, list) and not all(isinstance(item, str) for item in value):
        return text
    return value


def _task_rows(root: Path) -> list[dict[str, Any]]:
    """Read every task partition, adding the partition's task type to each row.

    Returns:
        Task rows from every ``task=<type>`` directory, since the partitions have different columns.
    """
    rows: list[dict[str, Any]] = []
    for file in sorted((root / "tasks").glob("task=*/*.parquet")):
        task_type = file.parent.name.split("=", 1)[1]
        for row in pq.read_table(file).to_pylist():
            row["task"] = task_type
            rows.append(row)
    return rows


def load_v01(version_dir: Path) -> TimeFDataset:  # noqa: PLR0912, PLR0914, PLR0915 - one pass over four tables
    """Build a dataset from a v0.1 version directory.

    Returns:
        The dataset, with lazy Signal values and a derived schema.
    """
    root = Path(version_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    dataset = TimeFDataset(metadata=_metadata_from_dict(manifest["metadata"]))
    specs = {
        entry["spec_type"]: TimeSeriesSpec(
            spec_type=entry["spec_type"],
            name=entry["name"],
            unit_value=ureg.Unit(entry["unit_value"]),
            dtype=entry["dtype"],
            categories=tuple(entry.get("categories") or ()),
            value_shape=tuple(entry.get("value_shape") or ()),
            dimension_names=tuple(entry.get("dimension_names") or ()),
            nullable=bool(entry.get("nullable", False)),
        )
        for entry in manifest["schema"]["time_series_specs"]
    }
    units = {entry["key"]: entry.get("unit") for entry in manifest["schema"].get("annotations", ())}

    def table(name: str) -> pa.Table:
        files = sorted((root / name).rglob("*.parquet")) if (root / name).exists() else []
        return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default") if files else pa.table({})

    shards = _ShardCache(root)
    # The index has one row per (record, series, chunk). A series shared by several records repeats
    # its chunk rows, so keep one row per chunk.
    index: dict[str, dict[int, dict[str, Any]]] = {}
    for row in table("time_series_index").to_pylist():
        index.setdefault(row["time_series_id"], {}).setdefault(row["chunk_idx"], row)
    chunks_of = {series_id: [chunks[key] for key in sorted(chunks)] for series_id, chunks in index.items()}

    def loader(series_id: str, spec: TimeSeriesSpec, offsets: bool) -> Any:
        column = "time_offsets_us" if offsets else "values"

        def load() -> pa.Array:
            parts = [
                shards.cell(c["chunk_file"], c["chunk_major_idx"], c["chunk_minor_idx"], column)
                for c in chunks_of[series_id]
            ]
            flat = parts[0] if len(parts) == 1 else pa.concat_arrays(parts)
            target = pa.int64() if offsets else pa.from_numpy_dtype(spec.dtype)
            return flat if flat.type == target else flat.cast(target)

        return load

    record_rows = table("records").to_pylist()
    # v0.1 let several records share one series. A Signal has one owning Source now, so a shared
    # series becomes one Signal per record with a record-scoped id, and every span that names it
    # is renamed the same way.
    owners: dict[str, int] = {}
    for row in record_rows:
        for series in row["time_series"] or ():
            owners[series["time_series_id"]] = owners.get(series["time_series_id"], 0) + 1
    shared = {series_id for series_id, count in owners.items() if count > 1}

    def renamer(record_ids: list[str]) -> Any:
        def rename(series_id: str) -> str:
            if series_id not in shared:
                return series_id
            return f"{record_ids[0]}/{series_id}" if record_ids else series_id

        return rename

    records: dict[str, Record] = {}
    for row in record_rows:
        by_source: dict[str, list[Signal]] = {}
        rename = renamer([row["record_id"]])
        for series in row["time_series"] or ():
            spec = specs[series["spec_type"]]
            signal_id = rename(series["time_series_id"])
            axis_id = f"{signal_id}/axis"
            axis: RegularAxis | IrregularAxis | OrdinalAxis
            offsets_loader = None
            if series["axis_type"] == "regular":
                axis = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(series["period_numerator_us"], series["period_denominator"] or 1),
                    start_index=series.get("start_index") or 0,
                )
            elif series["axis_type"] == "irregular":
                axis = IrregularAxis(
                    axis_id=axis_id, first_us=series["first_time_offset_us"], last_us=series["last_time_offset_us"]
                )
                offsets_loader = loader(series["time_series_id"], spec, offsets=True)
            else:
                axis = OrdinalAxis(axis_id=axis_id)
            by_source.setdefault(series["source_id"], []).append(
                Signal.from_loader(
                    id=signal_id,
                    name=series["signal"],
                    spec=spec,
                    time_axis=axis,
                    n_values=series["n_values"],
                    loader=loader(series["time_series_id"], spec, offsets=False),
                    time_offsets_loader=offsets_loader,
                )
            )
        record = Record(
            record_id=row["record_id"],
            # v0.1 source ids repeat across records and may be empty; a Source needs a dataset-wide id and a name.
            sources=tuple(
                Source(
                    id=f"{row['record_id']}/{source_id or 'source'}", name=source_id or "source", signals=tuple(signals)
                )
                for source_id, signals in by_source.items()
            ),
            subject_ids=tuple(row.get("subject_ids") or ()),
            start_time=row.get("start_time_us"),
            time_span=_interval(row.get("time_span"), rename),
        )
        records[record.record_id] = record
        dataset.add_record(record=record)

    occurrences: dict[tuple[str, str], Annotation] = {}
    dataset_level: dict[str, Annotation] = {}
    for row in table("annotations").to_pylist():
        record_ids = list(row.get("record_ids") or ())
        content = Annotation(
            id=row["id"],
            key=row["key"],
            value=_annotation_value(row.get("value")),
            unit=units.get(row["key"]),
            span=_span(row.get("span"), renamer(record_ids)),
            source=row.get("source"),
        )
        if not record_ids:
            dataset_level[row["id"]] = dataset.annotate(content)
        for record_id in record_ids:
            occurrences[row["id"], record_id] = records[record_id].add_annotation(content, warn_when_outside=False)

    def resolve(annotation_id: str, record_ids: list[str]) -> Annotation:
        for record_id in record_ids:
            found = occurrences.get((annotation_id, record_id))
            if found is not None:
                return found
        return dataset_level[annotation_id]

    tasks: dict[str, Task] = {}
    parents: dict[str, list[str]] = {}
    for row in _task_rows(root):
        task_type = TaskType(row["task"])
        cls = TASKS[task_type]
        record_ids = list(row["record_ids"] or ())
        rename = renamer(record_ids)
        target = row.get("target")
        if task_type is TaskType.TEMPORAL_LOCALIZATION:
            targets: tuple[Any, ...] | None = tuple(
                span for span in (_span(item, rename) for item in target or ()) if span is not None
            )
        elif target is None:
            targets = None
        else:
            targets = (target,)
        kwargs: dict[str, Any] = {
            "id": row["id"],
            "inputs": tuple(records[record_id] for record_id in record_ids),
            "prompt": row.get("prompt"),
            "scope": _span(row.get("scope"), rename),
            "rationale": row.get("rationale"),
            "targets": targets,
            "input_annotations": tuple(resolve(a, record_ids) for a in row.get("input_annotation_ids") or ()),
            "target_annotations": tuple(resolve(a, record_ids) for a in row.get("target_annotation_ids") or ()),
        }
        for field in ("target_schema", "unit", "target_name"):
            if row.get(field) is not None and field in cls.__dataclass_fields__:
                kwargs[field] = row[field]
        if row.get("mode") is not None and "mode" in cls.__dataclass_fields__:
            kwargs["mode"] = row["mode"]
        tasks[row["id"]] = cls(**kwargs)
        parents[row["id"]] = list(row.get("from_task_ids") or ())
    for task_id, parent_ids in parents.items():
        if parent_ids:
            tasks[task_id].from_tasks = tuple(tasks[parent_id] for parent_id in parent_ids)
    dataset.add_tasks(tasks=tasks.values())
    dataset.derive_schema()
    return dataset
