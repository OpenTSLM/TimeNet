"""Download and convert OpenRCA as shared system-day records with question-answering tasks."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime
from functools import lru_cache, partial
import json
from pathlib import Path
import re
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, InputModality, Split, TimeInterval, TimeOrigin, TimeSeriesSpec
from timenet_connectors.datasets.microsoft.openrca.preparation import (
    OpenRcaSource,
    load_manifest,
    prepare,
)
from timenet_connectors.datasets.microsoft.openrca.release import RELEASE, SystemRelease
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_KIND_ORDER = {"metrics": 0, "traces": 1, "logs": 2}
_MISSING_PREVIEW = 3


class OpenRcaConnector(BaseConnector[OpenRcaSource]):
    """Connector for Microsoft's OpenRCA root-cause-analysis benchmark."""

    def download(self, cache_dir: Path) -> list[OpenRcaSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch metadata, then only the full days referenced by benchmark tasks.

        Args:
            cache_dir: Connector cache directory.

        Returns:
            One lightweight reference to the pinned Hub snapshot.

        Raises:
            TimeNetDownloadError: If the pinned snapshot lacks a required source file.
        """
        metadata = _metadata_paths()
        root = hub_snapshot(RELEASE.repository, RELEASE.revision, cache_dir, metadata)
        telemetry = _telemetry_paths(root)
        required = (*metadata, *telemetry)
        root = hub_snapshot(RELEASE.repository, RELEASE.revision, cache_dir, required)
        missing = [relative for relative in required if not (root / relative).is_file()]
        if missing:
            preview = ", ".join(repr(path) for path in missing[:_MISSING_PREVIEW])
            suffix = "" if len(missing) <= _MISSING_PREVIEW else f" and {len(missing) - _MISSING_PREVIEW} more"
            raise TimeNetDownloadError(f"{RELEASE.repository!r} at {RELEASE.revision!r} lacks {preview}{suffix}")
        return [OpenRcaSource(cache_dir=cache_dir, root=root, revision=RELEASE.revision)]

    def convert(self, raw_refs: list[OpenRcaSource], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Prepare daily telemetry lazily, then build shared Records and AnswerTasks.

        Args:
            raw_refs: The single source returned by :meth:`download`.

        Returns:
            The populated OpenRCA dataset.

        Raises:
            TimeFValidationError: If the connector receives anything other than one source.
        """
        if len(raw_refs) != 1:
            raise TimeFValidationError(f"OpenRCA convert expects one source, got {len(raw_refs)}")
        manifest_path = prepare(raw_refs[0], RELEASE)
        manifest = load_manifest(manifest_path)
        prepared_root = manifest_path.parent
        systems = {system.key: system for system in RELEASE.systems}
        dataset = TimeFDataset(metadata=self.metadata())
        for raw_record in cast("list[dict[str, Any]]", manifest["records"]):
            system = systems[str(raw_record["system"])]
            record, shared_context = _record(raw_record, system, prepared_root)
            dataset.add_record(record=record)
            incident_annotations: dict[tuple[int, int], Annotation] = {}
            for task_row in cast("list[dict[str, Any]]", raw_record["tasks"]):
                bounds = (int(task_row["window_start_us"]), int(task_row["window_end_us"]))
                incident = incident_annotations.get(bounds)
                if incident is None:
                    incident = _incident_annotation(record, system, *bounds)
                    incident_annotations[bounds] = incident
                requested = cast("list[str]", task_row["requested_fields"])
                metadata: dict[str, object] = {
                    "system": system.key,
                    "source_row": int(task_row["row_index"]),
                    "upstream_task_index": str(task_row["task_index"]),
                    "requested_fields": requested,
                    "root_count": int(task_row["root_count"]),
                }
                if "root cause occurrence datetime" in requested:
                    metadata["time_tolerance_seconds"] = 60
                dataset.add_task(
                    task=AnswerTask(
                        id=str(task_row["id"]),
                        inputs=(record,),
                        prompt=str(task_row["prompt"]),
                        targets=(str(task_row["target"]),),
                        input_annotations=(*shared_context, incident),
                        split=Split.TEST,
                        metadata=metadata,
                    )
                )
        return dataset


def _metadata_paths() -> tuple[str, ...]:
    paths = {
        relative
        for system in RELEASE.systems
        for relative in (f"{system.prefix}/query.csv", f"{system.prefix}/record.csv")
    }
    paths.update(system.deployment_workbook for system in RELEASE.systems if system.deployment_workbook is not None)
    return tuple(sorted(paths))


def _telemetry_paths(root: Path) -> tuple[str, ...]:
    paths = []
    for system in RELEASE.systems:
        days = _record_days(root / system.prefix / "record.csv", system.display_name)
        paths.extend(f"{system.prefix}/telemetry/{day}/{table.path}" for day in days for table in system.tables)
    return tuple(paths)


def _record_days(path: Path, system: str) -> tuple[str, ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "datetime" not in (reader.fieldnames or ()):
                raise TimeNetDownloadError(f"OpenRCA {system} record table lacks a datetime column")
            days = {datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S").strftime("%Y_%m_%d") for row in reader}
    except (OSError, ValueError) as exc:
        raise TimeNetDownloadError(f"cannot read OpenRCA {system} record table {path}: {exc}") from exc
    if not days:
        raise TimeNetDownloadError(f"OpenRCA {system} record table {path} is empty")
    return tuple(sorted(days))


def _record(
    raw: Mapping[str, Any],
    system: SystemRelease,
    prepared_root: Path,
) -> tuple[Record, tuple[Annotation, ...]]:
    record_id = str(raw["id"])
    kind_sources: dict[str, list[Source]] = {}
    for table_index, table in enumerate(cast("list[dict[str, Any]]", raw["tables"])):
        source = _table_source(record_id, table_index, table, prepared_root)
        kind_sources.setdefault(str(table["kind"]), []).append(source)
    roots = tuple(
        Source(
            id=f"{record_id}-{kind}",
            name=kind,
            sources=tuple(kind_sources[kind]),
        )
        for kind in sorted(kind_sources, key=lambda item: _KIND_ORDER[item])
    )
    record = Record(
        record_id=record_id,
        start_time=TimeOrigin(int(raw["origin_us"])),
        time_span=TimeInterval.micros(0, int(raw["time_span_end_us"])),
        sources=roots,
        metadata={
            "system": system.key,
            "source_day": str(raw["day"]),
            "timezone": RELEASE.timezone,
        },
    )
    schema_context = json.dumps(
        [
            {
                "file": table["path"],
                "source_timestamp_unit": table["timestamp_unit"],
                "timestamp_factor_to_us": table["timestamp_factor_to_us"],
                "group_columns": table["group_columns"],
                "signal_columns": [signal["column"] for signal in table["signals"]],
            }
            for table in cast("list[dict[str, Any]]", raw["tables"])
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    annotations = [
        Annotation(
            id=f"openrca-{system.key}-component-candidates",
            key="root_cause_component_candidates",
            value=list(system.component_candidates),
            description="Closed component vocabulary supplied to the official OpenRCA agent.",
        ),
        Annotation(
            id=f"openrca-{system.key}-reason-candidates",
            key="root_cause_reason_candidates",
            value=list(system.reason_candidates),
            description="Closed reason vocabulary supplied to the official OpenRCA agent.",
        ),
        Annotation(
            id=f"openrca-{system.key}-system-context",
            key="system_context",
            value=system.context,
            description="System and telemetry semantics supplied to the official OpenRCA agent.",
        ),
        Annotation(
            id=f"{record_id}-telemetry-schema",
            key="telemetry_schema",
            value=schema_context,
            description="Machine-readable mapping from raw telemetry files to row-aligned Signals.",
        ),
        Annotation(
            id="openrca-timezone-asia-shanghai",
            key="timezone",
            value=RELEASE.timezone,
            description="Timezone required by the benchmark for incident timestamps.",
        ),
    ]
    topology = cast("list[str]", raw.get("deployment_topology", []))
    if topology:
        annotations.append(
            Annotation(
                id=f"openrca-{system.key}-deployment-topology",
                key="deployment_topology",
                value=topology,
                description="Rows from the official Telecom component deployment workbook.",
            )
        )
    return record, record.add_annotations(annotations)


def _incident_annotation(record: Record, system: SystemRelease, start_us: int, end_us: int) -> Annotation:
    origin_us = cast("int", record.start_time.timestamp)
    start_offset, end_offset = start_us - origin_us, end_us - origin_us
    return record.annotate(
        Annotation(
            id=f"openrca-{system.key}-incident-{start_us}",
            key="incident_window",
            span=TimeInterval.micros(start_offset, end_offset),
            description="Thirty-minute incident interval stated in the source question.",
            metadata={
                "upstream_bounds": "inclusive",
                "timef_bounds": "half-open",
                "source_start_unix_us": start_us,
                "source_end_unix_us": end_us,
            },
        ),
        warn_when_outside=False,
    )


def _table_source(
    record_id: str,
    table_index: int,
    table: Mapping[str, Any],
    prepared_root: Path,
) -> Source:
    table_id = f"{record_id}-table-{table_index:02d}"
    group_sources = []
    for group_index, group in enumerate(cast("list[dict[str, Any]]", table["groups"])):
        group_id = f"{table_id}-group-{group_index:05d}"
        part_sources = tuple(
            _part_source(
                group_id,
                part_index,
                part,
                cast("list[dict[str, str]]", table["signals"]),
                str(table["path"]),
                cast("dict[str, object]", group["values"]),
                prepared_root,
            )
            for part_index, part in enumerate(cast("list[dict[str, Any]]", group["parts"]))
        )
        values = cast("dict[str, object]", group["values"])
        label = ", ".join(f"{name}={value}" for name, value in values.items()) or "all rows"
        group_sources.append(
            Source(
                id=group_id,
                name=label,
                sources=part_sources,
                metadata={"group_values": values},
            )
        )
    return Source(
        id=table_id,
        name=str(table["name"]),
        sources=tuple(group_sources),
        metadata={
            "raw_path": str(table["path"]),
            "group_columns": cast("list[str]", table["group_columns"]),
            "row_alignment": "shared_time_axis",
        },
    )


def _part_source(  # noqa: PLR0913, PLR0917 - all fields identify one bounded table part
    group_id: str,
    part_index: int,
    part: Mapping[str, Any],
    signal_configs: list[dict[str, str]],
    raw_path: str,
    group_values: dict[str, object],
    prepared_root: Path,
) -> Source:
    part_id = f"{group_id}-part-{part_index:05d}"
    path = prepared_root / str(part["file"])
    row_group = int(part["row_group"])
    n_values = int(part["n_values"])
    axis = IrregularAxis(
        first_us=int(part["first_us"]),
        last_us=int(part["last_us"]),
        axis_id=f"{part_id}-axis",
    )
    offsets_loader = partial(_read_column, path, row_group, "time_offset_us", n_values)
    signals = tuple(
        Signal.from_loader(
            id=f"{part_id}-signal-{column_index:02d}",
            name=config["column"],
            spec=_signal_spec(raw_path, config),
            time_axis=axis,
            time_offsets_loader=offsets_loader,
            loader=partial(_read_column, path, row_group, config["column"], n_values),
            n_values=n_values,
            source_id=raw_path,
            metadata={
                "raw_column": config["column"],
                "group_values": group_values,
                "row_aligned": True,
            },
        )
        for column_index, config in enumerate(signal_configs)
    )
    return Source(
        id=part_id,
        name=f"part {part_index:05d}",
        signals=signals,
        metadata={
            "prepared_file": str(part["file"]),
            "row_group": row_group,
            "part_index": part_index,
            "n_rows": n_values,
        },
    )


def _signal_spec(raw_path: str, config: Mapping[str, str]) -> TimeSeriesSpec:
    column = config["column"]
    dtype = config["dtype"]
    if dtype == "int64":
        raise TimeFFormatError(f"OpenRCA Signal {raw_path}:{column} uses unsupported int64 values")
    safe = re.sub(r"[^a-z0-9]+", "_", f"{raw_path}_{column}".lower()).strip("_")
    return TimeSeriesSpec(
        spec_type=f"openrca_{safe}",
        name=f"OpenRCA {Path(raw_path).stem} {column}",
        unit_value=None,
        dtype=dtype,
        nullable=True,
        modality=InputModality.TEXT if config["modality"] == "text" else InputModality.TIME_SERIES,
    )


def _read_column(path: Path, row_group: int, column: str, expected_length: int) -> pa.Array:
    try:
        stat = path.stat()
        parquet = _open_parquet(path, stat.st_mtime_ns, stat.st_size)
        array = parquet.read_row_group(row_group, columns=[column]).column(0).combine_chunks()
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(
            f"cannot read prepared OpenRCA column {column!r} row group {row_group} from {path}: {exc}"
        ) from exc
    if len(array) != expected_length:
        raise TimeFFormatError(
            f"prepared OpenRCA column {column!r} in {path} has {len(array)} rows, expected {expected_length}"
        )
    return array


@lru_cache(maxsize=8)
def _open_parquet(path: Path, _mtime_ns: int, _size: int) -> pq.ParquetFile:
    return pq.ParquetFile(path)


CONNECTOR = OpenRcaConnector
