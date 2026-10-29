"""Prepare compressed OpenRCA telemetry for bounded lazy Signal loaders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.types.clock import unix_us
from timenet_connectors.datasets.microsoft.openrca.release import OpenRcaRelease, SystemRelease, TableRelease


_PREPARATION_VERSION = 3
_DAY_US = 86_400_000_000
_CSV_BLOCK_BYTES = 16 << 20
_DUCKDB_BATCH_ROWS = 262_144
_PARQUET_GROUP_BYTES = 32 << 20
_MILLISECONDS_THRESHOLD = 100_000_000_000
_TARGET_KEYS = {
    "datetime": "root cause occurrence datetime",
    "component": "root cause component",
    "reason": "root cause reason",
}


@dataclass(frozen=True)
class OpenRcaSource:
    """Pinned local Hub snapshot returned by download and consumed by conversion."""

    cache_dir: Path
    root: Path
    revision: str


def prepare(source: OpenRcaSource, release: OpenRcaRelease) -> Path:
    """Return an atomic prepared manifest, building it when the cache is absent or stale.

    Args:
        source: Validated release artifact paths.
        release: Pinned schemas and task definitions.

    Returns:
        Path to the prepared JSON manifest.
    """
    cache_key = _cache_key(source)
    destination = source.cache_dir / f"openrca-prepared-v{_PREPARATION_VERSION}"
    manifest_path = destination / "manifest.json"
    if _prepared_is_valid(manifest_path, cache_key):
        return manifest_path

    temporary = source.cache_dir / f".openrca-prepared-{uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        manifest = _prepare_release(source, release, temporary, cache_key)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest_path


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a prepared manifest and require its top-level mapping shape.

    Returns:
        The decoded manifest mapping.

    Raises:
        TimeFFormatError: If the manifest cannot be read as a JSON object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"cannot read prepared OpenRCA manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TimeFFormatError(f"prepared OpenRCA manifest {path} is not a JSON object")
    return cast("dict[str, Any]", value)


def _cache_key(source: OpenRcaSource) -> str:
    payload = {
        "version": _PREPARATION_VERSION,
        "revision": source.revision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _prepared_is_valid(manifest_path: Path, cache_key: str) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_manifest(manifest_path)
        if manifest.get("cache_key") != cache_key or manifest.get("preparation_version") != _PREPARATION_VERSION:
            return False
        return all(
            (manifest_path.parent / part["file"]).is_file()
            for record in manifest["records"]
            for table in record["tables"]
            for group in table["groups"]
            for part in group["parts"]
        )
    except (KeyError, TypeError, TimeFFormatError):
        return False


def _prepare_release(
    source: OpenRcaSource,
    release: OpenRcaRelease,
    root: Path,
    cache_key: str,
) -> dict[str, Any]:
    topology = _deployment_topology(source, release)
    timezone = ZoneInfo(release.timezone)
    records: list[dict[str, Any]] = []
    for system in release.systems:
        tasks_by_day = _tasks_by_day(source.root, system, release)
        for day, tasks in sorted(tasks_by_day.items()):
            records.append(_prepare_record(source.root, system, day, tasks, topology, timezone, root))
    return {
        "preparation_version": _PREPARATION_VERSION,
        "cache_key": cache_key,
        "upstream_commit": release.upstream_commit,
        "records": records,
    }


def _read_csv_rows(root: Path, member: str, required: frozenset[str]) -> list[dict[str, str]]:
    path = _require_file(root, member)
    try:
        with path.open(encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            fields = frozenset(reader.fieldnames or ())
            if not required <= fields:
                raise TimeFFormatError(f"OpenRCA member {member!r} lacks required columns {sorted(required - fields)}")
            return [dict(row) for row in reader]
    except UnicodeDecodeError as exc:
        raise TimeFFormatError(f"OpenRCA member {member!r} is not valid UTF-8 CSV") from exc


def _tasks_by_day(  # noqa: PLR0914 - source validation keeps row and grouped-root context together
    root: Path,
    system: SystemRelease,
    release: OpenRcaRelease,
) -> dict[str, list[dict[str, Any]]]:
    queries = _read_csv_rows(
        root,
        f"{system.prefix}/query.csv",
        frozenset({"task_index", "instruction", "scoring_points"}),
    )
    roots = _read_csv_rows(
        root,
        f"{system.prefix}/record.csv",
        frozenset({"timestamp", "datetime", "component", "reason"}),
    )
    if len(queries) != len(roots):
        raise TimeFFormatError(
            f"OpenRCA {system.display_name} has {len(queries)} query rows but {len(roots)} root-cause rows"
        )
    timezone = ZoneInfo(release.timezone)
    indexed_roots: list[tuple[int, int, dict[str, str]]] = []
    for index, row in enumerate(roots):
        try:
            timestamp = int(float(row["timestamp"]))
        except ValueError as exc:
            raise TimeFFormatError(
                f"OpenRCA {system.display_name} record row {index} has invalid timestamp {row['timestamp']!r}"
            ) from exc
        expected = datetime.fromtimestamp(timestamp, timezone).strftime("%Y-%m-%d %H:%M:%S")
        if row["datetime"] != expected:
            raise TimeFFormatError(
                f"OpenRCA {system.display_name} record row {index} datetime {row['datetime']!r} "
                f"does not match timestamp {timestamp} ({expected!r})"
            )
        if row["component"] not in system.component_candidates:
            raise TimeFFormatError(
                f"OpenRCA {system.display_name} record row {index} has unknown component {row['component']!r}"
            )
        if row["reason"] not in system.reason_candidates:
            raise TimeFFormatError(
                f"OpenRCA {system.display_name} record row {index} has unknown reason {row['reason']!r}"
            )
        indexed_roots.append((timestamp, index, row))

    roots_by_window: dict[int, list[tuple[int, int, dict[str, str]]]] = {}
    for item in indexed_roots:
        roots_by_window.setdefault(item[0] // 1800, []).append(item)
    tasks_by_day: dict[str, list[dict[str, Any]]] = {}
    for index, (query, indexed_root) in enumerate(zip(queries, indexed_roots, strict=True)):
        timestamp, _, _ = indexed_root
        task_index = query["task_index"]
        fields = release.task_fields.get(task_index)
        if fields is None:
            raise TimeFFormatError(
                f"OpenRCA {system.display_name} query row {index} has unknown task index {task_index!r}"
            )
        expected_start = datetime.fromtimestamp((timestamp // 1800) * 1800, timezone)
        start, end = _incident_window(query["instruction"], expected_start, system.display_name, index)
        window_roots = sorted(roots_by_window[timestamp // 1800], key=lambda item: (item[0], item[1]))
        target = {
            str(root_number): {_TARGET_KEYS[field]: root[field] for field in fields}
            for root_number, (_, _, root) in enumerate(window_roots, start=1)
        }
        day = start.strftime("%Y_%m_%d")
        tasks_by_day.setdefault(day, []).append(
            {
                "id": f"openrca-{system.key.replace('_', '-')}-{index:03d}",
                "row_index": index,
                "task_index": task_index,
                "prompt": query["instruction"],
                "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "requested_fields": [_TARGET_KEYS[field] for field in fields],
                "root_count": len(window_roots),
                "window_start_us": unix_us(start),
                "window_end_us": unix_us(end),
            }
        )
    return tasks_by_day


def _incident_window(
    prompt: str,
    expected_start: datetime,
    system: str,
    row_index: int,
) -> tuple[datetime, datetime]:
    end = expected_start + timedelta(minutes=30)
    date_forms = {
        expected_start.strftime("%Y-%m-%d"),
        f"{expected_start:%B} {expected_start.day}, {expected_start.year}",
        f"{expected_start:%b} {expected_start.day}, {expected_start.year}",
    }
    clock_forms = (expected_start.strftime("%H:%M"), end.strftime("%H:%M"))
    if not any(value.lower() in prompt.lower() for value in date_forms) or not all(
        value in prompt for value in clock_forms
    ):
        raise TimeFFormatError(
            f"OpenRCA {system} query row {row_index} does not state its expected incident window "
            f"{expected_start.isoformat()} to {end.isoformat()}"
        )
    return expected_start, end


def _prepare_record(  # noqa: PLR0913, PLR0917 - one record needs source, release, task, and cache context
    source_root: Path,
    system: SystemRelease,
    day: str,
    tasks: list[dict[str, Any]],
    topology: Mapping[str, list[str]],
    timezone: ZoneInfo,
    root: Path,
) -> dict[str, Any]:
    midnight = datetime.strptime(day, "%Y_%m_%d").replace(tzinfo=timezone)
    midnight_us = unix_us(midnight)
    record_root = root / "records" / system.key / day
    staged = []
    for config in system.tables:
        member = f"{system.prefix}/telemetry/{day}/{config.path}"
        raw_path, first_timestamp_us, timestamp_factor = _stage_table(source_root, member, config, record_root)
        staged.append((config, raw_path, first_timestamp_us, timestamp_factor))
    origin_us = min(
        midnight_us,
        *(first_timestamp_us for _, _, first_timestamp_us, _ in staged),
    )

    tables = []
    last_us = midnight_us + _DAY_US - origin_us - 1
    for config, raw_path, _, timestamp_factor in staged:
        table = _prepare_table(raw_path, config, timestamp_factor, origin_us, root)
        if table["groups"]:
            tables.append(table)
            last_us = max(last_us, *(part["last_us"] for group in table["groups"] for part in group["parts"]))
    record_id = f"openrca-{system.key.replace('_', '-')}-{day.replace('_', '')}"
    return {
        "id": record_id,
        "system": system.key,
        "day": day,
        "origin_us": origin_us,
        "time_span_end_us": max(_DAY_US, last_us + 1),
        "tables": tables,
        "tasks": tasks,
        "deployment_topology": topology.get(system.key, []),
    }


def _stage_table(
    source_root: Path,
    member: str,
    config: TableRelease,
    record_root: Path,
) -> tuple[Path, int, int]:
    table_slug = config.path.removesuffix(".csv").replace("/", "__")
    table_root = record_root / config.kind / table_slug
    table_root.mkdir(parents=True, exist_ok=True)
    raw_path = table_root / "raw.parquet"
    first_timestamp_us, timestamp_factor = _stage_csv(source_root, member, config, raw_path)
    return raw_path, first_timestamp_us, timestamp_factor


def _prepare_table(
    raw_path: Path,
    config: TableRelease,
    timestamp_factor: int,
    origin_us: int,
    prepared_root: Path,
) -> dict[str, Any]:
    try:
        groups = _group_table(raw_path, config, timestamp_factor, origin_us, prepared_root)
    finally:
        raw_path.unlink(missing_ok=True)
    signal_columns = [
        {
            "column": column,
            "dtype": dtype,
            "modality": "text" if column in config.text_columns else "time_series",
        }
        for column, dtype in config.columns.items()
        if column != config.timestamp and column not in config.groups
    ]
    return {
        "path": config.path,
        "name": Path(config.path).name,
        "kind": config.kind,
        "timestamp_unit": config.timestamp_unit,
        "timestamp_factor_to_us": timestamp_factor,
        "group_columns": list(config.groups),
        "signals": signal_columns,
        "groups": groups,
    }


def _stage_csv(  # noqa: PLR0914 - streaming state validates schema, row order, and timestamp units
    source_root: Path,
    member: str,
    config: TableRelease,
    destination: Path,
) -> tuple[int, int]:
    path = _require_file(source_root, member)
    with path.open(encoding="utf-8-sig", newline="") as text:
        header = next(csv.reader(text), None)
    expected = list(config.columns)
    if header != expected:
        raise TimeFFormatError(f"OpenRCA member {member!r} columns are {header!r}, expected {expected!r}")

    column_types = {name: _arrow_type(dtype) for name, dtype in config.columns.items()}
    options = pacsv.ConvertOptions(
        column_types=column_types,
        include_columns=expected,
        strings_can_be_null=False,
        null_values=[""],
        true_values=["True", "TRUE", "true", "1"],
        false_values=["False", "FALSE", "false", "0"],
    )
    writer: pq.ParquetWriter | None = None
    row_index = 0
    first_timestamp_us: int | None = None
    timestamp_factor: int | None = None
    timestamp_index = expected.index(config.timestamp)
    try:
        with path.open("rb") as stream:
            reader = pacsv.open_csv(
                stream,
                read_options=pacsv.ReadOptions(block_size=_CSV_BLOCK_BYTES, use_threads=True),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=options,
            )
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                indices = pa.array(np.arange(row_index, row_index + batch.num_rows, dtype=np.int64))
                indexed = batch.append_column("__row_index", indices)
                writer = writer or pq.ParquetWriter(destination, indexed.schema, compression="zstd")
                writer.write_batch(indexed)
                timestamps = batch.column(timestamp_index).to_numpy(zero_copy_only=False)
                batch_first = int(timestamps.min())
                batch_last = int(timestamps.max())
                batch_factor = _timestamp_factor(config.timestamp_unit, batch_first, batch_last, member)
                if timestamp_factor is not None and batch_factor != timestamp_factor:
                    raise TimeFFormatError(
                        f"OpenRCA telemetry member {member!r} changes timestamp units within the file"
                    )
                timestamp_factor = batch_factor
                batch_first_us = batch_first * batch_factor
                first_timestamp_us = (
                    batch_first_us if first_timestamp_us is None else min(first_timestamp_us, batch_first_us)
                )
                row_index += batch.num_rows
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise TimeFFormatError(f"cannot parse OpenRCA telemetry member {member!r}: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
    if row_index == 0:
        raise TimeFFormatError(f"OpenRCA telemetry member {member!r} contains no rows")
    if first_timestamp_us is None or timestamp_factor is None:  # pragma: no cover - row count proves this
        raise TimeFFormatError(f"OpenRCA telemetry member {member!r} has no timestamp")
    return first_timestamp_us, timestamp_factor


def _timestamp_factor(unit: str, first: int, last: int, member: str) -> int:
    if unit == "seconds":
        return 1_000_000
    if unit == "milliseconds":
        return 1_000
    if unit != "seconds_or_milliseconds":
        raise TimeFFormatError(f"unsupported OpenRCA timestamp unit {unit!r}")
    if last < _MILLISECONDS_THRESHOLD:
        return 1_000_000
    if first >= _MILLISECONDS_THRESHOLD:
        return 1_000
    raise TimeFFormatError(f"OpenRCA telemetry member {member!r} mixes Unix-second and Unix-millisecond timestamps")


def _arrow_type(dtype: str) -> pa.DataType:
    types = {"str": pa.string(), "float64": pa.float64(), "int64": pa.int64(), "bool": pa.bool_()}
    try:
        return types[dtype]
    except KeyError as exc:
        raise TimeFFormatError(f"unsupported OpenRCA release dtype {dtype!r}") from exc


def _group_table(
    raw_path: Path,
    config: TableRelease,
    timestamp_factor: int,
    origin_us: int,
    prepared_root: Path,
) -> list[dict[str, Any]]:
    table_root = raw_path.parent
    group_columns = list(config.groups)
    signal_columns = [column for column in config.columns if column != config.timestamp and column not in config.groups]
    selected = [*(_quoted(column) for column in group_columns)]
    selected.append(f"CAST({_quoted(config.timestamp)} AS BIGINT) * {timestamp_factor} - {origin_us} AS time_offset_us")
    selected.extend(_signal_expression(column, config) for column in signal_columns)
    ordering = [*(_quoted(column) for column in group_columns), _quoted(config.timestamp), '"__row_index"']
    escaped_path = str(raw_path).replace("'", "''")
    query = (
        # Identifiers come only from the checked-in release schema.
        f"SELECT {', '.join(selected)} FROM read_parquet('{escaped_path}') "  # noqa: S608
        f"ORDER BY {', '.join(ordering)}"
    )

    connection = duckdb.connect()
    temp_directory = table_root / "duckdb-tmp"
    temp_directory.mkdir()
    connection.execute("SET temp_directory = ?", [str(temp_directory)])
    groups: list[dict[str, Any]] = []
    current: _PreparedGroupWriter | None = None
    try:
        batches = connection.execute(query).to_arrow_reader(batch_size=_DUCKDB_BATCH_ROWS)
        for batch in batches:
            for key, data in _split_groups(batch, len(group_columns)):
                if current is None or current.key != key:
                    if current is not None:
                        groups.append(current.finish(prepared_root))
                    current = _PreparedGroupWriter(table_root, key, group_columns, data.schema)
                current.write(data)
        if current is not None:
            groups.append(current.finish(prepared_root))
    except duckdb.Error as exc:
        raise TimeFFormatError(f"cannot sort/group prepared OpenRCA table {config.path!r}: {exc}") from exc
    finally:
        connection.close()
        shutil.rmtree(temp_directory, ignore_errors=True)
    return groups


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _signal_expression(column: str, config: TableRelease) -> str:
    quoted = _quoted(column)
    if column not in config.nullable_strings:
        return quoted
    return f"CASE WHEN {quoted} IN ('', 'None', 'none', 'NULL', 'null') THEN NULL ELSE {quoted} END AS {quoted}"


def _split_groups(batch: pa.RecordBatch, group_count: int) -> Iterator[tuple[tuple[object, ...], pa.Table]]:
    if group_count == 0:
        yield (), pa.Table.from_batches([batch])
        return
    columns = [batch.column(index).to_pylist() for index in range(group_count)]
    keys = list(zip(*columns, strict=True))
    start = 0
    while start < batch.num_rows:
        stop = start + 1
        while stop < batch.num_rows and keys[stop] == keys[start]:
            stop += 1
        selected = batch.slice(start, stop - start).select(range(group_count, batch.num_columns))
        yield keys[start], pa.Table.from_batches([selected])
        start = stop


class _PreparedGroupWriter:
    """Write one contiguous semantic group as bounded Parquet row groups."""

    def __init__(
        self,
        table_root: Path,
        key: tuple[object, ...],
        group_columns: Sequence[str],
        schema: pa.Schema,
    ) -> None:
        self.key = key
        self._group_columns = tuple(group_columns)
        digest = hashlib.sha1(  # noqa: S324 - deterministic path identity, not cryptography
            json.dumps(key, ensure_ascii=False, default=str, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self._path = table_root / f"group-{digest}.parquet"
        self._writer = pq.ParquetWriter(self._path, schema, compression="zstd")

    def write(self, table: pa.Table) -> None:
        """Append a table, splitting it until each row group is memory-bounded."""
        for part in _bounded_tables(table):
            self._writer.write_table(part, row_group_size=part.num_rows)

    def finish(self, prepared_root: Path) -> dict[str, Any]:
        """Close the group and describe each resulting row group.

        Returns:
            Group values and bounded row-group descriptors.

        Raises:
            TimeFFormatError: If a row group's time offsets are invalid.
        """
        self._writer.close()
        parquet = pq.ParquetFile(self._path)
        time_index = parquet.schema_arrow.get_field_index("time_offset_us")
        parts = []
        for row_group in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group)
            statistics = metadata.column(time_index).statistics
            if statistics is None or not statistics.has_min_max:
                offsets = parquet.read_row_group(row_group, columns=["time_offset_us"]).column(0)
                first_us, last_us = int(offsets[0].as_py()), int(offsets[-1].as_py())
            else:
                first_us, last_us = int(statistics.min), int(statistics.max)
            if first_us < 0 or last_us < first_us:
                raise TimeFFormatError(f"prepared OpenRCA table {self._path} has invalid offsets {first_us}..{last_us}")
            parts.append(
                {
                    "file": self._path.relative_to(prepared_root).as_posix(),
                    "row_group": row_group,
                    "n_values": metadata.num_rows,
                    "first_us": first_us,
                    "last_us": last_us,
                }
            )
        return {
            "values": dict(zip(self._group_columns, self.key, strict=True)),
            "parts": parts,
        }


def _bounded_tables(table: pa.Table) -> Iterator[pa.Table]:
    if table.nbytes <= _PARQUET_GROUP_BYTES or table.num_rows == 1:
        yield table
        return
    rows = max(1, table.num_rows * _PARQUET_GROUP_BYTES // table.nbytes)
    for start in range(0, table.num_rows, rows):
        yield from _bounded_tables(table.slice(start, min(rows, table.num_rows - start)))


def _deployment_topology(source: OpenRcaSource, release: OpenRcaRelease) -> dict[str, list[str]]:
    systems = [system for system in release.systems if system.deployment_workbook is not None]
    if not systems:
        return {}
    try:
        import openpyxl  # noqa: PLC0415  # ty: ignore[unresolved-import] - connector dependency
    except ImportError as exc:
        raise TimeFFormatError(
            "OpenRCA's Telecom deployment map requires openpyxl; install the connector requirements"
        ) from exc
    topology: dict[str, list[str]] = {}
    for system in systems:
        workbook_name = cast("str", system.deployment_workbook)
        workbook = openpyxl.load_workbook(_require_file(source.root, workbook_name), read_only=True, data_only=True)
        rows = []
        try:
            for sheet in workbook.worksheets:
                for values in sheet.iter_rows(values_only=True):
                    present = [str(value).strip() for value in values if value is not None and str(value).strip()]
                    if present:
                        rows.append(f"{sheet.title}: " + " | ".join(present))
        finally:
            workbook.close()
        topology[system.key] = rows
    return topology


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise TimeFFormatError(f"OpenRCA source lacks required file {relative!r}")
    return path
