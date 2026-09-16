"""Load one dataset version's structure into a fresh embedded DuckDB database.

The writer attaches the database rather than opening it, because only a new file can take a block
size. One transaction holds everything: the DDL, every table's rows, the join that turns the
caller's ids into surrogate ids, then the checks in
:data:`~timenet.control_plane.checks.VALIDATIONS`. A failed check aborts before ``COMMIT``, so a
build that does not hold together publishes nothing.

A row that names an entity by the caller's id waits in a staging table while the walk runs. A record
can name a task the stream reaches later, and a task can name a record the same way. One join per
staged table resolves them all once the walk is over. The shipped file therefore joins on surrogate
ids and never on a string.

Rows reach DuckDB one Arrow table per batch, never through ``executemany``.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never

import duckdb
import pyarrow as pa

from timenet.control_plane import checks, schema as ddl
from timenet.control_plane.payload import PayloadKind, task_payload, text_answer
from timenet.control_plane.spans import span_row
from timenet.dataset import Record, TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, DatasetSchema, Task, TimeSeriesSpec


if TYPE_CHECKING:
    from timenet.values_backends.writer import ChunkPlacement


_INSERT_BATCH = 50_000
"""Rows buffered per table before the writer hands a batch to DuckDB."""

_STAGED = "_timef_staged_batch"
"""The name each batch is registered under while its ``INSERT`` runs."""

_ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "VARCHAR[]": pa.list_(pa.string()),
    "BIGINT": pa.int64(),
    "BIGINT[]": pa.list_(pa.int64()),
    "INTEGER": pa.int32(),
    "UINTEGER": pa.uint32(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
}
"""DuckDB's declared column types, mapped to the Arrow types a batch is built with."""

_STAGING_DDL_TEMPLATE = """
CREATE TEMP TABLE _stage_record_tasks (
    record_id  {id} NOT NULL,
    external_id VARCHAR NOT NULL
);

CREATE TEMP TABLE _stage_task_items (
    task_id   {id} NOT NULL,
    role       VARCHAR NOT NULL,
    position   INTEGER NOT NULL,
    item_type  VARCHAR NOT NULL,
    text_value VARCHAR,
    external_id VARCHAR
);

CREATE TEMP TABLE _stage_task_refs (
    task_id    {id} NOT NULL,
    field       VARCHAR NOT NULL,
    position    INTEGER NOT NULL,
    ref_kind    VARCHAR NOT NULL,
    external_id VARCHAR NOT NULL
);
"""
"""Where a row waits while the id it names is still unwritten.

A task can name a record, and a record a task, whichever the walk reaches first. The loader writes
the caller's id into one of these tables. Once the walk is over, :data:`_RESOLVE` turns the whole
table into surrogate ids with one join. These tables live in DuckDB's ``temp`` catalog and are never
part of the published file. They are declared here rather than beside the schema for that reason.
"""

_STAGING_DDL = _STAGING_DDL_TEMPLATE.format(id=ddl.ID_TYPE)
"""The staging tables, created beside the real ones inside the load transaction."""

# Each resolve sorts on the column its reads filter by, because the join that resolves the ids is
# free to hand its rows back in any order and an unsorted table loses the zone map that prunes the
# scan.
_RESOLVE = (
    (
        "record_tasks",
        "(record_id, task_id)",
        "SELECT s.record_id, t.task_id FROM _stage_record_tasks s "
        "LEFT JOIN tasks t ON t.external_id = s.external_id ORDER BY s.record_id",
    ),
    (
        "task_items",
        "(task_id, role, position, item_type, text_value, record_id)",
        "SELECT s.task_id, s.role, s.position, s.item_type, s.text_value, r.record_id "
        "FROM _stage_task_items s LEFT JOIN records r ON r.external_id = s.external_id "
        "ORDER BY s.task_id, s.role, s.item_type, s.position",
    ),
    (
        "task_refs",
        "(task_id, field, position, ref_kind, ref_id)",
        "SELECT s.task_id, s.field, s.position, s.ref_kind, "
        "CASE s.ref_kind WHEN 'record' THEN r.record_id ELSE ts.time_series_id END "
        "FROM _stage_task_refs s "
        "LEFT JOIN records r ON s.ref_kind = 'record' AND r.external_id = s.external_id "
        "LEFT JOIN time_series ts ON s.ref_kind = 'time_series' AND ts.external_id = s.external_id "
        "ORDER BY s.task_id, s.field, s.position",
    ),
)
"""Each staged table, its target's column list, and the query that resolves it.

The resolve step belongs to the write, not to the database, so it lives with the loader that runs
it. The published file has no staging table and no unresolved id.

An id that names nothing resolves to null rather than failing the insert. The load therefore reaches
:data:`~timenet.control_plane.checks.VALIDATIONS`, which reports that id by name with every other
one. Without the null, the bulk insert stops at the first offender.
"""


@dataclass(frozen=True)
class ValuesPlane:
    """What the values plane wrote, as the control plane needs to record it."""

    placements: dict[tuple[str, int], "ChunkPlacement"]
    """``(time_series_id, chunk_idx)`` mapped to where that chunk landed.

    The artifacts are the distinct ``chunk_file`` values these name. A Parquet chunk names a shard
    and a Zarr chunk names an array. The locator therefore says what an artifact is, not the file
    list the manifest keeps.
    """
    backend: str
    """Which backend wrote them, so a reader knows what the two chunk indexes mean."""


@dataclass(frozen=True)
class ControlPlaneCounts:
    """What the load stored, as the manifest's counts block records it."""

    records: int
    """How many records the version holds."""
    annotations: int
    """How many distinct annotation payloads were stored."""
    registered_annotations: int
    """How many of those no record carries, so the reader can skip the scan that recovers them."""
    tasks: dict[str, int]
    """How many tasks of each type."""
    chunks: int
    """How many chunk placements the values plane wrote."""
    record_series_chunks: int
    """Chunks counted once per referencing record."""
    specs: dict[str, int]
    """How many distinct series carry each spec type."""


def write_control_plane(  # noqa: PLR0913 - one parameter per plane the load needs to see
    db_path: Path,
    dataset: TimeFDataset,
    *,
    series: list[TimeSeries],
    series_to_records: dict[str, list[str]],
    values: ValuesPlane,
    tasks: Iterable[Task],
) -> ControlPlaneCounts:
    """Build the version's control database and leave it committed at ``db_path``.

    Args:
        db_path: Where to create the database. It must not exist yet.
        dataset: The populated dataset whose structure to load.
        series: The deduplicated series, in the order the values plane wrote them.
        series_to_records: Each series id mapped to the records that reference it.
        values: Where the values plane put every chunk.
        tasks: The dataset's tasks, materialized or streamed. Consumed once.

    Returns:
        The counts the manifest records.

    Raises:
        TimeFValidationError: If a check in :data:`~timenet.control_plane.checks.VALIDATIONS` finds a
            row. The database is left unpublished.
    """  # noqa: DOC502 - raised by _validate and by the loader's id counters
    connection = duckdb.connect()
    try:
        connection.execute(f"ATTACH '{db_path}' AS control (BLOCK_SIZE {ddl.BLOCK_SIZE})")
        connection.execute("USE control")
        connection.execute("BEGIN TRANSACTION")
        for statement in _statements(ddl.DDL + _STAGING_DDL):
            connection.execute(statement)
        counts = _load(connection, dataset, series, series_to_records, values, tasks)
        _resolve(connection)
        _validate(connection)
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT control")
    finally:
        connection.close()
    return counts


def _statements(script: str) -> list[str]:
    """Split a DDL script into individual statements.

    Args:
        script: The script to split.

    Returns:
        Each non-empty statement, stripped.
    """
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _resolve(connection: duckdb.DuckDBPyConnection) -> None:
    """Turn every staged caller id into the surrogate id of the row it names.

    One join per staged table, once the walk is over and both sides exist. An id that names nothing
    lands as null, which :data:`~timenet.control_plane.checks.VALIDATIONS` refuses.

    Args:
        connection: The connection holding the loaded, not yet committed database.
    """
    for table, columns, query in _RESOLVE:
        connection.execute(f"INSERT INTO {table} {columns} {query}")


def _validate(connection: duckdb.DuckDBPyConnection) -> None:
    """Check every invariant that stands in for a key constraint.

    Each check runs as a count, not as a row scan. The second query names the first few offending
    rows, and it runs only after a count finds a breach.

    Args:
        connection: The connection holding the loaded, not yet committed database.

    Raises:
        TimeFValidationError: If any check finds a row. The message names the invariant that broke
            and what to do about it.
    """
    for check in checks.VALIDATIONS:
        counted = connection.execute(check.count_query).fetchone()
        total = int(counted[0]) if counted else 0
        if total:
            sample = connection.execute(check.sample_query).fetchall()
            raise TimeFValidationError(check.failure(total, [row[0] for row in sample]))


class _Ids:
    """Hands out one entity kind's ids: 0, 1, 2, with no gap and no repeat."""

    def __init__(self, kind: str) -> None:
        """Start the counter for one kind of entity.

        Args:
            kind: What is being counted, named in the plural for the error message.
        """
        self._kind = kind
        self._next = 0

    def claim(self) -> int:
        """Take the next id.

        Returns:
            The id.

        Raises:
            TimeFValidationError: If the next id does not fit the column that stores it.
        """
        if self._next > ddl.MAX_ID:
            raise TimeFValidationError(f"too many {self._kind} for a {ddl.ID_TYPE} id: the limit is {ddl.MAX_ID + 1}")
        claimed = self._next
        self._next += 1
        return claimed


class _BatchInserter:
    """Buffers rows for one table and flushes them to DuckDB in batches.

    Nothing constrains the order rows arrive in. The database declares no foreign keys, and the
    checks that stand in for them run once against the finished load instead.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]) -> None:
        """Bind the inserter to its table and column list.

        Args:
            connection: The open database connection.
            table: The table to insert into.
            columns: The column names, in the order the rows supply them.
        """
        self._connection = connection
        self._columns = columns
        # Table and column names come from this package's own schema, never from input.
        self._sql = f"INSERT INTO {table} ({', '.join(columns)}) SELECT * FROM {_STAGED}"  # noqa: S608
        self._rows: list[tuple] = []
        self._schema: pa.Schema | None = None
        self.table = table
        self.count = 0

    def add(self, row: tuple) -> None:
        """Buffer one row, and flush the batch when it fills.

        Args:
            row: The row's values, in column order.
        """
        self._rows.append(row)
        self.count += 1
        if len(self._rows) >= _INSERT_BATCH:
            self.flush()

    def flush(self) -> None:
        """Write any buffered rows as one Arrow table."""
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        schema = self._arrow_schema()
        columns = zip(*rows, strict=True)
        staged = pa.Table.from_arrays(
            [pa.array(values, type=column.type) for values, column in zip(columns, schema, strict=True)],
            schema=schema,
        )
        self._connection.register(_STAGED, staged)
        try:
            self._connection.execute(self._sql)
        finally:
            self._connection.unregister(_STAGED)

    def _arrow_schema(self) -> pa.Schema:
        """Return the Arrow schema matching this table's columns, asking the database for the types.

        Returns:
            The schema, built once and reused for every batch.
        """
        if self._schema is None:
            described = dict(
                self._connection.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                    [self.table],
                ).fetchall()
            )
            self._schema = pa.schema([(name, _ARROW_TYPES[described[name]]) for name in self._columns])
        return self._schema


class _Attachments:
    """One annotation target kind: its table's inserter and its own dense attachment counter.

    Each kind counts from 0, so ``record_annotations`` and ``task_annotations`` both start at 0.
    Neither has to know how many rows the other holds.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, table: str, target: str | None, *, role: bool) -> None:
        """Bind the table to its target column and start its counter.

        Args:
            connection: The open database connection.
            table: The table holding this kind's attachments.
            target: The column naming the target, or ``None`` for the dataset's own table.
            role: Whether the table carries a ``role`` column.
        """
        columns = ["attachment_id", "annotation_id"]
        if target is not None:
            columns.append(target)
        if role:
            columns.append("role")
        columns.append("position")
        self.inserter = _BatchInserter(connection, table, tuple(columns))
        self._ids = _Ids(f"{table} rows")
        self._targeted = target is not None
        self._role = role

    def add(self, annotation_id: int, position: int, *, target: int | None = None, role: str | None = None) -> None:
        """Attach one annotation to one entity.

        Args:
            annotation_id: The id of the stored payload.
            position: Where the attachment sits in the target's own ordered list.
            target: The target entity's id, ignored by the dataset's table.
            role: Whether a task holds the annotation as input or as its answer.
        """
        row: tuple[Any, ...] = (self._ids.claim(), annotation_id)
        if self._targeted:
            row += (target,)
        if self._role:
            row += (role,)
        self.inserter.add((*row, position))


class _Loader:
    """Assigns every surrogate id and feeds every table's inserter."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.meta = _BatchInserter(connection, "meta", ("key", "value"))
        self.specs = _BatchInserter(
            connection,
            "specs",
            (
                "spec_id",
                "spec_type",
                "name",
                "unit_value",
                "data_source_type",
                "data_source_name",
                "data_source_provider",
                "dtype",
                "categories",
                "value_shape",
                "dimension_names",
                "nullable",
            ),
        )
        self.descriptors = _BatchInserter(
            connection,
            "annotation_descriptors",
            ("descriptor_id", "key", "annotation_type", "value_type", "unit", "description"),
        )
        self.axes = _BatchInserter(
            connection,
            "axes",
            ("axis_id", "axis_type", "period_numerator_us", "period_denominator", "start_index", "first_us", "last_us"),
        )
        self.records = _BatchInserter(
            connection,
            "records",
            (
                "record_id",
                "external_id",
                "start_time_us",
                "time_span_start_us",
                "time_span_end_us",
                "subject_ids",
            ),
        )
        self.record_tasks = _BatchInserter(connection, "_stage_record_tasks", ("record_id", "external_id"))
        self.series = _BatchInserter(
            connection,
            "time_series",
            ("time_series_id", "external_id", "signal", "source_id", "spec_id", "axis_id", "n_values"),
        )
        self.record_series = _BatchInserter(
            connection, "record_time_series", ("record_id", "time_series_id", "position")
        )
        self.annotations = _BatchInserter(
            connection,
            "annotations",
            (
                "annotation_id",
                "external_id",
                "key",
                "value",
                "source",
                "span_start_us",
                "span_end_us",
                "span_time_series_ids",
            ),
        )
        self.tasks = _BatchInserter(connection, "tasks", ("task_id", "external_id", "task_type", "prompt", "rationale"))
        self.task_items = _BatchInserter(
            connection,
            "_stage_task_items",
            ("task_id", "role", "position", "item_type", "text_value", "external_id"),
        )
        self.task_from_tasks = _BatchInserter(connection, "task_from_tasks", ("task_id", "position", "external_id"))
        self.task_fields = _BatchInserter(connection, "task_fields", ("task_id", "field", "text_value", "double_value"))
        self.task_refs = _BatchInserter(
            connection, "_stage_task_refs", ("task_id", "field", "position", "ref_kind", "external_id")
        )
        self.task_spans = _BatchInserter(
            connection,
            "task_spans",
            ("task_id", "field", "position", "frame", "start_at", "end_at", "time_series_ids"),
        )
        self.artifacts = _BatchInserter(connection, "values_artifacts", ("artifact_id", "chunk_file", "backend"))
        self.chunks = _BatchInserter(
            connection,
            "time_series_chunks",
            ("time_series_id", "chunk_idx", "artifact_id", "chunk_major_idx", "chunk_minor_idx", "n_values"),
        )
        self.attachments = {
            kind: _Attachments(connection, table, target, role=kind == "task")
            for kind, (table, target) in ddl.ANNOTATION_TABLES.items()
        }

        self.record_ids = _Ids("records")
        self.series_ids = _Ids("time series")
        self.task_ids = _Ids("tasks")
        self._axis_ids = _Ids("axes")
        self._annotation_ids = _Ids("annotations")
        self._artifact_ids = _Ids("values artifacts")

        self.series_id_of: dict[str, int] = {}
        self.annotation_id_of: dict[str, int] = {}
        self._spec_id_of: dict[str, int] = {}
        self._axis_id_of: dict[tuple, int] = {}
        self._artifact_id_of: dict[str, int] = {}

    def all(self) -> tuple[_BatchInserter, ...]:
        """Return every inserter, so the caller can flush them all at the end.

        Returns:
            Each table's inserter. Order does not matter: the database declares no foreign keys.
        """
        return (
            self.meta,
            self.specs,
            self.descriptors,
            self.axes,
            self.records,
            self.record_tasks,
            self.series,
            self.record_series,
            self.annotations,
            self.tasks,
            self.task_items,
            self.task_from_tasks,
            self.task_fields,
            self.task_refs,
            self.task_spans,
            self.artifacts,
            self.chunks,
            *(table.inserter for table in self.attachments.values()),
        )

    def declare(self, schema: DatasetSchema) -> None:
        """Store the dataset's derived type declaration, which every row below references.

        The specs and the annotation descriptors go in in the order the schema derived them. The ids
        the rest of the load references run in that order too.

        Args:
            schema: The dataset's derived schema.
        """
        for spec_id, spec in enumerate(schema.time_series_specs):
            self._spec_id_of[spec.spec_type] = spec_id
            self.specs.add((spec_id, *_spec_columns(spec)))
        for descriptor_id, descriptor in enumerate(schema.annotations):
            self.descriptors.add(
                (
                    descriptor_id,
                    descriptor.key,
                    str(descriptor.annotation_type),
                    descriptor.value_type,
                    descriptor.unit,
                    descriptor.description,
                )
            )

    def spec(self, spec_type: str) -> int:
        """Return the id of the stored spec a series carries.

        A million series reference the declaration by id, so the unit and the value type are stored
        once rather than per row.

        Args:
            spec_type: The spec's stable type tag.

        Returns:
            The spec id.

        Raises:
            TimeFValidationError: If the schema declares no spec of that type, which means the schema
                and the records it was derived from have drifted apart.
        """
        spec_id = self._spec_id_of.get(spec_type)
        if spec_id is None:
            raise TimeFValidationError(
                f"a series carries spec_type {spec_type!r}, which the dataset's schema does not declare"
            )
        return spec_id

    def axis(self, axis: TimeAxis) -> int:
        """Return the id of an axis, stored the first time a series uses it.

        Args:
            axis: The series' time axis.

        Returns:
            The axis id.
        """
        key = _axis_columns(axis)
        axis_id = self._axis_id_of.get(key)
        if axis_id is None:
            axis_id = self._axis_id_of[key] = self._axis_ids.claim()
            self.axes.add((axis_id, *key))
        return axis_id

    def annotation(self, annotation: Annotation) -> int:
        """Return the id of an annotation, with its payload stored the first time the id is seen.

        Args:
            annotation: The annotation whose payload to store.

        Returns:
            The annotation id.
        """
        annotation_id = self.annotation_id_of.get(annotation.id)
        if annotation_id is None:
            annotation_id = self.annotation_id_of[annotation.id] = self._annotation_ids.claim()
            # An annotation span is always on the recording timeline (Annotation rejects a step
            # span), so the frame column the task tables need is implied here.
            _, start_us, end_us, scope = (
                span_row(annotation.span) if annotation.span is not None else (None, None, None, None)
            )
            self.annotations.add(
                (
                    annotation_id,
                    annotation.id,
                    annotation.key,
                    None if annotation.value is None else json.dumps(annotation.value),
                    annotation.source,
                    start_us,
                    end_us,
                    scope,
                )
            )
        return annotation_id

    def artifact(self, chunk_file: str) -> int:
        """Return the id of a values artifact, assigned the first time the file is named.

        Args:
            chunk_file: The artifact's version-relative path.

        Returns:
            The artifact id.
        """
        artifact_id = self._artifact_id_of.get(chunk_file)
        if artifact_id is None:
            artifact_id = self._artifact_id_of[chunk_file] = self._artifact_ids.claim()
        return artifact_id


def _spec_columns(spec: TimeSeriesSpec) -> tuple:
    """Return a spec's stored columns, in the order the table declares them.

    Args:
        spec: The spec to store.

    Returns:
        Every column after ``spec_id``. A spec with no data source stores null in all three of its
        columns.
    """
    source = spec.data_source
    return (
        spec.spec_type,
        spec.name,
        str(spec.unit_value),
        None if source is None else source.data_source_type,
        None if source is None else source.name,
        None if source is None else source.provider,
        spec.dtype,
        list(spec.categories),
        list(spec.value_shape),
        list(spec.dimension_names),
        spec.nullable,
    )


def _axis_columns(axis: TimeAxis) -> tuple:
    """Return the stored columns of an axis, with every column its shape does not use set to null.

    The dispatch is positive and ends in :func:`~typing.assert_never`. A new axis shape therefore
    fails at type-check time, instead of storing a row of nulls under another shape's tag.

    Args:
        axis: The series' time axis.

    Returns:
        ``(axis_type, period_numerator_us, period_denominator, start_index, first_us, last_us)``.
    """
    if isinstance(axis, RegularAxis):
        return (str(axis.axis_type), axis.period_us.numerator, axis.period_us.denominator, axis.start_index, None, None)
    if isinstance(axis, IrregularAxis):
        return (str(axis.axis_type), None, None, None, axis.first_us, axis.last_us)
    if isinstance(axis, OrdinalAxis):
        return (str(axis.axis_type), None, None, None, None, None)
    assert_never(axis)


def _load(  # noqa: PLR0913, PLR0917
    connection: duckdb.DuckDBPyConnection,
    dataset: TimeFDataset,
    series: list[TimeSeries],
    series_to_records: dict[str, list[str]],
    values: ValuesPlane,
    tasks: Iterable[Task],
) -> ControlPlaneCounts:
    """Walk the dataset once and feed every table.

    Args:
        connection: The open connection to the database being built.
        dataset: The populated dataset.
        series: The deduplicated series, in the order the values plane wrote them.
        series_to_records: Each series id mapped to the records that reference it.
        values: Where the values plane put every chunk.
        tasks: The dataset's tasks. Consumed once.

    Returns:
        The counts the manifest records.

    Raises:
        TimeFValidationError: If the dataset's schema was never derived, so the database has no type
            declaration to store its rows against.
    """
    if dataset.schema is None:
        raise TimeFValidationError(
            "the control plane stores the dataset's derived schema, so derive_schema() must run before the write"
        )
    loader = _Loader(connection)
    loader.meta.add(("schema_version", str(ddl.SCHEMA_VERSION)))
    loader.meta.add(("dataset_id", dataset.metadata.dataset_id))
    loader.declare(dataset.schema)

    specs: dict[str, int] = {}
    for ts in series:
        series_id = loader.series_ids.claim()
        loader.series_id_of[ts.time_series_id] = series_id
        loader.series.add(
            (
                series_id,
                ts.time_series_id,
                ts.signal,
                ts.source_id,
                loader.spec(ts.spec.spec_type),
                loader.axis(ts.time_axis),
                ts.n_values,
            )
        )
        specs[ts.spec.spec_type] = specs.get(ts.spec.spec_type, 0) + 1

    # Records take their ids in sorted order, so the reader walks them in the order the Parquet
    # control plane sorted them into and a caller sees the same sequence as before.
    for record in sorted(dataset.records, key=lambda r: r.record_id):
        _load_record(loader, record)
    for position, annotation in enumerate(dataset.registered_annotations):
        loader.attachments["dataset"].add(loader.annotation(annotation), position)

    task_counts: dict[str, int] = {}
    for task in tasks:
        _load_task(loader, task)
        task_type = str(task.task_type)
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    # Declared before the chunks, so every chunk row can name an artifact that already has an id.
    for chunk_file in dict.fromkeys(placement.chunk_file for placement in values.placements.values()):
        loader.artifacts.add((loader.artifact(chunk_file), chunk_file, values.backend))
    for (time_series_id, chunk_idx), placement in values.placements.items():
        loader.chunks.add(
            (
                loader.series_id_of[time_series_id],
                chunk_idx,
                loader.artifact(placement.chunk_file),
                placement.data_index.major_idx,
                placement.data_index.minor_idx,
                placement.n_values,
            )
        )

    for inserter in loader.all():
        inserter.flush()
    chunks_per_series: dict[str, int] = {}
    for time_series_id, _chunk_idx in values.placements:
        chunks_per_series[time_series_id] = chunks_per_series.get(time_series_id, 0) + 1
    return ControlPlaneCounts(
        records=loader.records.count,
        annotations=loader.annotations.count,
        registered_annotations=len(dataset.registered_annotations),
        tasks=task_counts,
        chunks=len(values.placements),
        record_series_chunks=sum(
            chunks_per_series.get(time_series_id, 0) * len(record_ids)
            for time_series_id, record_ids in series_to_records.items()
        ),
        specs=specs,
    )


def _load_record(loader: _Loader, record: Record) -> None:
    """Insert one record, its links to its tasks and its series, and its annotations.

    Args:
        loader: The loader feeding the tables.
        record: The record to insert.
    """
    record_id = loader.record_ids.claim()
    time_span = record.time_span
    loader.records.add(
        (
            record_id,
            record.record_id,
            record.start_time,
            None if time_span is None else time_span.start_us,
            None if time_span is None else time_span.end_us,
            list(record.subject_ids),
        )
    )
    for task_id in record.task_ids:
        loader.record_tasks.add((record_id, task_id))
    for position, ts in enumerate(record.time_series):
        loader.record_series.add((record_id, loader.series_id_of[ts.time_series_id], position))
    for position, annotation in enumerate(record.annotations):
        loader.attachments["record"].add(loader.annotation(annotation), position, target=record_id)


def _load_task(loader: _Loader, task: Task) -> None:
    """Insert one task: the frame every task shares, then the payload its own type declares.

    A task's ordered items go in one table, whatever they are. The records it is about go in as
    input items, and a free-text answer as its target item. ``task_fields`` keeps what is a named
    field of the payload rather than an item of the task.

    Args:
        loader: The loader feeding the tables.
        task: The task to insert.

    Raises:
        TimeFValidationError: If the task references an annotation the version does not store.
    """
    task_id = loader.task_ids.claim()
    loader.tasks.add((task_id, task.id, str(task.task_type), task.prompt, task.rationale))
    for position, external_id in enumerate(task.record_ids):
        loader.task_items.add((task_id, "input", position, "record", None, external_id))
    answer = text_answer(task.task_type)
    if answer is not None and getattr(task, answer.name) is not None:
        loader.task_items.add((task_id, "target", 0, "text", str(getattr(task, answer.name)), None))
    for position, external_id in enumerate(task.from_task_ids):
        loader.task_from_tasks.add((task_id, position, external_id))
    if task.scope is not None:
        loader.task_spans.add((task_id, "scope", 0, *span_row(task.scope)))
    for role, annotation_ids in (("input", task.input_annotation_ids), ("target", task.target_annotation_ids)):
        for position, annotation_id in enumerate(annotation_ids):
            stored = loader.annotation_id_of.get(annotation_id)
            if stored is None:
                raise TimeFValidationError(
                    f"{type(task).__name__} {task.id!r} references annotation {annotation_id!r}, which no "
                    f"record carries and which is not registered with register_annotations"
                )
            loader.attachments["task"].add(stored, position, target=task_id, role=role)
    _load_task_payload(loader, task_id, task)


def _load_task_payload(loader: _Loader, task_id: int, task: Task) -> None:
    """Insert the payload fields a task's own type declares.

    Every field that is not ``None`` gets one ``task_fields`` row, whatever its kind. That row says
    the field is set. An empty list-valued field therefore stays distinguishable from an absent one.

    Args:
        loader: The loader feeding the tables.
        task_id: The task's surrogate id.
        task: The task whose payload to insert.

    Raises:
        TimeFValidationError: If the task's payload and its declaration have drifted apart.
    """  # noqa: DOC502 - raised by payload.task_payload
    answer = text_answer(task.task_type)
    for declared in task_payload(task.task_type):
        value = getattr(task, declared.name)
        # The free-text answer is an item of the task, stored by _load_task beside its records.
        if value is None or declared is answer:
            continue
        text = double = None
        if declared.kind is PayloadKind.TEXT:
            text = str(value)
        elif declared.kind is PayloadKind.NUMBER:
            double = float(value)
        loader.task_fields.add((task_id, declared.name, text, double))
        if not declared.stores_elements:
            continue
        items: tuple = tuple(value) if declared.is_list else (value,)
        if declared.kind is PayloadKind.SPAN:
            for position, span in enumerate(items):
                loader.task_spans.add((task_id, declared.name, position, *span_row(span)))
        else:
            ref_kind = "record" if declared.kind is PayloadKind.RECORD_REF else "time_series"
            for position, external_id in enumerate(items):
                loader.task_refs.add((task_id, declared.name, position, ref_kind, external_id))
