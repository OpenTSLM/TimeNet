"""Compile a declarative hierarchy into a dataset version: one control database plus Parquet shards.

The writer walks ``Task -> Record -> Source -> Signal``, hands every entity the dense ``INTEGER`` id
the control plane joins on, streams the values into the values plane, and inserts the structure into
an embedded DuckDB database. The caller's own string id rides along as ``external_id`` on the entity
that owns it, so a record stays addressable by the name it has outside the dataset.

Commit works the way the current writer's does: everything is built in a ``<version>.tmp-<uuid>``
staging directory, and the finished directory is moved into place with one atomic rename. A reader
either sees a complete version or sees nothing.
"""

from collections.abc import Iterable
from pathlib import Path
import shutil
from types import TracebackType
from typing import Any, Self
import uuid

import duckdb
import pyarrow as pa

from timenet.control_plane import schema as ddl
from timenet.control_plane.model import Annotation, DeclarativeDataset, Record, RecordRef, Signal, Source, Task
from timenet.control_plane.values import PARQUET, ChunkLocator, PendingSignal, ValuesWriter, axis_columns, values_writer
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import CONTROL_DB_FILE, MANIFEST_FILE
from timenet.manifest import Manifest
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import FilePart, ManifestFiles
from timenet.types import DatasetMetadata


# Rows buffered per table before a batch is handed to DuckDB. The hierarchy is walked as a stream
# and flushed in batches, so a build never holds every row of a table in memory at once.
_INSERT_BATCH = 50_000

# The name each batch is registered under while its INSERT runs.
_STAGED = "_timef_staged_batch"

# Where task items wait until every record has been loaded, so their caller-supplied record names
# can be resolved in one join.
_STAGED_ITEMS = "_timef_staged_task_items"

_STAGED_ITEMS_DDL = f"""
CREATE TEMP TABLE {_STAGED_ITEMS} (
    task_id            {ddl.ID_TYPE} NOT NULL,
    role               VARCHAR NOT NULL,
    position           INTEGER NOT NULL,
    item_type          VARCHAR NOT NULL,
    text_value         VARCHAR,
    record_external_id VARCHAR
)
"""

# DuckDB's declared column types, mapped to the Arrow types a batch is built with.
_ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "BIGINT": pa.int64(),
    "UBIGINT": pa.uint64(),
    "INTEGER": pa.int32(),
    "UINTEGER": pa.uint32(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
}

# DuckDB allocates storage a block at a time, and every table and index segment claims at least one
# block, so the default 256 KiB block sets a floor of a few megabytes on a database that holds a few
# hundred rows. A dataset's control plane is small and is downloaded before it is read, so the floor
# matters more than the throughput a larger block buys. Measured on the demo dataset: 12.9 MB at the
# default, 815 KB at 16 KiB. 16 KiB is DuckDB's minimum.
_BLOCK_SIZE = 16_384

# Only a source attachment records the record it sits in. A record attachment already names one, and
# a signal shared by several records belongs to no single one.
_SCOPED_ATTACHMENTS = frozenset({"source"})


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
            TimeFValidationError: If the id would not fit the ``INTEGER`` column that stores it.
        """
        if self._next > ddl.MAX_ID:
            raise TimeFValidationError(f"too many {self._kind} for an INTEGER id: the limit is {ddl.MAX_ID + 1}")
        claimed = self._next
        self._next += 1
        return claimed


class _BatchInserter:
    """Buffers rows for one table and flushes them to DuckDB in batches.

    Nothing constrains the order rows arrive in: the shipped database declares no foreign keys, and
    the invariants they used to enforce are checked once against the finished database instead. The
    walk can therefore interleave records, sources, and signals freely.
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
        # Table and column names come from this module's own schema, never from input.
        self._sql = f"INSERT INTO {table} ({', '.join(columns)}) SELECT * FROM {_STAGED}"  # noqa: S608
        self._rows: list[tuple] = []
        self._schema: pa.Schema | None = None
        self.table = table
        self.count = 0

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

    def add(self, row: tuple) -> None:
        """Buffer one row, flushing when the batch fills.

        Args:
            row: The row's values, in column order.
        """
        self._rows.append(row)
        self.count += 1
        if len(self._rows) >= _INSERT_BATCH:
            self.flush()

    def flush(self) -> None:
        """Write any buffered rows.

        The batch goes in as one Arrow table rather than through ``executemany``. Measured on the
        ECG-QA control plane, 1.35M rows: 22 minutes row-by-row against 17.9 seconds this way.
        """
        if not self._rows:
            return
        rows = self._rows
        self._rows = []
        self._insert(rows)

    def _insert(self, rows: list[tuple]) -> None:
        """Write one batch as a single Arrow table.

        Args:
            rows: The rows to write, in column order.
        """
        schema = self._arrow_schema()
        staged = pa.Table.from_arrays(
            [pa.array(column, type=field.type) for column, field in zip(zip(*rows, strict=True), schema, strict=True)],
            schema=schema,
        )
        self._connection.register(_STAGED, staged)
        try:
            self._connection.execute(self._sql)
        finally:
            self._connection.unregister(_STAGED)


class _AttachmentTable:
    """One annotation target kind: its table's inserter and its own dense attachment counter.

    Each kind counts from 0, so ``record_annotations`` and ``signal_annotations`` both start at zero
    and neither has to know how many rows the other holds.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, kind: str, table: str, target: str | None) -> None:
        """Bind the table to its target column and start its counter.

        Args:
            connection: The open database connection.
            kind: The object kind, a key of :data:`~timenet.control_plane.schema.ANNOTATION_TABLES`.
            table: The table holding this kind's attachments.
            target: The column naming the target, or ``None`` for the dataset's own table.
        """
        columns = ["attachment_id", "annotation_id"]
        if target is not None:
            columns.append(target)
        if kind in _SCOPED_ATTACHMENTS:
            columns.append("scope_record_id")
        columns += ["span_type", "start_us", "end_us", "provenance", "confidence"]
        self.inserter = _BatchInserter(connection, table, tuple(columns))
        self._ids = _Ids(f"{table} rows")
        self._targeted = target is not None
        self._scoped = kind in _SCOPED_ATTACHMENTS

    def add(self, annotation_id: int, annotation: Annotation, target: int | None, scope_record_id: int | None) -> None:
        """Attach one annotation to one object.

        Args:
            annotation_id: The id of the stored payload.
            annotation: The annotation, for its span and provenance columns.
            target: The target object's id, ignored by the dataset's table.
            scope_record_id: The record the target sits in, used by ``source_annotations`` only.
        """
        row: tuple[Any, ...] = (self._ids.claim(), annotation_id)
        if self._targeted:
            row += (target,)
        if self._scoped:
            row += (scope_record_id,)
        self.inserter.add(
            (
                *row,
                annotation.span_type,
                annotation.start_us,
                annotation.end_us,
                annotation.provenance,
                annotation.confidence,
            )
        )


class TimeFWriter:
    """Writes one dataset version: a DuckDB control plane plus a Parquet or Zarr values plane."""

    def __init__(
        self, root: Path, metadata: DatasetMetadata, *, values_backend: str = PARQUET, **values_options: Any
    ) -> None:
        """Bind the writer to its output location and the version's metadata.

        Args:
            root: The registry root. The version lands at ``<root>/<dataset_id>/<version>``.
            metadata: The dataset's metadata, including its id and version.
            values_backend: Which values plane to write, from
                :data:`~timenet.control_plane.values.VALUES_BACKENDS`.
            values_options: Byte budgets and codec settings forwarded to that backend.

        Raises:
            TimeFValidationError: If the version is already committed at this location.
        """
        self._metadata = metadata
        self._final_dir = Path(root) / metadata.dataset_id / str(metadata.dataset_version)
        if (self._final_dir / MANIFEST_FILE).exists():
            raise TimeFValidationError(f"{self._final_dir} already holds a committed version")
        self._staging_dir = self._final_dir.parent / f"{self._final_dir.name}.tmp-{uuid.uuid4().hex}"
        self._values_backend = values_backend
        self._values_options = values_options
        self._manifest: Manifest | None = None

    def __enter__(self) -> Self:
        """Create the staging directory.

        Returns:
            This writer.
        """
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Remove the staging directory if the build failed."""
        if exc_type is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)

    @property
    def manifest(self) -> Manifest:
        """The manifest of the committed version.

        Raises:
            TimeFValidationError: If the version has not been written yet.
        """
        if self._manifest is None:
            raise TimeFValidationError("nothing written yet")
        return self._manifest

    def write(self, dataset: DeclarativeDataset) -> Manifest:
        """Write a whole in-memory dataset and commit it atomically.

        Args:
            dataset: The declarative hierarchy to compile.

        Returns:
            The manifest of the committed version.
        """
        return self.write_stream(dataset.records, dataset.tasks, annotations=dataset.annotations)

    def write_stream(
        self,
        records: Iterable[Record],
        tasks: Iterable[Task] = (),
        *,
        annotations: Iterable[Annotation] = (),
    ) -> Manifest:
        """Write a dataset from iterators, without ever holding the whole hierarchy in memory.

        Records are consumed one at a time: each one's control rows go into the batch inserters and
        its values into the shard streams, and both flush as they fill. A corpus larger than memory
        can therefore be written, which the in-memory path cannot do -- 618,508 records do not fit on
        a 16 GB machine.

        Tasks may be yielded in any order relative to the records they name, since a task item's
        record is resolved by a join once every record is loaded. An id naming no record is still
        rejected by the write-time validation.

        Args:
            records: The records to write, in any order.
            tasks: The tasks to write.
            annotations: Annotations that apply to the dataset as a whole.

        Returns:
            The manifest of the committed version.
        """
        db_path = self._staging_dir / CONTROL_DB_FILE
        values = values_writer(self._values_backend, self._staging_dir, **self._values_options)
        # The block size can only be set when the database is created, so the file is attached
        # rather than opened directly.
        connection = duckdb.connect()
        try:
            connection.execute(f"ATTACH '{db_path}' AS control (BLOCK_SIZE {_BLOCK_SIZE})")
            connection.execute("USE control")
            connection.execute("BEGIN TRANSACTION")
            for statement in _statements(ddl.DDL):
                connection.execute(statement)
            counts = _stream_into(connection, values, records, tasks, annotations, metadata=self._metadata)
            _validate(connection)
            connection.execute("COMMIT")
            for statement in _statements(ddl.INDEXES):
                connection.execute(statement)
            connection.execute("CHECKPOINT control")
        finally:
            connection.close()

        self._manifest = self._commit(db_path, values, counts)
        return self._manifest

    def _commit(self, db_path: Path, values: ValuesWriter, counts: dict[str, int]) -> Manifest:
        """Write the manifest, then move the staging directory into place atomically.

        Returns:
            The manifest of the committed version.

        Raises:
            TimeFValidationError: If the control database is missing from the staging directory.
        """
        manifest = Manifest(
            dataset_id=self._metadata.dataset_id,
            metadata=self._metadata,
            files=ManifestFiles(
                control_db=_file_part(self._staging_dir, CONTROL_DB_FILE),
                time_series=tuple(_file_part(self._staging_dir, part) for part in values.parts),
            ),
            counts=ManifestCounts(
                records=counts["records"],
                annotations=counts["annotations"],
                registered_annotations=counts["attachments"],
                time_series_chunks=counts["signal_chunks"],
                time_series_index_rows=counts["signal_chunks"],
            ),
            value_encoding=values.value_encoding,
            values_backend=self._values_backend,
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())
        if not db_path.exists():
            raise TimeFValidationError(f"control database missing at {db_path}")
        if self._final_dir.exists():
            shutil.rmtree(self._final_dir)
        self._staging_dir.replace(self._final_dir)
        return manifest


def _validate(connection: duckdb.DuckDBPyConnection) -> None:
    """Check every invariant the dropped key constraints used to enforce.

    A version is written once and never changed, so these hold for the rest of its life once they
    hold here. Each check is one bulk anti-join, and a failure aborts the build before anything is
    published.

    Args:
        connection: The connection holding the freshly loaded, not yet committed database.

    Raises:
        TimeFValidationError: If any check finds a row.
    """
    for description, query in ddl.VALIDATIONS:
        offending = connection.execute(query).fetchall()
        if offending:
            sample = ", ".join(str(row[0]) for row in offending[:3])
            raise TimeFValidationError(
                f"{description}: {len(offending)} row(s), for example {sample}. The version was not published."
            )


def _file_part(root: Path, relpath: str) -> FilePart:
    """Describe one written file for the manifest.

    Returns:
        The file's path, checksum, and size.
    """
    path = root / relpath
    return FilePart(path=relpath, checksum=file_checksum(path), size=path.stat().st_size)


def _statements(script: str) -> Iterable[str]:
    """Split a DDL script into individual statements.

    Returns:
        Each non-empty statement, stripped.
    """
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _stream_into(  # noqa: PLR0913
    connection: duckdb.DuckDBPyConnection,
    values: ValuesWriter,
    records: Iterable[Record],
    tasks: Iterable[Task],
    annotations: Iterable[Annotation],
    *,
    metadata: DatasetMetadata,
) -> dict[str, int]:
    """Walk the records and tasks once, feeding both planes as it goes.

    Args:
        connection: The open connection to the database being built.
        values: The values-plane writer to stream signal values into.
        records: The records to write.
        tasks: The tasks to write.
        annotations: Dataset-level annotations.
        metadata: The dataset's metadata, whose id goes into ``meta``.

    Returns:
        The row count of each loaded table.
    """
    connection.execute(_STAGED_ITEMS_DDL)
    loader = _Loader(connection)
    loader.meta.add(("schema_version", str(ddl.SCHEMA_VERSION)))
    loader.meta.add(("dataset_id", metadata.dataset_id))
    loader.annotate(list(annotations), "dataset", None)

    # A series shared by several records is written once. _load_record already knows which signals it
    # inserted for the first time, so it hands them back and the walk needs no set of every signal id
    # it has seen: 1.66 million strings on the largest corpus built so far.
    for record in records:
        for signal, signal_id in _load_record(loader, record):
            for locator in values.add(_pending(signal, signal_id)):
                _add_chunk(loader, locator)
    for task in tasks:
        _load_task(loader, task)
    for locator in values.finish():
        _add_chunk(loader, locator)
    # Declared after the values plane closes, so a chunk row can be anti-joined against the
    # artifacts that really exist rather than against the ones the writer meant to create. An
    # artifact no chunk names is given an id here, which keeps the ids dense.
    for chunk_file, backend in values.artifacts:
        loader.artifacts.add((loader.artifact(chunk_file), chunk_file, backend))

    for inserter in loader.all():
        inserter.flush()
    _resolve_task_items(connection)
    return {
        "records": loader.records.count,
        "sources": loader.sources.count,
        "signals": loader.signals.count,
        "source_signals": loader.source_signals.count,
        "values_artifacts": loader.artifacts.count,
        "signal_chunks": loader.chunks.count,
        "tasks": loader.tasks.count,
        "task_items": loader.staged_items.count,
        "annotations": loader.contents.count,
        "attachments": loader.attachment_count,
    }


def _resolve_task_items(connection: duckdb.DuckDBPyConnection) -> None:
    """Turn every staged task item's record name into a record id, then drop the staging table.

    A task names its records by the id the caller gave them, so the writer would otherwise have to
    hold every record's name in a dict until the tasks arrive: 618,508 entries on SLIP, kept alive
    for the whole build. One join against the loaded ``records`` table does the same work in bulk.
    A name that matches nothing lands as a record item with a null ``record_id``, which
    ``schema.VALIDATIONS`` rejects.

    Args:
        connection: The connection holding the loaded, not yet committed database.
    """
    connection.execute(
        "INSERT INTO task_items (task_id, role, position, item_type, text_value, record_id) "  # noqa: S608
        "SELECT s.task_id, s.role, s.position, s.item_type, s.text_value, r.record_id "
        f"FROM {_STAGED_ITEMS} s LEFT JOIN records r ON r.external_id = s.record_external_id"
    )
    connection.execute(f"DROP TABLE {_STAGED_ITEMS}")


def _pending(signal: Signal, signal_id: int) -> PendingSignal:
    """Describe one signal for the values plane.

    Args:
        signal: The signal to describe.
        signal_id: The id the control plane assigned it.

    Returns:
        What the values plane needs to write it.
    """
    return PendingSignal(
        signal_id=signal_id,
        name=signal.name,
        spec_type=signal.spec.spec_type,
        dtype=signal.spec.dtype,
        values=signal.values,
        time_offsets_us=signal.time_offsets_us,
    )


def _add_chunk(loader: "_Loader", locator: ChunkLocator) -> None:
    """Record where one chunk of values landed.

    Args:
        loader: The loader feeding the tables.
        locator: The chunk's location in the values plane.
    """
    loader.chunks.add(
        (
            locator.signal_id,
            locator.chunk_idx,
            loader.artifact(locator.chunk_file),
            locator.chunk_major_idx,
            locator.chunk_minor_idx,
            locator.n_values,
        )
    )


class _Loader:
    """Walks the hierarchy once, assigns every id, and feeds every table's inserter."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.records = _BatchInserter(connection, "records", ("record_id", "external_id", "start_time_us"))
        self.axes = _BatchInserter(
            connection,
            "axes",
            ("axis_id", "axis_type", "period_numerator_us", "period_denominator", "start_index", "first_us", "last_us"),
        )
        self.specs = _BatchInserter(connection, "specs", ("spec_id", "spec_type", "name", "unit", "dtype", "nullable"))
        self.sources = _BatchInserter(
            connection,
            "sources",
            ("source_id", "external_id", "record_id", "parent_source_id", "path", "depth", "name", "position"),
        )
        self.signals = _BatchInserter(
            connection, "signals", ("signal_id", "external_id", "name", "axis_id", "spec_id", "n_values")
        )
        self.source_signals = _BatchInserter(connection, "source_signals", ("source_id", "signal_id", "position"))
        self.artifacts = _BatchInserter(connection, "values_artifacts", ("artifact_id", "chunk_file", "backend"))
        self.chunks = _BatchInserter(
            connection,
            "signal_chunks",
            ("signal_id", "chunk_idx", "artifact_id", "chunk_major_idx", "chunk_minor_idx", "n_values"),
        )
        self.tasks = _BatchInserter(connection, "tasks", ("task_id", "external_id", "prompt"))
        self.staged_items = _BatchInserter(
            connection,
            _STAGED_ITEMS,
            ("task_id", "role", "position", "item_type", "text_value", "record_external_id"),
        )
        self.contents = _BatchInserter(connection, "annotations", ("annotation_id", "name", "value", "unit"))
        self.meta = _BatchInserter(connection, "meta", ("key", "value"))
        self._attachments = {
            kind: _AttachmentTable(connection, kind, table, target)
            for kind, (table, target) in ddl.ANNOTATION_TABLES.items()
        }

        self.record_ids = _Ids("records")
        self.source_ids = _Ids("sources")
        self.task_ids = _Ids("tasks")
        self._signal_ids = _Ids("signals")
        self._axis_ids = _Ids("axes")
        self._spec_ids = _Ids("specs")
        self._annotation_ids = _Ids("annotations")
        self._artifact_ids = _Ids("values artifacts")

        # A signal object can hang off sources in several records, and the second source still needs
        # its id for the link row, so the map keeps the id rather than just the fact it was seen.
        self._signal_id_of: dict[str, int] = {}
        self._annotation_id_of: dict[tuple[str, str, str | None], int] = {}
        self._axis_id_of: dict[tuple[Any, ...], int] = {}
        self._spec_id_of: dict[tuple[Any, ...], int] = {}
        self._artifact_id_of: dict[str, int] = {}

    def all(self) -> tuple[_BatchInserter, ...]:
        """Every inserter, so the caller can flush them all at the end.

        Returns:
            Each table's inserter. Order does not matter: the shipped database declares no foreign
            keys, and the invariants are checked once the whole load is in.
        """
        return (
            self.meta,
            self.records,
            self.axes,
            self.specs,
            self.sources,
            self.signals,
            self.source_signals,
            self.artifacts,
            self.chunks,
            self.tasks,
            self.staged_items,
            self.contents,
            *(table.inserter for table in self._attachments.values()),
        )

    @property
    def attachment_count(self) -> int:
        """How many annotations were attached to something, summed over the five target tables."""
        return sum(table.inserter.count for table in self._attachments.values())

    def artifact(self, chunk_file: str) -> int:
        """Return the id of one values artifact, assigning it the first time the file is named.

        The values plane keeps emitting path strings and knows nothing about this id. One dict
        entry per shard is small enough to hold for the whole build.

        Args:
            chunk_file: The artifact's version-relative path.

        Returns:
            The artifact id.
        """
        artifact_id = self._artifact_id_of.get(chunk_file)
        if artifact_id is None:
            artifact_id = self._artifact_id_of[chunk_file] = self._artifact_ids.claim()
        return artifact_id

    def annotate(
        self,
        annotations: list[Annotation],
        object_type: str,
        object_id: int | None,
        *,
        scope_record_id: int | None = None,
    ) -> None:
        """Record one object's annotations: the payload once, an attachment every time.

        The attachment goes into the table that holds this kind of target, so nothing stores a
        discriminator column. A source's attachment also carries the record it sits in, so the
        renderer can pull every annotation in a record with one equality.

        Args:
            annotations: The annotations attached to the object.
            object_type: The object's kind, a key of
                :data:`~timenet.control_plane.schema.ANNOTATION_TABLES`.
            object_id: The object's id, or ``None`` for the dataset itself.
            scope_record_id: The record the object sits in, for a source.
        """
        table = self._attachments[object_type]
        for annotation in annotations:
            table.add(self._annotation(annotation), annotation, object_id, scope_record_id)

    def _annotation(self, annotation: Annotation) -> int:
        """Store an annotation's payload the first time this statement is made, and return its id.

        Args:
            annotation: The annotation whose payload to store.

        Returns:
            The annotation id.
        """
        key = annotation.content_key()
        annotation_id = self._annotation_id_of.get(key)
        if annotation_id is None:
            annotation_id = self._annotation_id_of[key] = self._annotation_ids.claim()
            self.contents.add((annotation_id, *key))
        return annotation_id

    def signal(self, signal: Signal) -> tuple[int, bool]:
        """Store a signal's own row the first time it is seen, and return its id.

        Args:
            signal: The signal.

        Returns:
            The signal's id, and whether this call was the one that stored it.
        """
        signal_id = self._signal_id_of.get(signal.id)
        if signal_id is not None:
            return signal_id, False
        signal_id = self._signal_id_of[signal.id] = self._signal_ids.claim()
        self.signals.add((signal_id, signal.id, signal.name, self.axis(signal), self.spec(signal), len(signal.values)))
        # A shared signal belongs to no single record, so its annotations carry no record scope;
        # the reader reaches them through source_signals instead.
        self.annotate(signal.annotations, "signal", signal_id)
        return signal_id, True

    def axis(self, signal: Signal) -> int:
        """Record a signal's axis if this is the first signal to use it, and return its id.

        Args:
            signal: The signal whose axis to record.

        Returns:
            The axis id.
        """
        columns = axis_columns(signal.time_axis)
        key: tuple[Any, ...] = (
            str(signal.time_axis.axis_type),
            columns["period_numerator_us"],
            columns["period_denominator"],
            columns["start_index"],
            columns["first_us"],
            columns["last_us"],
        )
        axis_id = self._axis_id_of.get(key)
        if axis_id is None:
            axis_id = self._axis_id_of[key] = self._axis_ids.claim()
            self.axes.add((axis_id, *key))
        return axis_id

    def spec(self, signal: Signal) -> int:
        """Record a signal's spec if this is the first signal to use it, and return its id.

        Args:
            signal: The signal whose spec to record.

        Returns:
            The spec id.
        """
        spec = signal.spec
        key: tuple[Any, ...] = (spec.spec_type, spec.name, str(spec.unit_value), spec.dtype, spec.nullable)
        spec_id = self._spec_id_of.get(key)
        if spec_id is None:
            spec_id = self._spec_id_of[key] = self._spec_ids.claim()
            self.specs.add((spec_id, *key))
        return spec_id


def _load_record(loader: _Loader, record: Record) -> list[tuple[Signal, int]]:
    """Insert one record, its source tree, and its signals.

    Args:
        loader: The loader feeding the tables.
        record: The record to insert.

    Returns:
        Every signal this record stored for the first time, with the id it was given. A signal
        already stored under another record is left out, so the caller writes its values once.
    """
    record_id = loader.record_ids.claim()
    loader.records.add((record_id, record.id, record.start_time_us))
    loader.annotate(record.annotations, "record", record_id, scope_record_id=record_id)
    fresh: list[tuple[Signal, int]] = []
    # Breadth-first, so a parent row is always inserted before the child that references it. Each
    # source's path is its parent's path plus its own position, which makes the whole ancestry a
    # prefix and display order a plain sort.
    queue: list[tuple[Source, int | None, str | None, int]] = [
        (root, None, None, index) for index, root in enumerate(record.sources)
    ]
    while queue:
        source, parent_id, parent_path, position = queue.pop(0)
        source_id = loader.source_ids.claim()
        path = ddl.source_path(parent_path, position)
        depth = path.count(".")
        loader.sources.add((source_id, source.id, record_id, parent_id, path, depth, source.name, position))
        loader.annotate(source.annotations, "source", source_id, scope_record_id=record_id)
        for index, signal in enumerate(source.signals):
            stored = _load_signal(loader, signal, source_id=source_id, position=index)
            if stored is not None:
                fresh.append(stored)
        queue.extend((child, source_id, path, index) for index, child in enumerate(source.sources))
    return fresh


def _load_signal(loader: _Loader, signal: Signal, *, source_id: int, position: int) -> tuple[Signal, int] | None:
    """Link one signal to a source, inserting the signal itself the first time it is seen.

    The same series can hang off sources in several records, so the signal row and its annotations
    are written once and the link table records each use.

    Args:
        loader: The loader feeding the tables.
        signal: The signal to link.
        source_id: The source the signal hangs off.
        position: The signal's position among that source's signals.

    Returns:
        The signal and its id when this call stored it, and ``None`` when it was already stored.
    """
    signal_id, stored = loader.signal(signal)
    loader.source_signals.add((source_id, signal_id, position))
    return (signal, signal_id) if stored else None


def _load_task(loader: _Loader, task: Task) -> None:
    """Insert one task and its ordered inputs and targets.

    A record item is staged under the caller's own record id and resolved to a record id by one
    join once the load finishes.

    Args:
        loader: The loader feeding the tables.
        task: The task to insert.
    """
    task_id = loader.task_ids.claim()
    loader.tasks.add((task_id, task.id, task.prompt))
    loader.annotate(task.annotations, "task", task_id)
    for role, items in (("input", task.inputs), ("target", task.target)):
        for position, item in enumerate(items):
            if isinstance(item, Record | RecordRef):
                loader.staged_items.add((task_id, role, position, "record", None, item.id))
            else:
                loader.staged_items.add((task_id, role, position, "text", item, None))
