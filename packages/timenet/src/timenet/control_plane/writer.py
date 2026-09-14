"""Compile a declarative hierarchy into a dataset version: one control database plus Parquet shards.

The writer walks ``Task -> Record -> Source -> Signal``, derives every id the storage layer needs
(annotation content ids, occurrence ids, axis ids), streams the values into the Parquet values
plane, and inserts the structure into an embedded DuckDB database.

Commit works the way the current writer's does: everything is built in a ``<version>.tmp-<uuid>``
staging directory, and the finished directory is moved into place with one atomic rename. A reader
either sees a complete version or sees nothing.
"""

from collections.abc import Iterable
import hashlib
import json
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

# DuckDB's declared column types, mapped to the Arrow types a batch is built with.
_ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "BIGINT": pa.int64(),
    "INTEGER": pa.int32(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
}

# DuckDB allocates storage a block at a time, and every table and index segment claims at least one
# block, so the default 256 KiB block sets a floor of a few megabytes on a database that holds a few
# hundred rows. A dataset's control plane is small and is downloaded before it is read, so the floor
# matters more than the throughput a larger block buys. Measured on the demo dataset: 12.9 MB at the
# default, 815 KB at 16 KiB. 16 KiB is DuckDB's minimum.
_BLOCK_SIZE = 16_384


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


def annotation_id(annotation: Annotation) -> str:
    """Return the id of an annotation's reusable payload.

    The id is derived from the payload itself, so the same statement made about ten thousand records
    resolves to one stored row without the writer keeping a lookup table.

    Args:
        annotation: The annotation whose payload to identify.

    Returns:
        The content id.
    """
    name, value, unit = annotation.content_key()
    digest = hashlib.sha256("\x00".join([name, value, unit or ""]).encode()).hexdigest()
    return f"ac-{digest[:24]}"


def axis_id(axis: Any) -> str:
    """Return the id of a time axis, derived from the axis itself so equal axes share one row.

    Args:
        axis: The axis object.

    Returns:
        The axis id.
    """
    axis_type = axis.axis_type
    columns = axis_columns(axis)
    parts = [str(axis_type), *(str(columns[key]) for key in sorted(columns))]
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
    return f"ax-{digest[:20]}"


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

        A record must be yielded before any task that refers to it, and every record a task names
        must be yielded at some point, or the write-time validation will reject the dangling
        reference.

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
            counts = _stream_into(connection, values, records, tasks, annotations)
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
                registered_annotations=counts["entities_to_annotations"],
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


def _stream_into(
    connection: duckdb.DuckDBPyConnection,
    values: ValuesWriter,
    records: Iterable[Record],
    tasks: Iterable[Task],
    annotations: Iterable[Annotation],
) -> dict[str, int]:
    """Walk the records and tasks once, feeding both planes as it goes.

    Args:
        connection: The open connection to the database being built.
        values: The values-plane writer to stream signal values into.
        records: The records to write.
        tasks: The tasks to write.
        annotations: Dataset-level annotations.

    Returns:
        The row count of each loaded table.
    """
    loader = _Loader(connection)
    dataset_id = "dataset"
    loader.meta.add(("schema_version", str(ddl.SCHEMA_VERSION)))
    loader.meta.add(("dataset_id", dataset_id))
    loader.datasets.add((dataset_id, None))
    loader.annotate(list(annotations), "dataset", dataset_id)

    written_values: set[str] = set()
    for record in records:
        _load_record(loader, record)
        for signal in record.signals():
            # A series shared by several records is written once.
            if signal.id in written_values:
                continue
            written_values.add(signal.id)
            for locator in values.add(_pending(signal)):
                _add_chunk(loader, locator)
    for task in tasks:
        _load_task(loader, task)
    for locator in values.finish():
        _add_chunk(loader, locator)
    # Declared after the values plane closes, so a chunk row can be anti-joined against the
    # artifacts that really exist rather than against the ones the writer meant to create.
    for chunk_file, backend in values.artifacts:
        loader.artifacts.add((chunk_file, backend))

    for inserter in loader.all():
        inserter.flush()
    return {
        "records": loader.records.count,
        "sources": loader.sources.count,
        "signals": loader.signals.count,
        "source_signals": loader.source_signals.count,
        "values_artifacts": loader.artifacts.count,
        "signal_chunks": loader.chunks.count,
        "tasks": loader.tasks.count,
        "annotations": loader.contents.count,
        "entities_to_annotations": loader.occurrences.count,
    }


def _pending(signal: Signal) -> PendingSignal:
    """Describe one signal for the values plane.

    Args:
        signal: The signal to describe.

    Returns:
        What the values plane needs to write it.
    """
    return PendingSignal(
        signal_id=signal.id,
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
            locator.chunk_file,
            locator.chunk_major_idx,
            locator.chunk_minor_idx,
            locator.n_values,
        )
    )


def _collect_signals(dataset: DeclarativeDataset) -> list[PendingSignal]:
    """Gather every distinct signal in the dataset, with the facts the values plane needs.

    A series used by several records is written once. Without this the values plane would store the
    same waveform once per referencing record.

    Returns:
        One entry per distinct signal id, in record then source then signal order.
    """
    pending: list[PendingSignal] = []
    seen: set[str] = set()
    for record in dataset.records:
        for signal in record.signals():
            if signal.id in seen:
                continue
            seen.add(signal.id)
            pending.append(
                PendingSignal(
                    signal_id=signal.id,
                    name=signal.name,
                    spec_type=signal.spec.spec_type,
                    dtype=signal.spec.dtype,
                    values=signal.values,
                    time_offsets_us=signal.time_offsets_us,
                )
            )
    return pending


def _json_or_none(value: dict[str, Any]) -> str | None:
    """Serialize a metadata dict, or return ``None`` when it is empty.

    Returns:
        The JSON text, or ``None`` for an empty dict.
    """
    return json.dumps(value, sort_keys=True) if value else None


class _Loader:
    """Walks the hierarchy once and feeds every table's inserter."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.records = _BatchInserter(connection, "records", ("record_id", "start_time_us", "metadata"))
        self.axes = _BatchInserter(
            connection,
            "axes",
            ("axis_id", "axis_type", "period_numerator_us", "period_denominator", "start_index", "first_us", "last_us"),
        )
        self.specs = _BatchInserter(connection, "specs", ("spec_id", "name", "unit", "dtype", "nullable"))
        self.sources = _BatchInserter(
            connection,
            "sources",
            ("source_id", "record_id", "parent_source_id", "path", "depth", "name", "position", "metadata"),
        )
        self.signals = _BatchInserter(
            connection, "signals", ("signal_id", "name", "axis_id", "spec_id", "n_values", "metadata")
        )
        self.source_signals = _BatchInserter(connection, "source_signals", ("source_id", "signal_id", "position"))
        self.artifacts = _BatchInserter(connection, "values_artifacts", ("chunk_file", "backend"))
        self.chunks = _BatchInserter(
            connection,
            "signal_chunks",
            ("signal_id", "chunk_idx", "chunk_file", "chunk_major_idx", "chunk_minor_idx", "n_values"),
        )
        self.tasks = _BatchInserter(connection, "tasks", ("task_id", "prompt", "metadata"))
        self.items = _BatchInserter(
            connection, "task_items", ("task_id", "role", "position", "item_type", "text_value", "record_id")
        )
        self.contents = _BatchInserter(connection, "annotations", ("content_id", "name", "value", "unit", "metadata"))
        self.occurrences = _BatchInserter(
            connection,
            "entities_to_annotations",
            (
                "occurrence_id",
                "content_id",
                "object_type",
                "on_dataset_id",
                "on_task_id",
                "on_record_id",
                "on_source_id",
                "on_signal_id",
                "scope_record_id",
                "span_type",
                "start_us",
                "end_us",
                "provenance",
                "confidence",
                "metadata",
            ),
        )
        self.datasets = _BatchInserter(connection, "datasets", ("dataset_id", "metadata"))
        self.meta = _BatchInserter(connection, "meta", ("key", "value"))
        self.seen_signals: set[str] = set()
        self._seen_contents: set[str] = set()
        self._seen_axes: set[str] = set()
        self._seen_specs: set[str] = set()
        self._occurrence_count = 0

    def all(self) -> tuple[_BatchInserter, ...]:
        """Every inserter, so the caller can flush them all at the end.

        Returns:
            Each table's inserter. Order does not matter: the shipped database declares no foreign
            keys, and the invariants are checked once the whole load is in.
        """
        return (
            self.meta,
            self.datasets,
            self.records,
            self.axes,
            self.specs,
            self.sources,
            self.signals,
            self.source_signals,
            self.artifacts,
            self.chunks,
            self.tasks,
            self.items,
            self.contents,
            self.occurrences,
        )

    def annotate(
        self, annotations: list[Annotation], object_type: str, object_id: str, *, scope_record_id: str | None = None
    ) -> None:
        """Record one object's annotations: the payload once, the occurrence every time.

        The occurrence carries the target in the one foreign key column that matches its kind, and
        carries ``scope_record_id`` when the target sits inside a record, so the renderer can pull
        every annotation in a record with one equality.
        """
        column = ddl.TARGET_COLUMNS[object_type]
        for annotation in annotations:
            identifier = annotation_id(annotation)
            if identifier not in self._seen_contents:
                self._seen_contents.add(identifier)
                self.contents.add(
                    (
                        identifier,
                        annotation.name,
                        annotation.value,
                        annotation.unit,
                        _json_or_none(annotation.metadata),
                    )
                )
            targets = dict.fromkeys(ddl.TARGET_COLUMNS.values())
            targets[column] = object_id
            self._occurrence_count += 1
            self.occurrences.add(
                (
                    self._occurrence_count,
                    identifier,
                    object_type,
                    targets["on_dataset_id"],
                    targets["on_task_id"],
                    targets["on_record_id"],
                    targets["on_source_id"],
                    targets["on_signal_id"],
                    scope_record_id,
                    annotation.span_type,
                    annotation.start_us,
                    annotation.end_us,
                    annotation.provenance,
                    annotation.confidence,
                    _json_or_none(annotation.metadata),
                )
            )

    def axis(self, signal: Signal) -> str:
        """Record a signal's axis if this is the first signal to use it, and return its id.

        Returns:
            The axis id.
        """
        identifier = axis_id(signal.time_axis)
        if identifier not in self._seen_axes:
            self._seen_axes.add(identifier)
            columns = axis_columns(signal.time_axis)
            self.axes.add(
                (
                    identifier,
                    str(signal.time_axis.axis_type),
                    columns["period_numerator_us"],
                    columns["period_denominator"],
                    columns["start_index"],
                    columns["first_us"],
                    columns["last_us"],
                )
            )
        return identifier

    def spec(self, signal: Signal) -> str:
        """Record a signal's spec if this is the first signal to use it, and return its id.

        Returns:
            The spec id.
        """
        spec = signal.spec
        if spec.spec_type not in self._seen_specs:
            self._seen_specs.add(spec.spec_type)
            self.specs.add((spec.spec_type, spec.name, str(spec.unit_value), spec.dtype, spec.nullable))
        return spec.spec_type


def _load(
    connection: duckdb.DuckDBPyConnection, dataset: DeclarativeDataset, locators: list[ChunkLocator]
) -> dict[str, int]:
    """Insert the whole hierarchy, parents before children so every foreign key resolves.

    Returns:
        The row count of each loaded table.
    """
    loader = _Loader(connection)
    dataset_id = dataset_id_of(dataset)
    loader.meta.add(("schema_version", str(ddl.SCHEMA_VERSION)))
    loader.meta.add(("dataset_id", dataset_id))
    loader.datasets.add((dataset_id, None))
    loader.datasets.flush()

    loader.annotate(dataset.annotations, "dataset", dataset_id)

    for record in dataset.records:
        _load_record(loader, record)
    for task in dataset.tasks:
        _load_task(loader, task)
    for locator in locators:
        loader.chunks.add(
            (
                locator.signal_id,
                locator.chunk_idx,
                locator.chunk_file,
                locator.chunk_major_idx,
                locator.chunk_minor_idx,
                locator.n_values,
            )
        )

    for inserter in loader.all():
        inserter.flush()
    return {
        "records": loader.records.count,
        "sources": loader.sources.count,
        "signals": loader.signals.count,
        "source_signals": loader.source_signals.count,
        "values_artifacts": loader.artifacts.count,
        "signal_chunks": loader.chunks.count,
        "tasks": loader.tasks.count,
        "annotations": loader.contents.count,
        "entities_to_annotations": loader.occurrences.count,
    }


def dataset_id_of(dataset: DeclarativeDataset) -> str:
    """Return a stable id for the dataset object itself, for dataset-level annotations.

    Args:
        dataset: The dataset.

    Returns:
        The literal ``dataset``. A version holds exactly one dataset, so it needs no further
        qualification inside its own database.
    """
    del dataset
    return "dataset"


def _load_record(loader: _Loader, record: Record) -> None:
    """Insert one record, its source tree, and its signals."""
    loader.records.add((record.id, record.start_time_us, _json_or_none(record.metadata)))
    loader.annotate(record.annotations, "record", record.id, scope_record_id=record.id)
    # Breadth-first, so a parent row is always inserted before the child that references it. Each
    # source's path is its parent's path plus its own position, which makes the whole ancestry a
    # prefix and display order a plain sort.
    queue: list[tuple[Source, str | None, str | None, int]] = [
        (root, None, None, index) for index, root in enumerate(record.sources)
    ]
    while queue:
        source, parent_id, parent_path, position = queue.pop(0)
        path = ddl.source_path(parent_path, position)
        depth = path.count(".")
        loader.sources.add(
            (source.id, record.id, parent_id, path, depth, source.name, position, _json_or_none(source.metadata))
        )
        loader.annotate(source.annotations, "source", source.id, scope_record_id=record.id)
        for index, signal in enumerate(source.signals):
            _load_signal(loader, signal, source_id=source.id, position=index)
        queue.extend((child, source.id, path, index) for index, child in enumerate(source.sources))


def _load_signal(loader: _Loader, signal: Signal, *, source_id: str, position: int) -> None:
    """Link one signal to a source, inserting the signal itself the first time it is seen.

    The same series can hang off sources in several records, so the signal row and its annotations
    are written once and the link table records each use.
    """
    if signal.id not in loader.seen_signals:
        loader.seen_signals.add(signal.id)
        loader.signals.add(
            (
                signal.id,
                signal.name,
                loader.axis(signal),
                loader.spec(signal),
                len(signal.values),
                _json_or_none(signal.metadata),
            )
        )
        # A shared signal belongs to no single record, so its annotations carry no record scope;
        # the reader reaches them through source_signals instead.
        loader.annotate(signal.annotations, "signal", signal.id)
    loader.source_signals.add((source_id, signal.id, position))


def _load_task(loader: _Loader, task: Task) -> None:
    """Insert one task and its ordered inputs and targets."""
    loader.tasks.add((task.id, task.prompt, _json_or_none(task.metadata)))
    loader.annotate(task.annotations, "task", task.id)
    for role, items in (("input", task.inputs), ("target", task.target)):
        for position, item in enumerate(items):
            if isinstance(item, Record | RecordRef):
                loader.items.add((task.id, role, position, "record", None, item.id))
            else:
                loader.items.add((task.id, role, position, "text", item, None))
