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

from timenet.control_plane import schema as ddl
from timenet.control_plane.model import Annotation, DeclarativeDataset, Record, Signal, Source, Task
from timenet.control_plane.values import ChunkLocator, PendingSignal, ValuesPlaneWriter, axis_columns
from timenet.errors import TimeFValidationError
from timenet.format.checksums import file_checksum
from timenet.format.constants import CONTROL_DB_FILE, MANIFEST_FILE
from timenet.manifest import Manifest
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import FilePart, ManifestFiles
from timenet.types import DatasetMetadata


# Rows handed to DuckDB per executemany call. The hierarchy is walked as a stream and flushed in
# batches, so a build never holds every row of a table in memory at once.
_INSERT_BATCH = 10_000

# DuckDB allocates storage a block at a time, and every table and index segment claims at least one
# block, so the default 256 KiB block sets a floor of a few megabytes on a database that holds a few
# hundred rows. A dataset's control plane is small and is downloaded before it is read, so the floor
# matters more than the throughput a larger block buys. Measured on the demo dataset: 12.9 MB at the
# default, 815 KB at 16 KiB. 16 KiB is DuckDB's minimum.
_BLOCK_SIZE = 16_384


class _BatchInserter:
    """Buffers rows for one table and flushes them to DuckDB in batches.

    Foreign keys are checked as each row goes in, and DuckDB has no deferred constraints, so a batch
    must never reach the database before the rows it references. Each inserter therefore knows which
    inserters it depends on and flushes them first. That keeps the build streaming: the walk can
    interleave records, sources, and signals freely without a batch boundary breaking a key.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]) -> None:
        """Bind the inserter to its table and column list.

        Args:
            connection: The open database connection.
            table: The table to insert into.
            columns: The column names, in the order the rows supply them.
        """
        self._connection = connection
        # Table and column names come from this module's own schema, never from input; every
        # value is bound as a parameter.
        self._sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"  # noqa: S608
        self._rows: list[tuple] = []
        self._depends_on: tuple[_BatchInserter, ...] = ()
        self.table = table
        self.count = 0

    def depends_on(self, *inserters: "_BatchInserter") -> "_BatchInserter":
        """Declare the tables this one's foreign keys point at.

        Args:
            inserters: The inserters to flush before this one writes.

        Returns:
            This inserter, so the declaration can be chained onto construction.
        """
        self._depends_on = tuple(other for other in inserters if other is not self)
        return self

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
        """Write any buffered rows, after flushing everything they reference."""
        if not self._rows:
            return
        for dependency in self._depends_on:
            dependency.flush()
        self._connection.executemany(self._sql, self._rows)
        self._rows.clear()


def content_id(annotation: Annotation) -> str:
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
    """Writes one dataset version: a DuckDB control plane plus a Parquet values plane."""

    def __init__(self, root: Path, metadata: DatasetMetadata, **values_options: Any) -> None:
        """Bind the writer to its output location and the version's metadata.

        Args:
            root: The registry root. The version lands at ``<root>/<dataset_id>/<version>``.
            metadata: The dataset's metadata, including its id and version.
            values_options: Byte budgets forwarded to :class:`~timenet.control_plane.values.ValuesPlaneWriter`.

        Raises:
            TimeFValidationError: If the version is already committed at this location.
        """
        self._metadata = metadata
        self._final_dir = Path(root) / metadata.dataset_id / str(metadata.dataset_version)
        if (self._final_dir / MANIFEST_FILE).exists():
            raise TimeFValidationError(f"{self._final_dir} already holds a committed version")
        self._staging_dir = self._final_dir.parent / f"{self._final_dir.name}.tmp-{uuid.uuid4().hex}"
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
        """Write the whole dataset and commit it atomically.

        Args:
            dataset: The declarative hierarchy to compile.

        Returns:
            The manifest of the committed version.
        """
        pending = _collect_signals(dataset)
        values = ValuesPlaneWriter(self._staging_dir, **self._values_options)
        locators = values.write(pending)

        db_path = self._staging_dir / CONTROL_DB_FILE
        counts = self._build_database(db_path, dataset, locators)

        self._manifest = self._commit(db_path, values, counts)
        return self._manifest

    @staticmethod
    def _build_database(db_path: Path, dataset: DeclarativeDataset, locators: list[ChunkLocator]) -> dict[str, int]:
        """Create the control database and load the hierarchy into it.

        Returns:
            The row count of each loaded table.
        """
        # The block size can only be set when the database is created, so the file is attached
        # rather than opened directly.
        connection = duckdb.connect()
        try:
            connection.execute(f"ATTACH '{db_path}' AS control (BLOCK_SIZE {_BLOCK_SIZE})")
            connection.execute("USE control")
            connection.execute("BEGIN TRANSACTION")
            for statement in _statements(ddl.DDL):
                connection.execute(statement)
            counts = _load(connection, dataset, locators)
            connection.execute("COMMIT")
            # Indexes are built after the load: maintaining them row by row during the insert is
            # slower, and nothing reads the database until it is committed anyway.
            for statement in _statements(ddl.INDEXES):
                connection.execute(statement)
            connection.execute("CHECKPOINT control")
        finally:
            connection.close()
        return counts

    def _commit(self, db_path: Path, values: ValuesPlaneWriter, counts: dict[str, int]) -> Manifest:
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
                annotations=counts["annotation_contents"],
                registered_annotations=counts["annotation_occurrences"],
                time_series_chunks=counts["signal_chunks"],
                time_series_index_rows=counts["signal_chunks"],
            ),
            value_encoding=values.value_encoding,
        )
        (self._staging_dir / MANIFEST_FILE).write_text(manifest.to_json())
        if not db_path.exists():
            raise TimeFValidationError(f"control database missing at {db_path}")
        if self._final_dir.exists():
            shutil.rmtree(self._final_dir)
        self._staging_dir.replace(self._final_dir)
        return manifest


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


def _collect_signals(dataset: DeclarativeDataset) -> list[PendingSignal]:
    """Gather every signal in the dataset, with the facts the values plane needs.

    Returns:
        One entry per signal, in record then source then signal order.
    """
    pending: list[PendingSignal] = []
    for record in dataset.records:
        for signal in record.signals():
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
            connection,
            "signals",
            ("signal_id", "source_id", "record_id", "name", "position", "axis_id", "spec_id", "n_values", "metadata"),
        )
        self.chunks = _BatchInserter(
            connection, "signal_chunks", ("signal_id", "chunk_idx", "chunk_file", "row_group", "row_offset", "n_values")
        )
        self.tasks = _BatchInserter(connection, "tasks", ("task_id", "prompt", "metadata"))
        self.items = _BatchInserter(
            connection, "task_items", ("task_id", "role", "position", "item_type", "text_value", "record_id")
        )
        self.contents = _BatchInserter(
            connection, "annotation_contents", ("content_id", "name", "value", "unit", "metadata")
        )
        self.occurrences = _BatchInserter(
            connection,
            "annotation_occurrences",
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
        self._seen_contents: set[str] = set()
        self._seen_axes: set[str] = set()
        self._seen_specs: set[str] = set()
        self._occurrence_count = 0

        self.sources.depends_on(self.records)
        self.signals.depends_on(self.sources, self.records, self.axes, self.specs)
        self.chunks.depends_on(self.signals)
        self.items.depends_on(self.tasks, self.records)
        self.occurrences.depends_on(self.contents, self.datasets, self.tasks, self.records, self.sources, self.signals)

    def all(self) -> tuple[_BatchInserter, ...]:
        """Every inserter, so the caller can flush them in dependency order.

        Returns:
            Each table's inserter.
        """
        return (
            self.meta,
            self.datasets,
            self.records,
            self.axes,
            self.specs,
            self.sources,
            self.signals,
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
            identifier = content_id(annotation)
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
                locator.row_group,
                locator.row_offset,
                locator.n_values,
            )
        )

    for inserter in loader.all():
        inserter.flush()
    return {
        "records": loader.records.count,
        "sources": loader.sources.count,
        "signals": loader.signals.count,
        "signal_chunks": loader.chunks.count,
        "tasks": loader.tasks.count,
        "annotation_contents": loader.contents.count,
        "annotation_occurrences": loader.occurrences.count,
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
            _load_signal(loader, signal, source_id=source.id, record_id=record.id, position=index)
        queue.extend((child, source.id, path, index) for index, child in enumerate(source.sources))


def _load_signal(loader: _Loader, signal: Signal, *, source_id: str, record_id: str, position: int) -> None:
    """Insert one signal, with its axis and spec recorded once each."""
    axis = loader.axis(signal)
    spec = loader.spec(signal)
    loader.signals.add(
        (
            signal.id,
            source_id,
            record_id,
            signal.name,
            position,
            axis,
            spec,
            len(signal.values),
            _json_or_none(signal.metadata),
        )
    )
    loader.annotate(signal.annotations, "signal", signal.id, scope_record_id=record_id)


def _load_task(loader: _Loader, task: Task) -> None:
    """Insert one task and its ordered inputs and targets."""
    loader.tasks.add((task.id, task.prompt, _json_or_none(task.metadata)))
    loader.annotate(task.annotations, "task", task.id)
    for role, items in (("input", task.inputs), ("target", task.target)):
        for position, item in enumerate(items):
            if isinstance(item, Record):
                loader.items.add((task.id, role, position, "record", None, item.id))
            else:
                loader.items.add((task.id, role, position, "text", item, None))
