"""Read a version's control database: one query per table per batch, never one query per record.

Every call here answers for a whole batch of records or for the whole version. Hydrating a record at
a time costs 34 ms on a 618,508-record corpus, because each query scans its table; asking for a
thousand records at once costs 0.111 ms per record, because the same scan answers all of them.

The database opens read-only, so no write-ahead log appears beside a file the manifest has already
checksummed. A version whose files live on an object store is copied to a local temporary file
first: DuckDB reads a database through its own filesystem layer, not through the pyarrow one the
rest of the reader uses.
"""

from collections.abc import Iterator, Sequence
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Final

import duckdb
import pyarrow.fs as pafs

from timenet.control_plane import schema as ddl
from timenet.errors import TimeFFormatError
from timenet.format.constants import CONTROL_DB_FILE


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


_SPECS: Final = """
SELECT spec_type, name, unit_value, data_source_type, data_source_name, data_source_provider,
       dtype, categories, value_shape, dimension_names, nullable
FROM specs ORDER BY spec_id
"""

_ANNOTATION_DESCRIPTORS: Final = """
SELECT key, annotation_type, value_type, unit, description FROM annotation_descriptors ORDER BY descriptor_id
"""

_RECORDS: Final = """
SELECT record_id, external_id, start_time_us, time_span_start_us, time_span_end_us, subject_ids
FROM records WHERE record_id IN (SELECT unnest(?)) ORDER BY record_id
"""

_RECORD_TASKS: Final = """
SELECT l.record_id, t.external_id
FROM record_tasks l
JOIN tasks t ON t.task_id = l.task_id
WHERE l.record_id IN (SELECT unnest(?))
ORDER BY l.record_id, l.task_id
"""

_RECORD_SERIES: Final = """
SELECT l.record_id, s.external_id, s.signal, s.source_id, s.n_values, sp.spec_type,
       a.axis_type, a.period_numerator_us, a.period_denominator, a.start_index, a.first_us, a.last_us
FROM record_time_series l
JOIN time_series s ON s.time_series_id = l.time_series_id
JOIN specs sp ON sp.spec_id = s.spec_id
JOIN axes a ON a.axis_id = s.axis_id
WHERE l.record_id IN (SELECT unnest(?))
ORDER BY l.record_id, l.position
"""

_RECORD_ANNOTATIONS: Final = """
SELECT t.record_id, n.external_id, n.key, n.value, n.source,
       n.span_start_us, n.span_end_us, n.span_time_series_ids
FROM record_annotations t
JOIN annotations n ON n.annotation_id = t.annotation_id
WHERE t.record_id IN (SELECT unnest(?))
ORDER BY t.record_id, t.position
"""

_REGISTERED_ANNOTATIONS: Final = """
SELECT n.external_id, n.key, n.value, n.source, n.span_start_us, n.span_end_us, n.span_time_series_ids
FROM dataset_annotations t
JOIN annotations n ON n.annotation_id = t.annotation_id
ORDER BY t.position
"""

# A series shared by several records is stored once, so its chunks are reached through the link
# table. The filter is on the record's dense surrogate, which is also the order the link table was
# written in, so the scan prunes on its zone map instead of comparing the caller's id string on
# every row. Ordering by the record, then by the series' own id, then by chunk keeps a series'
# chunks contiguous and in the order the values plane concatenates them.
_RECORD_CHUNKS: Final = """
SELECT l.record_id, s.external_id AS time_series_id, sp.spec_type, c.chunk_idx, v.chunk_file,
       c.chunk_major_idx, c.chunk_minor_idx, c.n_values
FROM record_time_series l
JOIN time_series s ON s.time_series_id = l.time_series_id
JOIN specs sp ON sp.spec_id = s.spec_id
JOIN time_series_chunks c ON c.time_series_id = s.time_series_id
JOIN values_artifacts v ON v.artifact_id = c.artifact_id
WHERE l.record_id IN (SELECT unnest(?))
ORDER BY l.record_id, s.time_series_id, c.chunk_idx
"""

_TASKS: Final = "SELECT task_id, external_id, task_type, prompt, rationale FROM tasks ORDER BY task_id"

# A task's items are one ordered list whatever they name, so the text it answers with and the records
# it is about come back from one query. The record's own id is joined back in, because that is what
# the task's field holds.
_TASK_ITEMS: Final = """
SELECT i.task_id, i.role, i.item_type, i.text_value, r.external_id
FROM task_items i
LEFT JOIN records r ON r.record_id = i.record_id
ORDER BY i.task_id, i.role, i.item_type, i.position
"""

_TASK_FROM_TASKS: Final = "SELECT task_id, external_id FROM task_from_tasks ORDER BY task_id, position"

_TASK_FIELDS: Final = "SELECT task_id, field, text_value, double_value FROM task_fields"

_TASK_REFS: Final = """
SELECT p.task_id, p.field, coalesce(r.external_id, s.external_id) AS external_id
FROM task_refs p
LEFT JOIN records r ON p.ref_kind = 'record' AND r.record_id = p.ref_id
LEFT JOIN time_series s ON p.ref_kind = 'time_series' AND s.time_series_id = p.ref_id
ORDER BY p.task_id, p.field, p.position
"""

_TASK_SPANS: Final = """
SELECT task_id, field, frame, start_at, end_at, time_series_ids
FROM task_spans ORDER BY task_id, field, position
"""

_TASK_ANNOTATIONS: Final = """
SELECT t.task_id, t.role, n.external_id
FROM task_annotations t
JOIN annotations n ON n.annotation_id = t.annotation_id
ORDER BY t.task_id, t.role, t.position
"""


class ControlPlaneReader:
    """An open, read-only view of one version's control database."""

    def __init__(self, version: "DatasetVersion") -> None:
        """Bind the view to a version. The database opens on the first query.

        Args:
            version: The opened version handle, for its filesystem and root.
        """
        self._version = version
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._materialized: Path | None = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The connection, opened on first use and checked against this reader's schema version.

        Returns:
            The connection, with the control database as its default catalog.

        Raises:
            TimeFFormatError: If the database is missing, unreadable, carries no ``meta`` row, or was
                written by a newer schema than this reader knows.
        """
        if self._connection is not None:
            return self._connection
        target = self._database_path()
        try:
            connection = duckdb.connect(str(target), read_only=True)
            row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        except duckdb.Error as exc:
            raise TimeFFormatError(f"cannot read the control database at {target}: {exc}") from exc
        if row is None:
            raise TimeFFormatError(f"the control database at {target} has no schema_version in its meta table")
        if int(row[0]) > ddl.SCHEMA_VERSION:
            connection.close()
            raise TimeFFormatError(
                f"the control database at {target} uses schema version {row[0]}; this reader supports "
                f"{ddl.SCHEMA_VERSION}"
            )
        self._connection = connection
        return connection

    def _database_path(self) -> Path:
        """Return a local path to the control database, copied down first when it is remote.

        Returns:
            The path DuckDB opens.

        Raises:
            TimeFFormatError: If the version has no control database.
        """
        relpath = self._version.path(CONTROL_DB_FILE)
        if isinstance(self._version.filesystem, pafs.LocalFileSystem):
            local = Path(relpath)
            if not local.exists():
                raise TimeFFormatError(f"no control database at {local}")
            return local
        if self._materialized is None:
            handle, path = tempfile.mkstemp(prefix="timef-control-", suffix=".duckdb")
            os.close(handle)
            self._materialized = Path(path)
            try:
                with self._version.filesystem.open_input_stream(relpath) as source:
                    self._materialized.write_bytes(source.readall())
            except OSError as exc:
                self.close()
                raise TimeFFormatError(f"cannot fetch the control database at {relpath}: {exc}") from exc
        return self._materialized

    def close(self) -> None:
        """Close the connection and remove any local copy. Safe to call more than once."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._materialized is not None:
            self._materialized.unlink(missing_ok=True)
            self._materialized = None

    # ---- the type declaration --------------------------------------------------------------

    def specs(self) -> list[dict]:
        """Return the dataset's time-series specs, in the order the schema derived them.

        Returns:
            One row per spec, with every column of its descriptor.
        """
        return self._rows(_SPECS, [])

    def annotation_descriptors(self) -> list[dict]:
        """Return the dataset's annotation descriptors, in the order the schema derived them.

        Returns:
            One row per annotation key.
        """
        return self._rows(_ANNOTATION_DESCRIPTORS, [])

    # ---- records ---------------------------------------------------------------------------

    def record_id_batches(self, batch_size: int) -> Iterator[list[int]]:
        """Yield every record's surrogate id, in stored order, in batches.

        Args:
            batch_size: How many ids per batch.

        Yields:
            One batch of ids at a time.
        """
        # The ids stream from their own cursor, because hydrating a batch runs more queries and a
        # second query on the same connection would discard the result this one is still reading.
        cursor = self.connection.cursor()
        try:
            found = cursor.execute("SELECT record_id FROM records ORDER BY record_id")
            while rows := found.fetchmany(batch_size):
                yield [row[0] for row in rows]
        finally:
            cursor.close()

    def resolve_record_ids(self, external_ids: Sequence[str]) -> tuple[list[int], list[str]]:
        """Turn the caller's record ids into surrogate ids with one query.

        Args:
            external_ids: The ids the caller asked for.

        Returns:
            The surrogate ids of the records that exist, in stored order, and the ids that name no
            record, in the order they were asked for.
        """
        found: dict[str, int] = dict(
            self.connection.execute(
                "SELECT external_id, record_id FROM records WHERE external_id IN (SELECT unnest(?))",
                [list(external_ids)],
            ).fetchall()
        )
        return sorted(found.values()), [name for name in external_ids if name not in found]

    def records(self, record_ids: Sequence[int]) -> list[dict]:
        """Return one batch of record rows, in stored order.

        Args:
            record_ids: The surrogate ids to read.

        Returns:
            One row per record.
        """
        return self._rows(_RECORDS, [list(record_ids)])

    def record_tasks(self, record_ids: Sequence[int]) -> list[dict]:
        """Return the tasks a batch of records carries, by the caller's task id.

        Args:
            record_ids: The surrogate ids of the records to read.

        Returns:
            One row per (record, task) link, in task order within each record.
        """
        return self._rows(_RECORD_TASKS, [list(record_ids)])

    def record_series(self, record_ids: Sequence[int]) -> list[dict]:
        """Return the series of a batch of records, with their spec and axis resolved.

        Args:
            record_ids: The surrogate ids of the records to read.

        Returns:
            One row per (record, series) pair, in each record's own series order.
        """
        return self._rows(_RECORD_SERIES, [list(record_ids)])

    def record_annotations(self, record_ids: Sequence[int]) -> list[dict]:
        """Return the annotations a batch of records carries.

        Args:
            record_ids: The surrogate ids of the records to read.

        Returns:
            One row per attachment, in each record's own annotation order.
        """
        return self._rows(_RECORD_ANNOTATIONS, [list(record_ids)])

    def registered_annotations(self) -> list[dict]:
        """Return the annotations no record carries, which exist only for tasks to reference.

        Returns:
            One row per registered annotation, in registration order.
        """
        return self._rows(_REGISTERED_ANNOTATIONS, [])

    def record_chunks(self, record_ids: Sequence[int]) -> list[dict]:
        """Return every chunk locator of a batch of records' series.

        One statement answers for the whole batch, like every other query here. A read that asked
        one record at a time paid a scan per record, which is the cost this batching removes.

        Args:
            record_ids: The surrogate ids of the records to read, as :meth:`records` reports them.

        Returns:
            One row per chunk, grouped by record and then by series, and ordered by ``chunk_idx``
            within each series.
        """
        return self._rows(_RECORD_CHUNKS, [list(record_ids)])

    # ---- tasks -----------------------------------------------------------------------------

    def tasks(self) -> dict[str, list[dict]]:
        """Return every task and its payload, one query per table.

        Returns:
            The ``tasks``, ``items``, ``from_tasks``, ``fields``, ``refs``, ``spans`` and
            ``annotations`` rows, each ordered so a task's own rows are contiguous and in position
            order.
        """
        return {
            "tasks": self._rows(_TASKS, []),
            "items": self._rows(_TASK_ITEMS, []),
            "from_tasks": self._rows(_TASK_FROM_TASKS, []),
            "fields": self._rows(_TASK_FIELDS, []),
            "refs": self._rows(_TASK_REFS, []),
            "spans": self._rows(_TASK_SPANS, []),
            "annotations": self._rows(_TASK_ANNOTATIONS, []),
        }

    # ---- helpers ---------------------------------------------------------------------------

    def _rows(self, query: str, parameters: list[Any]) -> list[dict]:
        """Run one query and return its rows as dicts.

        The result comes back through Arrow rather than row by row, so a batch of a thousand records
        crosses the boundary once.

        Args:
            query: The SQL to run.
            parameters: Its bound parameters.

        Returns:
            One dict per row, keyed by column name.
        """
        return self.connection.execute(query, parameters).to_arrow_table().to_pylist()
