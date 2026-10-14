"""Audit a finished ``control.duckdb`` for relational integrity.

The control database carries no persisted constraints: ART indexes on the large
``signal_chunks`` tables would slow every write and bloat the file. The writer instead keeps the
invariants while it streams rows, from key sequences it owns and from ID sets it checks in Python.
This module holds the SQL form of the same invariants. The writer scans the finished catalog
before publication. Tooling can also audit an external or published version.
"""

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.control_schema import CONTROL_TABLES, Table
from timenet.format.duckdb import check_control_schema


_DOMAIN_CHECKS = """
    SELECT 'sources.parent_source_key' AS field
    FROM sources WHERE parent_source_key = source_key
    UNION ALL
    SELECT 'datasets' FROM (SELECT count(*) AS n FROM datasets) WHERE n <> 1
    UNION ALL
    SELECT 'signals.n_values' FROM signals WHERE n_values <= 0
    UNION ALL
    SELECT 'signal_chunks.n_values' FROM signal_chunks WHERE n_values <= 0
    UNION ALL
    SELECT 'annotation_occurrences.object_type' FROM annotation_occurrences
    WHERE object_type NOT IN ('Dataset', 'Task', 'Record', 'Source', 'Signal')
    UNION ALL
    SELECT 'annotation_occurrences.span_type' FROM annotation_occurrences
    WHERE span_type NOT IN ('static', 'point', 'interval')
    UNION ALL
    SELECT 'task_targets.target_kind' FROM task_targets
    WHERE target_kind NOT IN (
        'text', 'integer', 'float', 'boolean', 'record', 'signal',
        'time_point', 'time_interval', 'step_point', 'step_interval'
    )
    UNION ALL
    SELECT 'task_dependencies.parent_task_key' FROM task_dependencies
    WHERE task_key = parent_task_key
    UNION ALL
    SELECT 'annotation_contents.value_kind' FROM annotation_contents
    WHERE value_kind NOT IN ('text', 'integer', 'float', 'boolean', 'text_list')
    UNION ALL
    SELECT 'tasks.scope_type' FROM tasks
    WHERE scope_type NOT IN ('time_point', 'time_interval', 'step_point', 'step_interval')
    LIMIT 1
"""

_CROSS_RECORD_PARENT = """
    SELECT child.source_id
    FROM sources child
    JOIN sources parent ON parent.source_key = child.parent_source_key
    WHERE child.record_key <> parent.record_key
    LIMIT 1
"""

_SOURCE_CYCLE = """
    WITH RECURSIVE ancestors(source_key, parent_source_key, path, cyclic) AS (
        SELECT source_key, parent_source_key, [source_key], false FROM sources
        UNION ALL
        SELECT parent.source_key, parent.parent_source_key,
               list_append(ancestors.path, parent.source_key),
               list_contains(ancestors.path, parent.source_key)
        FROM ancestors
        JOIN sources parent ON parent.source_key = ancestors.parent_source_key
        WHERE NOT ancestors.cyclic
    )
    SELECT source_key FROM ancestors WHERE cyclic LIMIT 1
"""

_AXIS_LENGTH = """
    SELECT signals.signal_id
    FROM signals
    JOIN axes USING (axis_key)
    LEFT JOIN (
        SELECT object_id AS axis_id, CAST(json_extract_string(descriptor, '$.length') AS BIGINT) AS offset_count
        FROM payloads JOIN payload_descriptors USING (descriptor_key) WHERE object_type='Axis' AND field='offsets'
    ) offsets USING (axis_id)
    WHERE (axes.axis_type = 'irregular' AND coalesce(offset_count, 0) <> signals.n_values)
       OR (axes.axis_type <> 'irregular' AND coalesce(offset_count, 0) <> 0)
    LIMIT 1
"""

_CHUNK_COVERAGE = """
    SELECT signals.signal_id
    FROM signals
    LEFT JOIN (
        SELECT signal_key, sum(n_values) AS stored_values
        FROM signal_chunks GROUP BY signal_key
    ) chunks USING (signal_key)
    LEFT JOIN payloads p ON p.object_type='Signal' AND p.object_id=signals.signal_id AND p.field='values' AND p.position=0
    LEFT JOIN payload_stores store USING (store_key)
    LEFT JOIN payload_descriptors descriptor USING (descriptor_key)
    WHERE CASE WHEN store.dataset_key IS NOT NULL OR store.prefix <> ''
        THEN CAST(json_extract_string(descriptor.descriptor, '$.length') AS BIGINT) <> signals.n_values
        ELSE coalesce(stored_values, 0) <> signals.n_values END
    LIMIT 1
"""

_TASK_RECORD_SCOPE = """
    WITH spans AS (
        SELECT row_number() OVER () AS span_key, task_key, signal_keys
        FROM (
            SELECT task_key, scope_signal_keys AS signal_keys FROM tasks
            WHERE scope_type IN ('time_point', 'time_interval')
            UNION ALL
            SELECT task_key, signal_keys FROM task_targets
            WHERE target_kind IN ('time_point', 'time_interval')
        )
    )
    SELECT tasks.task_id
    FROM spans
    JOIN tasks USING (task_key)
    LEFT JOIN task_record_refs refs ON refs.task_key = spans.task_key AND refs.field = 'inputs'
    LEFT JOIN sources ON sources.record_key = refs.record_key
    LEFT JOIN signals USING (source_key)
    GROUP BY spans.span_key, tasks.task_id, spans.signal_keys
    HAVING count(DISTINCT CASE
        WHEN spans.signal_keys IS NULL OR list_contains(spans.signal_keys, signals.signal_key)
        THEN refs.record_key END) <> 1
    LIMIT 1
"""

_PAYLOAD_INTEGRITY = """
    SELECT 'payload chunks' AS problem
    FROM payload_chunks c JOIN payloads p USING (payload_key)
    JOIN payload_files f USING (file_key)
    WHERE c.major < 0 OR c.minor < 0 OR c.n_values <= 0 OR c.position < 0
       OR p.store_key <> f.store_key
    UNION ALL
    SELECT 'payload coverage' FROM payloads p
    JOIN payload_descriptors d USING (descriptor_key)
    LEFT JOIN (
        SELECT payload_key, count(*) n, min(position) lo, max(position) hi, sum(n_values) total
        FROM payload_chunks GROUP BY payload_key
    ) c USING (payload_key)
    WHERE p.position < 0 OR c.n IS NULL OR c.lo <> 0 OR c.hi <> c.n - 1
       OR (json_extract_string(d.descriptor, '$.kind') = 'array'
           AND c.total <> CAST(json_extract_string(d.descriptor, '$.length') AS BIGINT))
    UNION ALL
    SELECT 'payload store' FROM payload_stores
    WHERE (dataset_key IS NULL) <> (manifest_checksum IS NULL)
       OR codec NOT IN ('parquet', 'zarr', 'text')
    UNION ALL
    SELECT 'object lineage' FROM origins
    WHERE origin_id = '' OR origin_kind NOT IN (
        'record', 'source', 'signal', 'axis', 'annotation_content', 'annotation_occurrence', 'task'
    )
    LIMIT 1
"""

_PAYLOAD_OBJECTS = """
    WITH objects AS (
        SELECT 'Signal' AS kind, signal_id AS id FROM signals
        UNION ALL SELECT 'Axis', axis_id FROM axes
        UNION ALL SELECT 'AnnotationContent', content_id FROM annotation_contents
        UNION ALL SELECT 'Task', task_id FROM tasks
    )
    SELECT p.object_type, p.object_id FROM payloads p
    LEFT JOIN objects o ON p.object_type = o.kind AND p.object_id = o.id
    WHERE o.id IS NULL LIMIT 1
"""


def audit_control_database(connection: duckdb.DuckDBPyConnection, *, require_chunks: bool = True) -> None:
    """Scan every control table and raise on the first violated invariant.

    Args:
        connection: An open connection to the control database.
        require_chunks: Whether every Signal must be covered exactly by ``signal_chunks`` rows.
            A metadata-only database written without value placements has none.

    Raises:
        TimeFFormatError: If a key is repeated, a relationship dangles, a value is outside its
            domain, the Source tree is not a forest of per-Record trees, or an axis or chunk
            length disagrees with its Signal.
    """
    for table in CONTROL_TABLES:
        _audit_unique_keys(connection, table)
        _audit_foreign_keys(connection, table)

    invalid = connection.execute(_DOMAIN_CHECKS).fetchone()
    if invalid is not None:
        raise TimeFFormatError(f"control database contains an invalid {invalid[0]} value")

    check_control_schema(connection)
    invalid_modality = connection.execute(
        """SELECT task_id FROM tasks
           WHERE len(input_modalities) = 0
              OR EXISTS (SELECT 1 FROM unnest(input_modalities) AS m(value)
                         WHERE value NOT IN ('text', 'time_series', 'image', 'audio', 'no_input'))
              OR len(input_modalities) <> len(list_distinct(input_modalities))
              OR (prompt IS NOT NULL AND prompt <> ''
                  AND NOT list_contains(input_modalities, 'text'))
              OR (list_contains(input_modalities, 'no_input')
                  AND (len(list_filter(input_modalities, x -> x NOT IN ('text', 'no_input'))) > 0
                       OR EXISTS (SELECT 1 FROM task_record_refs r
                                  WHERE r.task_key = tasks.task_key AND r.field = 'inputs')))
           LIMIT 1"""
    ).fetchone()
    if invalid_modality is not None:
        raise TimeFFormatError(f"task {invalid_modality[0]!r} has invalid input_modalities")

    invalid_clock_scope = connection.execute(_TASK_RECORD_SCOPE).fetchone()
    if invalid_clock_scope is not None:
        raise TimeFFormatError(f"task {invalid_clock_scope[0]!r} time span must resolve to exactly one input Record")

    wrong_parent = connection.execute(_CROSS_RECORD_PARENT).fetchone()
    if wrong_parent is not None:
        raise TimeFFormatError(f"source {wrong_parent[0]!r} and its parent belong to different records")

    cycle = connection.execute(_SOURCE_CYCLE).fetchone()
    if cycle is not None:
        raise TimeFFormatError(f"source hierarchy contains a cycle at {cycle[0]!r}")

    bad_axis = connection.execute(_AXIS_LENGTH).fetchone()
    if bad_axis is not None:
        raise TimeFFormatError(f"signal {bad_axis[0]!r} has an axis length that does not match its values")

    if require_chunks:
        bad_chunks = connection.execute(_CHUNK_COVERAGE).fetchone()
        if bad_chunks is not None:
            raise TimeFFormatError(f"signal {bad_chunks[0]!r} has value chunks whose lengths do not match n_values")

    invalid = connection.execute(_PAYLOAD_INTEGRITY).fetchone()
    if invalid is not None:
        raise TimeFFormatError(f"control database contains invalid {invalid[0]}")
    invalid = connection.execute(_PAYLOAD_OBJECTS).fetchone()
    if invalid is not None:
        raise TimeFFormatError(f"payload refers to absent {invalid[0]} {invalid[1]!r}")


def _audit_unique_keys(connection: duckdb.DuckDBPyConnection, table: Table) -> None:
    """Raise on the first repeated logical key of ``table``.

    Raises:
        TimeFFormatError: If the table holds two rows with the same key columns.
    """
    for columns in table.unique:
        names = ", ".join(columns)
        duplicate = connection.execute(
            f"SELECT {names} FROM {table.name} GROUP BY {names} HAVING count(*) > 1 LIMIT 1"  # noqa: S608
        ).fetchone()
        if duplicate is not None:
            raise TimeFFormatError(f"{table.name} contains duplicate values for {names}: {duplicate!r}")


def _audit_foreign_keys(connection: duckdb.DuckDBPyConnection, table: Table) -> None:
    """Raise on the first dangling relationship from ``table``.

    Raises:
        TimeFFormatError: If a child column refers to a parent key that does not exist.
    """
    for key in table.foreign_keys:
        orphan = connection.execute(
            f"""SELECT child.{key.column}
                FROM {table.name} child
                LEFT JOIN {key.table} parent
                  ON parent.{key.references} = child.{key.column}
                WHERE child.{key.column} IS NOT NULL
                  AND parent.{key.references} IS NULL
                LIMIT 1"""  # noqa: S608
        ).fetchone()
        if orphan is not None:
            raise TimeFFormatError(
                f"{table.name}.{key.column} refers to missing {key.table}.{key.references} value {orphan[0]!r}"
            )
