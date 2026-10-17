"""Audit a finished ``control.duckdb`` for relational integrity.

The control database carries no persisted constraints: ART indexes on the large ``axis_offsets`` and
``signal_chunks`` tables would slow every write and bloat the file. The writer instead keeps the
invariants while it streams rows, from the ID sets it checks in Python.
This module holds the SQL form of the same invariants. It scans every table, so it is not part of
the write path. Tests run it after each round trip, and tooling can run it over a published version.
"""

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.control_schema import ANNOTATED_OBJECT_TYPES, CONTROL_TABLES, Table
from timenet.format.duckdb import check_control_schema


_OBJECT_TYPES_SQL = ", ".join(f"'{name}'" for name in sorted(ANNOTATED_OBJECT_TYPES))

_DOMAIN_CHECKS = f"""
    SELECT 'sources.parent_source_id' AS field
    FROM sources WHERE parent_source_id = source_id
    UNION ALL
    SELECT 'datasets' FROM (SELECT count(*) AS n FROM datasets) WHERE n <> 1
    UNION ALL
    SELECT 'signals.n_values' FROM signals WHERE n_values <= 0
    UNION ALL
    SELECT 'signal_chunks.n_values' FROM signal_chunks WHERE n_values <= 0
    UNION ALL
    SELECT 'annotation_occurrences.object_type' FROM annotation_occurrences
    WHERE object_type NOT IN ({_OBJECT_TYPES_SQL})
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
    SELECT 'task_dependencies.parent_task_id' FROM task_dependencies
    WHERE task_id = parent_task_id
    UNION ALL
    SELECT 'annotation_contents.value_kind' FROM annotation_contents
    WHERE value_kind NOT IN ('text', 'integer', 'float', 'boolean', 'text_list')
    UNION ALL
    SELECT 'tasks.scope_type' FROM tasks
    WHERE scope_type NOT IN ('time_point', 'time_interval', 'step_point', 'step_interval')
    UNION ALL
    SELECT 'records.record_id' FROM records WHERE record_id <= 0
    UNION ALL
    SELECT 'sources.source_id' FROM sources WHERE source_id <= 0
    UNION ALL
    SELECT 'axes.axis_id' FROM axes WHERE axis_id <= 0
    UNION ALL
    SELECT 'signals.signal_id' FROM signals WHERE signal_id <= 0
    UNION ALL
    SELECT 'annotation_contents.content_id' FROM annotation_contents WHERE content_id <= 0
    UNION ALL
    SELECT 'annotation_occurrences.occurrence_id' FROM annotation_occurrences WHERE occurrence_id <= 0
    UNION ALL
    SELECT 'tasks.task_id' FROM tasks WHERE task_id <= 0
    LIMIT 1
"""  # noqa: S608 - fixed type names

_OCCURRENCE_OWNER = """
    SELECT o.occurrence_id, o.object_type, o.object_id
    FROM annotation_occurrences o
    LEFT JOIN records ON o.object_type = 'Record' AND records.record_id = o.object_id
    LEFT JOIN sources ON o.object_type = 'Source' AND sources.source_id = o.object_id
    LEFT JOIN signals ON o.object_type = 'Signal' AND signals.signal_id = o.object_id
    LEFT JOIN tasks ON o.object_type = 'Task' AND tasks.task_id = o.object_id
    WHERE CASE o.object_type
        WHEN 'Dataset' THEN o.object_id IS NOT NULL
        WHEN 'Record' THEN records.record_id IS NULL
        WHEN 'Source' THEN sources.source_id IS NULL
        WHEN 'Signal' THEN signals.signal_id IS NULL
        WHEN 'Task' THEN tasks.task_id IS NULL
    END
    LIMIT 1
"""

_CROSS_RECORD_PARENT = """
    SELECT child.source_id
    FROM sources child
    JOIN sources parent ON parent.source_id = child.parent_source_id
    WHERE child.record_id <> parent.record_id
    LIMIT 1
"""

_SOURCE_CYCLE = """
    WITH RECURSIVE ancestors(source_id, parent_source_id, path, cyclic) AS (
        SELECT source_id, parent_source_id, [source_id], false FROM sources
        UNION ALL
        SELECT parent.source_id, parent.parent_source_id,
               list_append(ancestors.path, parent.source_id),
               list_contains(ancestors.path, parent.source_id)
        FROM ancestors
        JOIN sources parent ON parent.source_id = ancestors.parent_source_id
        WHERE NOT ancestors.cyclic
    )
    SELECT source_id FROM ancestors WHERE cyclic LIMIT 1
"""

_AXIS_LENGTH = """
    SELECT signals.signal_id
    FROM signals
    JOIN axes USING (axis_id)
    LEFT JOIN (
        SELECT axis_id, count(*) AS offset_count FROM axis_offsets GROUP BY axis_id
    ) offsets USING (axis_id)
    WHERE (axes.axis_type = 'irregular' AND coalesce(offset_count, 0) <> signals.n_values)
       OR (axes.axis_type <> 'irregular' AND coalesce(offset_count, 0) <> 0)
    LIMIT 1
"""

_CHUNK_COVERAGE = """
    SELECT signals.signal_id
    FROM signals
    LEFT JOIN (
        SELECT signal_id, sum(n_values) AS stored_values
        FROM signal_chunks GROUP BY signal_id
    ) chunks USING (signal_id)
    WHERE coalesce(stored_values, 0) <> signals.n_values
    LIMIT 1
"""

_TASK_SPAN_SIGNALS = """
    SELECT spans.task_id, spans.signal_id
    FROM (
        SELECT task_id, unnest(scope_signal_ids) AS signal_id FROM tasks
        UNION ALL
        SELECT task_id, unnest(span_signal_ids) FROM task_targets
    ) spans
    LEFT JOIN signals ON signals.signal_id = spans.signal_id
    WHERE signals.signal_id IS NULL
    LIMIT 1
"""

_TASK_RECORD_SCOPE = """
    WITH spans AS (
        SELECT row_number() OVER () AS span_number, task_id, signal_ids
        FROM (
            SELECT task_id, scope_signal_ids AS signal_ids FROM tasks
            WHERE scope_type IN ('time_point', 'time_interval')
            UNION ALL
            SELECT task_id, span_signal_ids FROM task_targets
            WHERE target_kind IN ('time_point', 'time_interval')
        )
    )
    SELECT spans.task_id
    FROM spans
    LEFT JOIN task_record_refs refs ON refs.task_id = spans.task_id AND refs.field = 'inputs'
    LEFT JOIN sources ON sources.record_id = refs.record_id
    LEFT JOIN signals USING (source_id)
    GROUP BY spans.span_number, spans.task_id, spans.signal_ids
    HAVING count(DISTINCT CASE
        WHEN spans.signal_ids IS NULL OR list_contains(spans.signal_ids, signals.signal_id)
        THEN refs.record_id END) <> 1
    LIMIT 1
"""


def audit_control_database(connection: duckdb.DuckDBPyConnection, *, require_chunks: bool = True) -> None:
    """Scan every control table and raise on the first violated invariant.

    Args:
        connection: An open connection to the control database.
        require_chunks: Whether every Signal must be covered exactly by ``signal_chunks`` rows.
            A metadata-only database written without value placements has none.

    Raises:
        TimeFFormatError: If a key is repeated, a relationship dangles, a value is outside its
            domain, a task span names a missing Signal, the Source tree is not a forest of
            per-Record trees, or an axis or chunk length disagrees with its Signal.
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
                                  WHERE r.task_id = tasks.task_id AND r.field = 'inputs')))
           LIMIT 1"""
    ).fetchone()
    if invalid_modality is not None:
        raise TimeFFormatError(f"task {invalid_modality[0]!r} has invalid input_modalities")

    orphan_occurrence = connection.execute(_OCCURRENCE_OWNER).fetchone()
    if orphan_occurrence is not None:
        occurrence_id, object_type, object_id = orphan_occurrence
        if object_type == "Dataset":
            raise TimeFFormatError(
                f"annotation occurrence {occurrence_id!r} on the Dataset has object id {object_id!r}"
            )
        raise TimeFFormatError(f"annotation occurrence {occurrence_id!r} refers to missing {object_type} {object_id!r}")

    missing_span_signal = connection.execute(_TASK_SPAN_SIGNALS).fetchone()
    if missing_span_signal is not None:
        task_id, signal_id = missing_span_signal
        raise TimeFFormatError(f"task {task_id!r} has a span that refers to missing Signal {signal_id!r}")

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
