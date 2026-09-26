"""Audit a finished ``control.duckdb`` for relational integrity.

The control database carries no persisted constraints: ART indexes on the large ``axis_offsets`` and
``signal_chunks`` tables would slow every write and bloat the file. The writer instead keeps the
invariants while it streams rows, from key sequences it owns and from ID sets it checks in Python.
This module holds the SQL form of the same invariants. It scans every table, so it is not part of
the write path. Tests run it after each round trip, and tooling can run it over a published version.
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
        SELECT axis_key, count(*) AS offset_count FROM axis_offsets GROUP BY axis_key
    ) offsets USING (axis_key)
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
    WHERE coalesce(stored_values, 0) <> signals.n_values
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
