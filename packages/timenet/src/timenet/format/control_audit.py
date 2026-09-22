"""Audit a finished ``control.duckdb`` for relational integrity.

The control database carries no persisted constraints: ART indexes on the large ``axis_offsets`` and
``signal_chunks`` tables would slow every write and bloat the file. The writer instead keeps the
invariants while it streams rows, from key sequences it owns and from ID sets it checks in Python.
This module holds the SQL form of the same invariants. It scans every table, so it is not part of
the write path. Tests run it after each round trip, and tooling can run it over a published version.
"""

from collections.abc import Iterable

import duckdb

from timenet.errors import TimeFFormatError


_UNIQUE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("control_metadata", ("key",)),
    ("datasets", ("dataset_key",)),
    ("datasets", ("dataset_id",)),
    ("records", ("record_key",)),
    ("records", ("record_id",)),
    ("sources", ("source_key",)),
    ("sources", ("source_id",)),
    ("axes", ("axis_key",)),
    ("axes", ("axis_id",)),
    ("axis_offsets", ("axis_key", "position")),
    ("signals", ("signal_key",)),
    ("signals", ("signal_id",)),
    ("signal_chunks", ("signal_key", "chunk_index")),
    ("annotation_contents", ("content_key",)),
    ("annotation_contents", ("content_id",)),
    ("annotation_occurrences", ("occurrence_key",)),
    ("annotation_occurrences", ("occurrence_id",)),
    ("tasks", ("task_key",)),
    ("tasks", ("task_id",)),
    ("task_targets", ("task_key", "position")),
    ("task_record_refs", ("task_key", "field", "position")),
    ("task_annotation_refs", ("task_key", "field", "position")),
    ("task_dependencies", ("task_key", "position")),
)
"""Column sets that identify one row of their table."""

_FOREIGN_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("sources", "record_key", "records", "record_key"),
    ("sources", "parent_source_key", "sources", "source_key"),
    ("axis_offsets", "axis_key", "axes", "axis_key"),
    ("signals", "source_key", "sources", "source_key"),
    ("signals", "axis_key", "axes", "axis_key"),
    ("signal_chunks", "signal_key", "signals", "signal_key"),
    ("annotation_occurrences", "content_key", "annotation_contents", "content_key"),
    ("task_targets", "task_key", "tasks", "task_key"),
    ("task_targets", "record_key", "records", "record_key"),
    ("task_targets", "signal_key", "signals", "signal_key"),
    ("task_record_refs", "task_key", "tasks", "task_key"),
    ("task_record_refs", "record_key", "records", "record_key"),
    ("task_annotation_refs", "task_key", "tasks", "task_key"),
    ("task_annotation_refs", "occurrence_key", "annotation_occurrences", "occurrence_key"),
    ("task_dependencies", "task_key", "tasks", "task_key"),
    ("task_dependencies", "parent_task_key", "tasks", "task_key"),
)
"""``(child table, child column, parent table, parent column)`` relationships."""

_DOMAIN_CHECKS = """
    SELECT 'sources.parent_source_key' AS field
    FROM sources WHERE parent_source_key = source_key
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
    _audit_unique_keys(connection, _UNIQUE_KEYS)
    _audit_foreign_keys(connection, _FOREIGN_KEYS)

    invalid = connection.execute(_DOMAIN_CHECKS).fetchone()
    if invalid is not None:
        raise TimeFFormatError(f"control database contains an invalid {invalid[0]} value")

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


def _audit_unique_keys(
    connection: duckdb.DuckDBPyConnection,
    unique_keys: Iterable[tuple[str, tuple[str, ...]]],
) -> None:
    """Raise on the first repeated logical key.

    Raises:
        TimeFFormatError: If a table holds two rows with the same key columns.
    """
    for table, columns in unique_keys:
        names = ", ".join(columns)
        duplicate = connection.execute(
            f"SELECT {names} FROM {table} GROUP BY {names} HAVING count(*) > 1 LIMIT 1"  # noqa: S608
        ).fetchone()
        if duplicate is not None:
            raise TimeFFormatError(f"{table} contains duplicate values for {names}: {duplicate!r}")


def _audit_foreign_keys(
    connection: duckdb.DuckDBPyConnection,
    foreign_keys: Iterable[tuple[str, str, str, str]],
) -> None:
    """Raise on the first dangling relationship.

    Raises:
        TimeFFormatError: If a child column refers to a parent key that does not exist.
    """
    for child_table, child_column, parent_table, parent_column in foreign_keys:
        orphan = connection.execute(
            f"""SELECT child.{child_column}
                FROM {child_table} child
                LEFT JOIN {parent_table} parent
                  ON parent.{parent_column} = child.{child_column}
                WHERE child.{child_column} IS NOT NULL
                  AND parent.{parent_column} IS NULL
                LIMIT 1"""  # noqa: S608
        ).fetchone()
        if orphan is not None:
            raise TimeFFormatError(
                f"{child_table}.{child_column} refers to missing {parent_table}.{parent_column} value {orphan[0]!r}"
            )
