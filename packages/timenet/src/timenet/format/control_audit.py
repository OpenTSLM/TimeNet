"""Audit a finished ``control.duckdb`` for relational integrity.

The control database has no persisted constraints. ART indexes on the large ``axis_offsets`` and
``signal_chunks`` tables make every write slower and the file larger. Instead, the writer keeps the
invariants while it streams rows. It checks them in Python with ID sets. This module holds the SQL
form of the same invariants. It scans every table, so it is not part of the write path. Tests run it
after each round trip, and tooling can run it on a published version.
"""

from collections.abc import Iterable

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.annotation_value_kind import AnnotationValueKind
from timenet.format.control_schema import (
    ANNOTATED_OBJECT_TYPES,
    ANNOTATED_OWNER_TABLES,
    CONTROL_TABLES,
    EXTERNAL_OWNER_TABLES,
    Table,
)
from timenet.format.duckdb import check_control_schema
from timenet.format.stored_values import PackedField, PackedValueKind
from timenet.format.task_codec import ANNOTATION_FIELDS, RECORD_FIELDS, TaskRelationshipField
from timenet.format.task_target_kind import SPAN_KINDS, TaskTargetKind
from timenet.types import ObjectKind


def _sql_in(values: Iterable[str]) -> str:
    return ", ".join(f"'{value}'" for value in sorted(values))


_TARGET_KINDS = _sql_in(TaskTargetKind)
_ANNOTATION_VALUE_KINDS = _sql_in(AnnotationValueKind)
_TIME_TARGET_KINDS = _sql_in((TaskTargetKind.TIME_POINT, TaskTargetKind.TIME_INTERVAL))
_SPAN_KINDS = _sql_in(SPAN_KINDS)
_RECORD_FIELDS = _sql_in(RECORD_FIELDS)
_ANNOTATION_FIELDS = _sql_in(ANNOTATION_FIELDS)
_OBJECT_TYPES_SQL = _sql_in(ANNOTATED_OBJECT_TYPES)
_EXTERNAL_KINDS_SQL = _sql_in(EXTERNAL_OWNER_TABLES)

_ENTITY_ID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("records", "record_id"),
    ("sources", "source_id"),
    ("axes", "axis_id"),
    ("signals", "signal_id"),
    ("annotation_contents", "content_id"),
    ("annotation_occurrences", "occurrence_id"),
    ("tasks", "task_id"),
)
"""The primary id column of each entity table."""

_ENTITY_ID_CHECKS = " UNION ALL ".join(
    f"SELECT '{table}.{column}' FROM {table} WHERE {column} <= 0"  # noqa: S608 - fixed table names
    for table, column in _ENTITY_ID_COLUMNS
)

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
    SELECT 'external_values.object_kind' FROM external_values
    WHERE object_kind NOT IN ({_EXTERNAL_KINDS_SQL})
    UNION ALL
    SELECT 'annotation_occurrences.span_type' FROM annotation_occurrences
    WHERE span_type NOT IN ('static', 'point', 'interval')
    UNION ALL
    SELECT 'task_targets.target_kind' FROM task_targets
    WHERE target_kind NOT IN ({_TARGET_KINDS})
    UNION ALL
    SELECT 'task_record_refs.field' FROM task_record_refs
    WHERE field NOT IN ({_RECORD_FIELDS})
    UNION ALL
    SELECT 'task_annotation_refs.field' FROM task_annotation_refs
    WHERE field NOT IN ({_ANNOTATION_FIELDS})
    UNION ALL
    SELECT 'task_dependencies.parent_task_id' FROM task_dependencies
    WHERE task_id = parent_task_id
    UNION ALL
    SELECT 'annotation_contents.value_kind' FROM annotation_contents
    WHERE value_kind NOT IN ({_ANNOTATION_VALUE_KINDS})
    UNION ALL
    SELECT 'tasks.scope_type' FROM tasks
    WHERE scope_type NOT IN ({_SPAN_KINDS})
    UNION ALL
    {_ENTITY_ID_CHECKS}
    LIMIT 1
"""  # noqa: S608 - tags come only from internal enum values and fixed table names

_OWNERLESS = {
    object_type: (
        "object_id IS NOT NULL"
        if owner is None
        else f"NOT EXISTS (SELECT 1 FROM {owner[0]} WHERE {owner[0]}.{owner[1]} = object_id)"  # noqa: S608
    )
    for object_type, owner in ANNOTATED_OWNER_TABLES.items()
}
"""Per object type, the predicate of an occurrence whose owner does not exist."""

_OCCURRENCE_OWNER = " UNION ALL ".join(
    f"SELECT occurrence_id, object_type, object_id FROM annotation_occurrences "  # noqa: S608
    f"WHERE object_type = '{object_type}' AND {predicate}"
    for object_type, predicate in _OWNERLESS.items()
)

_EXTERNAL_OWNER = " UNION ALL ".join(
    f"SELECT field, object_kind, object_id FROM external_values WHERE object_kind = '{kind}' "  # noqa: S608
    f"AND NOT EXISTS (SELECT 1 FROM {table} WHERE {table}.{column} = external_values.object_id)"
    for kind, (table, column) in EXTERNAL_OWNER_TABLES.items()
)
"""The external values whose owner row does not exist."""

_EXTERNAL_TARGET = f"""
    SELECT external.object_id, external.position
    FROM external_values external
    WHERE external.object_kind = '{ObjectKind.TASK}' AND external.field = '{PackedField.TARGETS}'
      AND NOT EXISTS (
          SELECT 1 FROM task_targets targets
          WHERE targets.task_id = external.object_id AND targets.position = external.position
            AND targets.target_kind = '{TaskTargetKind.TEXT}'
      )
    LIMIT 1
"""  # noqa: S608 - tags come only from internal enum values
"""A packed target whose task has no text target at its position."""

_EXTERNAL_CONTENT_KIND = f"""
    SELECT external.object_id, external.value_kind, contents.value_kind
    FROM external_values external
    JOIN annotation_contents contents ON contents.content_id = external.object_id
    WHERE external.object_kind = '{ObjectKind.ANNOTATION_CONTENT}'
      AND contents.value_kind IS DISTINCT FROM CASE external.value_kind
          WHEN '{PackedValueKind.TEXT}' THEN '{AnnotationValueKind.TEXT}'
          WHEN '{PackedValueKind.TEXT_LIST}' THEN '{AnnotationValueKind.TEXT_LIST}'
      END
    LIMIT 1
"""  # noqa: S608 - tags come only from internal enum values
"""A packed annotation value whose content row holds another kind of value."""

CROSS_RECORD_PARENT = """
    SELECT child.source_id
    FROM sources child
    JOIN sources parent ON parent.source_id = child.parent_source_id
    WHERE child.record_id <> parent.record_id
    LIMIT 1
"""

SOURCE_CYCLE = """
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

_TASK_RECORD_SCOPE = f"""
    WITH spans AS (
        SELECT row_number() OVER () AS span_number, task_id, signal_ids
        FROM (
            SELECT task_id, scope_signal_ids AS signal_ids FROM tasks
            WHERE scope_type IN ({_TIME_TARGET_KINDS})
            UNION ALL
            SELECT task_id, span_signal_ids FROM task_targets
            WHERE target_kind IN ({_TIME_TARGET_KINDS})
        )
    )
    SELECT spans.task_id
    FROM spans
    LEFT JOIN task_record_refs refs ON refs.task_id = spans.task_id AND refs.field = '{TaskRelationshipField.INPUTS}'
    LEFT JOIN sources ON sources.record_id = refs.record_id
    LEFT JOIN signals USING (source_id)
    GROUP BY spans.span_number, spans.task_id, spans.signal_ids
    HAVING count(DISTINCT CASE
        WHEN spans.signal_ids IS NULL OR list_contains(spans.signal_ids, signals.signal_id)
        THEN refs.record_id END) <> 1
    LIMIT 1
"""  # noqa: S608 - tags come only from internal enum values


def audit_control_database(connection: duckdb.DuckDBPyConnection, *, require_chunks: bool = True) -> None:
    """Scan every control table and raise on the first violated invariant.

    Args:
        connection: An open connection to the control database.
        require_chunks: Whether every Signal must be covered exactly by ``signal_chunks`` rows.
            A metadata-only database written without value placements has none.

    Raises:
        TimeFFormatError: If a key is repeated, a relationship dangles, a value is outside its
            domain, the Source tree is not a forest of per-Record trees, or an axis or chunk length
            disagrees with its Signal.
    """
    for table in CONTROL_TABLES:
        _audit_unique_keys(connection, table)
        _audit_foreign_keys(connection, table)

    invalid = connection.execute(_DOMAIN_CHECKS).fetchone()
    if invalid is not None:
        raise TimeFFormatError(f"control database contains an invalid {invalid[0]} value")

    check_control_schema(connection)
    invalid_modality = connection.execute(
        f"""SELECT task_id FROM tasks
           WHERE len(input_modalities) = 0
              OR EXISTS (SELECT 1 FROM unnest(input_modalities) AS m(value)
                         WHERE value NOT IN ('text', 'time_series', 'image', 'audio', 'no_input'))
              OR len(input_modalities) <> len(list_distinct(input_modalities))
              OR ((prompt IS NOT NULL AND prompt <> ''
                   OR EXISTS (SELECT 1 FROM external_values external
                              WHERE external.object_kind = '{ObjectKind.TASK}'
                                AND external.object_id = tasks.task_id
                                AND external.field = '{PackedField.PROMPT}'))
                  AND NOT list_contains(input_modalities, 'text'))
              OR (list_contains(input_modalities, 'no_input')
                  AND (len(list_filter(input_modalities, x -> x NOT IN ('text', 'no_input'))) > 0
                       OR EXISTS (SELECT 1 FROM task_record_refs r
                                  WHERE r.task_id = tasks.task_id AND r.field = '{TaskRelationshipField.INPUTS}')))
           LIMIT 1"""  # noqa: S608 - fixed field name
    ).fetchone()
    if invalid_modality is not None:
        raise TimeFFormatError(f"task {invalid_modality[0]!r} has invalid input_modalities")

    orphan_occurrence = connection.execute(f"{_OCCURRENCE_OWNER} LIMIT 1").fetchone()
    if orphan_occurrence is not None:
        occurrence_id, object_type, object_id = orphan_occurrence
        if object_type == ObjectKind.DATASET:
            raise TimeFFormatError(
                f"annotation occurrence {occurrence_id!r} on the Dataset has object id {object_id!r}"
            )
        raise TimeFFormatError(f"annotation occurrence {occurrence_id!r} refers to missing {object_type} {object_id!r}")

    _audit_external_values(connection)

    invalid_clock_scope = connection.execute(_TASK_RECORD_SCOPE).fetchone()
    if invalid_clock_scope is not None:
        raise TimeFFormatError(f"task {invalid_clock_scope[0]!r} time span must resolve to exactly one input Record")

    wrong_parent = connection.execute(CROSS_RECORD_PARENT).fetchone()
    if wrong_parent is not None:
        raise TimeFFormatError(f"source {wrong_parent[0]!r} and its parent belong to different records")

    cycle = connection.execute(SOURCE_CYCLE).fetchone()
    if cycle is not None:
        raise TimeFFormatError(f"source hierarchy contains a cycle at {cycle[0]!r}")

    bad_axis = connection.execute(_AXIS_LENGTH).fetchone()
    if bad_axis is not None:
        raise TimeFFormatError(f"signal {bad_axis[0]!r} has an axis length that does not match its values")

    if require_chunks:
        bad_chunks = connection.execute(_CHUNK_COVERAGE).fetchone()
        if bad_chunks is not None:
            raise TimeFFormatError(f"signal {bad_chunks[0]!r} has value chunks whose lengths do not match n_values")


def _audit_external_values(connection: duckdb.DuckDBPyConnection) -> None:
    """Raise on the first external value that does not fit the row it belongs to.

    Raises:
        TimeFFormatError: If a locator's owner row is missing, a packed target has no text target
            row at its position, or a packed annotation value is of another kind than its content.
    """
    orphan_value = connection.execute(f"{_EXTERNAL_OWNER} LIMIT 1").fetchone()
    if orphan_value is not None:
        field, object_kind, object_id = orphan_value
        raise TimeFFormatError(f"external value {field!r} refers to missing {object_kind} {object_id!r}")

    orphan_target = connection.execute(_EXTERNAL_TARGET).fetchone()
    if orphan_target is not None:
        task_id, position = orphan_target
        raise TimeFFormatError(f"external target {position} of Task {task_id!r} has no text target row")

    wrong_kind = connection.execute(_EXTERNAL_CONTENT_KIND).fetchone()
    if wrong_kind is not None:
        content_id, packed_kind, content_kind = wrong_kind
        raise TimeFFormatError(
            f"external value of Content {content_id!r} is {packed_kind}, but the content holds {content_kind}"
        )


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
        child = (
            f"(SELECT unnest({key.column}) AS {key.column} FROM {table.name})" if key.many else table.name  # noqa: S608
        )
        orphan = connection.execute(
            f"""SELECT child.{key.column}
                FROM {child} child
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
