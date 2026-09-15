"""What the writer checks before it commits, as queries that must return no rows.

These are not tests. They run inside the load transaction, against the loaded tables, in the moment
between the last insert and ``COMMIT``, and they stand in for the primary and foreign keys the
shipped database does not declare. That trade is measured: on ECG-QA (1.35M control rows) plain
tables are 19.7 MB and the same content with keys and indexes is 189.0 MB, and the write takes 3.9 s
against 11.9 s. A version is written once by one process and is immutable afterwards, so a
constraint carried in the file would re-check, for the rest of the dataset's life, something that
cannot change. Checking each invariant once, here, costs the load one bulk anti-join per check and
costs every later reader nothing.

A failed check aborts the transaction, so a build that does not hold together publishes nothing and
leaves no file behind. Each check reports every offending row it found rather than dying on the
first, which is why a resolved id that names nothing is stored as null and caught here instead of
failing its insert.

The builders below generate the repetitive checks: the same density check for every keyed table, the
same anti-join for every attachment table, and the payload checks that come from the declaration in
:mod:`timenet.control_plane.payload`.
"""

from collections.abc import Sequence
from typing import Final

from timenet.control_plane.payload import (
    TASK_PAYLOAD,
    PayloadKind,
    required_payload_fields,
    task_payload,
    text_answer,
)
from timenet.control_plane.schema import ANNOTATION_TABLES, EXTERNAL_IDS, KEYED_TABLES


def _dense_ids(table: str, column: str) -> tuple[str, str]:
    """Return the check that one table's ids are 0, 1, 2, with no gap and no repeat.

    Density is not cosmetic. A reader that partitions a corpus across workers with
    ``id % num_workers = worker_index`` covers every row exactly once only while the ids run without
    gaps, and a gap would show up as a worker silently seeing fewer records.

    Args:
        table: The table to check.
        column: Its surrogate id column.

    Returns:
        A description and the query, which must return no rows.
    """
    return (
        f"{table}.{column} is not dense",
        # Counting distinct ids catches a repeat; comparing the row count against the largest id
        # catches a gap. Both are needed: one repeat plus one gap leaves the row count unchanged.
        f"SELECT count(*) FROM {table} HAVING count(*) <> count(DISTINCT {column}) "  # noqa: S608
        f"OR count(*) - 1 <> max({column}) OR min({column}) <> 0",
    )


def _unique_external_id(table: str) -> tuple[str, str]:
    """Return the check that one table's external ids are unique.

    Args:
        table: The table to check.

    Returns:
        A description and the query, which must return no rows.
    """
    return (
        f"duplicate {table}.external_id",
        f"SELECT external_id FROM {table} GROUP BY external_id HAVING count(*) > 1",  # noqa: S608
    )


def _attachment_targets() -> tuple[tuple[str, str], ...]:
    """Return the reference checks for every attachment table.

    Each is a bare anti-join with no ``WHERE``, because a table-per-kind layout makes every target
    column ``NOT NULL``.

    Returns:
        One description and query per check.
    """
    entity_of = {"record": "records", "task": "tasks"}
    checks: list[tuple[str, str]] = []
    for kind, (table, column) in ANNOTATION_TABLES.items():
        checks.append(
            (
                f"{table} names an annotation that does not exist",
                f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                f"ANTI JOIN annotations n ON n.annotation_id = a.annotation_id",
            )
        )
        if column is not None:
            checks.append(
                (
                    f"{table} names a {kind} that does not exist",
                    f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                    f"ANTI JOIN {entity_of[kind]} e ON e.{column} = a.{column}",
                )
            )
    return tuple(checks)


def _task_payload_owners() -> tuple[tuple[str, str], ...]:
    """Return the check that every payload row belongs to a task that exists.

    Returns:
        One description and query per payload table.
    """
    return tuple(
        (
            f"{table} names a task that does not exist",
            f"SELECT p.task_id FROM {table} p ANTI JOIN tasks t ON t.task_id = p.task_id",  # noqa: S608
        )
        for table in ("task_items", "task_from_tasks", "task_fields", "task_refs", "task_spans")
    )


def _rows(rows: Sequence[tuple[str, ...]]) -> str:
    """Render a declaration as a SQL ``VALUES`` list.

    Args:
        rows: The tuples to render. Every value comes from this package's own declarations or from a
            task dataclass, never from input.

    Returns:
        The rendered list, ready to follow ``VALUES``.
    """
    return ", ".join("(" + ", ".join(f"'{value}'" for value in row) + ")" for row in rows)


def _typed_task_payload() -> tuple[tuple[str, str], ...]:
    """Return the checks that each task's stored payload is the one its class declares.

    Main enforced the typing with one table per task type: a column per field, and the database
    refused a row that did not fit. An entity-attribute-value payload buys a stable set of tables at
    the cost of that refusal, so these checks put it back. They compare the stored rows against the
    declaration in one pass each, joining on a rendered ``VALUES`` list rather than scanning the
    payload tables once per task type.

    Returns:
        One description and query per check. Each must return no rows.
    """
    field_rows: list[tuple[str, ...]] = []
    ref_rows: list[tuple[str, ...]] = []
    span_rows: list[tuple[str, ...]] = []
    required_rows: list[tuple[str, ...]] = []
    answering_types: list[str] = []
    for task_type in TASK_PAYLOAD:
        answer = text_answer(task_type)
        if answer is not None:
            answering_types.append(str(task_type))
        # Every task may carry a scope, stored in task_spans under its own field name.
        span_rows.append((str(task_type), "scope"))
        for name in required_payload_fields(task_type):
            required_rows.append((str(task_type), name))
        for declared in task_payload(task_type):
            if declared is answer:
                continue
            kind = "element" if declared.stores_elements else declared.kind.value
            field_rows.append((str(task_type), declared.name, kind))
            if declared.kind is PayloadKind.SPAN:
                span_rows.append((str(task_type), declared.name))
            elif declared.kind is PayloadKind.RECORD_REF:
                ref_rows.append((str(task_type), declared.name, "record"))
            elif declared.kind is PayloadKind.TIME_SERIES_REF:
                ref_rows.append((str(task_type), declared.name, "time_series"))

    declared_fields = [(task_type, name) for task_type, name, _ in field_rows]
    answering = ", ".join(f"'{name}'" for name in answering_types)
    checks = [
        (
            "a task stores a payload field its type does not declare",
            f"SELECT t.external_id FROM task_fields f JOIN tasks t ON t.task_id = f.task_id "  # noqa: S608
            f"ANTI JOIN (VALUES {_rows(declared_fields)}) AS d(task_type, field) "
            f"ON d.task_type = t.task_type AND d.field = f.field",
        ),
        (
            "a task stores a payload field in the wrong column",
            f"SELECT t.external_id FROM task_fields f JOIN tasks t ON t.task_id = f.task_id "  # noqa: S608
            f"JOIN (VALUES {_rows(field_rows)}) AS d(task_type, field, kind) "
            f"ON d.task_type = t.task_type AND d.field = f.field WHERE "
            f"(d.kind = '{PayloadKind.TEXT.value}' AND (f.text_value IS NULL OR f.double_value IS NOT NULL)) OR "
            f"(d.kind = '{PayloadKind.NUMBER.value}' AND (f.double_value IS NULL OR f.text_value IS NOT NULL)) OR "
            f"(d.kind = 'element' AND (f.text_value IS NOT NULL OR f.double_value IS NOT NULL))",
        ),
        (
            "a task stores a payload span its type does not declare",
            f"SELECT t.external_id FROM task_spans s JOIN tasks t ON t.task_id = s.task_id "  # noqa: S608
            f"ANTI JOIN (VALUES {_rows(span_rows)}) AS d(task_type, field) "
            f"ON d.task_type = t.task_type AND d.field = s.field",
        ),
        (
            "a task stores a text answer its type does not declare",
            f"SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "  # noqa: S608
            f"WHERE i.role = 'target' AND i.item_type = 'text' AND t.task_type NOT IN ({answering})",
        ),
        (
            "a task stores more than one text answer",
            "SELECT task_id FROM task_items WHERE role = 'target' AND item_type = 'text' "
            "GROUP BY task_id HAVING count(*) > 1",
        ),
    ]
    if ref_rows:
        checks.append(
            (
                "a task stores a payload reference its type does not declare",
                f"SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "  # noqa: S608
                f"ANTI JOIN (VALUES {_rows(ref_rows)}) AS d(task_type, field, ref_kind) "
                f"ON d.task_type = t.task_type AND d.field = p.field AND d.ref_kind = p.ref_kind",
            )
        )
    if required_rows:
        checks.append(
            (
                "a task is missing a payload field its type requires",
                f"SELECT t.external_id FROM tasks t "  # noqa: S608
                f"JOIN (VALUES {_rows(required_rows)}) AS d(task_type, field) ON d.task_type = t.task_type "
                f"ANTI JOIN task_fields f ON f.task_id = t.task_id AND f.field = d.field",
            )
        )
    return tuple(checks)


VALIDATIONS: Final = (
    *(_dense_ids(table, column) for table, column in KEYED_TABLES.items()),
    *(_unique_external_id(table) for table in EXTERNAL_IDS),
    (
        "duplicate (record_id, time_series_id) link",
        "SELECT record_id, time_series_id FROM record_time_series "
        "GROUP BY record_id, time_series_id HAVING count(*) > 1",
    ),
    (
        "duplicate (time_series_id, chunk_idx)",
        "SELECT time_series_id, chunk_idx FROM time_series_chunks "
        "GROUP BY time_series_id, chunk_idx HAVING count(*) > 1",
    ),
    (
        "duplicate values artifact",
        "SELECT chunk_file FROM values_artifacts GROUP BY chunk_file HAVING count(*) > 1",
    ),
    (
        "duplicate specs.spec_type",
        "SELECT spec_type FROM specs GROUP BY spec_type HAVING count(*) > 1",
    ),
    (
        "duplicate annotation_descriptors.key",
        "SELECT key FROM annotation_descriptors GROUP BY key HAVING count(*) > 1",
    ),
    (
        "spec carries half a data source",
        "SELECT spec_type FROM specs WHERE (data_source_type IS NULL) <> (data_source_name IS NULL)",
    ),
    (
        "annotation names a key the schema does not declare",
        "SELECT n.external_id FROM annotations n ANTI JOIN annotation_descriptors d ON d.key = n.key",
    ),
    (
        "link names a record that does not exist",
        "SELECT l.record_id FROM record_time_series l ANTI JOIN records r ON r.record_id = l.record_id",
    ),
    (
        "link names a series that does not exist",
        "SELECT l.time_series_id FROM record_time_series l "
        "ANTI JOIN time_series s ON s.time_series_id = l.time_series_id",
    ),
    (
        "series names a spec that does not exist",
        "SELECT s.time_series_id FROM time_series s ANTI JOIN specs sp ON sp.spec_id = s.spec_id",
    ),
    (
        "series names an axis that does not exist",
        "SELECT s.time_series_id FROM time_series s ANTI JOIN axes a ON a.axis_id = s.axis_id",
    ),
    (
        "chunk names a series that does not exist",
        "SELECT c.time_series_id FROM time_series_chunks c "
        "ANTI JOIN time_series s ON s.time_series_id = c.time_series_id",
    ),
    (
        "chunk names an artifact the values plane did not declare",
        "SELECT c.artifact_id FROM time_series_chunks c ANTI JOIN values_artifacts a ON a.artifact_id = c.artifact_id",
    ),
    # The backend-neutral locator cannot say in its column types that a Parquet chunk needs both
    # indexes and a Zarr chunk needs only one. The artifact's backend says it instead, once the rows
    # are in.
    (
        "parquet chunk without a row offset",
        "SELECT a.chunk_file FROM time_series_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
        "WHERE a.backend = 'parquet' AND c.chunk_minor_idx IS NULL",
    ),
    (
        "zarr chunk with a row offset",
        "SELECT a.chunk_file FROM time_series_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
        "WHERE a.backend = 'zarr' AND c.chunk_minor_idx IS NOT NULL",
    ),
    (
        "series length disagrees with its chunks",
        "SELECT s.time_series_id FROM time_series s JOIN ("
        "SELECT time_series_id, sum(n_values) AS total FROM time_series_chunks GROUP BY time_series_id"
        ") c ON c.time_series_id = s.time_series_id WHERE c.total <> s.n_values",
    ),
    (
        "series has no chunk in the values plane",
        "SELECT s.time_series_id FROM time_series s "
        "ANTI JOIN time_series_chunks c ON c.time_series_id = s.time_series_id",
    ),
    *_attachment_targets(),
    *_task_payload_owners(),
    (
        "duplicate (record_id, task_id) link",
        "SELECT record_id, task_id FROM record_tasks GROUP BY record_id, task_id HAVING count(*) > 1",
    ),
    (
        "record_tasks names a record that does not exist",
        "SELECT l.record_id FROM record_tasks l ANTI JOIN records r ON r.record_id = l.record_id",
    ),
    # A resolved id column holds null where the caller's id named nothing, so each of these reports
    # the row that named it rather than the id that is missing.
    (
        "record names a task that does not exist",
        "SELECT r.external_id FROM record_tasks l JOIN records r ON r.record_id = l.record_id WHERE l.task_id IS NULL",
    ),
    (
        "task item names a record that does not exist",
        "SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "
        "WHERE i.item_type = 'record' AND i.record_id IS NULL",
    ),
    (
        "task item disagrees with its item_type",
        "SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "
        "WHERE (i.item_type = 'text' AND (i.text_value IS NULL OR i.record_id IS NOT NULL)) "
        "OR (i.item_type = 'record' AND i.text_value IS NOT NULL)",
    ),
    (
        "duplicate (task_id, role, item_type, position) item",
        "SELECT task_id, role, item_type, position FROM task_items "
        "GROUP BY task_id, role, item_type, position HAVING count(*) > 1",
    ),
    (
        "task payload names a record that does not exist",
        "SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "
        "WHERE p.ref_kind = 'record' AND p.ref_id IS NULL",
    ),
    (
        "task payload names a series that does not exist",
        "SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "
        "WHERE p.ref_kind = 'time_series' AND p.ref_id IS NULL",
    ),
    *_typed_task_payload(),
    (
        "an interval span ends before it starts",
        "SELECT task_id FROM task_spans WHERE end_at IS NOT NULL AND end_at <= start_at "
        "UNION ALL "
        "SELECT annotation_id FROM annotations WHERE span_end_us IS NOT NULL AND span_end_us <= span_start_us",
    ),
    (
        "annotation span bound without a start",
        "SELECT annotation_id FROM annotations WHERE span_start_us IS NULL AND span_end_us IS NOT NULL",
    ),
)
"""Every invariant the dropped key constraints used to enforce, as a query that must return no rows.

The writer runs these against the loaded database before it commits, which is the only moment they
can be violated: a version is written once by one process and is immutable afterwards. Each check is
one bulk anti-join rather than a lookup per row.

``task_from_tasks`` is deliberately unchecked, and is why that one table still keeps the caller's
string id. A streamed task skips the cross-task checks that
:meth:`~timenet.dataset.TimeFDataset.add_task` runs, so a dangling derivation can reach the writer.
Refusing it here would reject a write main accepted, and resolving it to a dense id would leave a
null that no longer says which task is missing. The reader reports it by name instead.
"""
