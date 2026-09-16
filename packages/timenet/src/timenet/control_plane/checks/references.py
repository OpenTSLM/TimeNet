"""References: every id that names another row names a row that is there.

These are the foreign keys the shipped database does not declare. Each check is one anti-join
against the finished load.

A reference reaches this point in one of two shapes. A row that names an entity by its surrogate id
is dangling when the anti-join finds no target. A row that named an entity by the caller's id was
resolved by a join at the end of the walk, and an id that named nothing became null. The checks in
the second group therefore look for a null, and they report the row that holds it. The id itself is
gone by then, so the message names the record or the task that used it.

The two generators below cover the repetitive part: one anti-join per attachment table, and one per
table that holds a piece of a task payload.
"""

from typing import Final

from timenet.control_plane.checks._check import Check
from timenet.control_plane.schema import ANNOTATION_TABLES


def _attachment_targets() -> tuple[Check, ...]:
    """Return the reference checks for every attachment table.

    Each is a bare anti-join with no ``WHERE``, because a table-per-kind layout makes every target
    column ``NOT NULL``.

    Returns:
        The named checks, two per table that names a target and one for the rest.
    """
    entity_of = {"record": "records", "task": "tasks"}
    checks: list[Check] = []
    for kind, (table, column) in ANNOTATION_TABLES.items():
        checks.append(
            Check(
                name=f"{table}_annotation_exists",
                invariant=f"Every attachment in {table} names an annotation the version holds.",
                sql=(
                    f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                    f"ANTI JOIN annotations n ON n.annotation_id = a.annotation_id"
                ),
                remedy="Report this build. The loader attached an annotation that it did not store.",
            )
        )
        if column is not None:
            checks.append(
                Check(
                    name=f"{table}_{kind}_exists",
                    invariant=f"Every attachment in {table} names a {kind} the version holds.",
                    sql=(
                        f"SELECT a.attachment_id FROM {table} a "  # noqa: S608
                        f"ANTI JOIN {entity_of[kind]} e ON e.{column} = a.{column}"
                    ),
                    remedy=f"Report this build. The loader attached an annotation to a {kind} that it did not store.",
                )
            )
    return tuple(checks)


def _task_payload_owners() -> tuple[Check, ...]:
    """Return the check that every payload row belongs to a task that exists.

    Returns:
        One named check per payload table.
    """
    return tuple(
        Check(
            name=f"{table}_task_exists",
            invariant=f"Every row of {table} belongs to a task the version holds.",
            sql=f"SELECT p.task_id FROM {table} p ANTI JOIN tasks t ON t.task_id = p.task_id",  # noqa: S608
            remedy="Report this build. The loader wrote a payload row for a task that it did not store.",
        )
        for table in ("task_items", "task_from_tasks", "task_fields", "task_refs", "task_spans")
    )


REFERENCES: Final = (
    Check(
        name="annotations_key_declared",
        invariant="Every annotation has a key the dataset schema declares.",
        sql="SELECT n.external_id FROM annotations n ANTI JOIN annotation_descriptors d ON d.key = n.key",
        remedy="Declare the key in the dataset schema, or correct the key on the annotation.",
    ),
    Check(
        name="record_time_series_record_exists",
        invariant="Every record-to-series link names a record the version holds.",
        sql="SELECT l.record_id FROM record_time_series l ANTI JOIN records r ON r.record_id = l.record_id",
        remedy="Add the record to the dataset before you give it a series.",
    ),
    Check(
        name="record_time_series_series_exists",
        invariant="Every record-to-series link names a series the version holds.",
        sql=(
            "SELECT l.time_series_id FROM record_time_series l "
            "ANTI JOIN time_series s ON s.time_series_id = l.time_series_id"
        ),
        remedy="Add the series to the record before you write the dataset.",
    ),
    Check(
        name="time_series_spec_exists",
        invariant="Every series names a spec the version declares.",
        sql="SELECT s.time_series_id FROM time_series s ANTI JOIN specs sp ON sp.spec_id = s.spec_id",
        remedy="Give the series a spec, and derive the dataset schema before you write.",
    ),
    Check(
        name="time_series_axis_exists",
        invariant="Every series names an axis the version holds.",
        sql="SELECT s.time_series_id FROM time_series s ANTI JOIN axes a ON a.axis_id = s.axis_id",
        remedy="Report this build. The loader stored a series without its time axis.",
    ),
    Check(
        name="time_series_chunks_series_exists",
        invariant="Every chunk belongs to a series the version holds.",
        sql=(
            "SELECT c.time_series_id FROM time_series_chunks c "
            "ANTI JOIN time_series s ON s.time_series_id = c.time_series_id"
        ),
        remedy="Report this build. The values plane wrote a chunk for a series that is not stored.",
    ),
    *_attachment_targets(),
    *_task_payload_owners(),
    Check(
        name="record_tasks_record_exists",
        invariant="Every record-to-task link names a record the version holds.",
        sql="SELECT l.record_id FROM record_tasks l ANTI JOIN records r ON r.record_id = l.record_id",
        remedy="Add the record to the dataset before a task names it.",
    ),
    # The checks below read a resolved id column, which is null where the caller's id named
    # nothing. Each one reports the row that named it, because the id itself is no longer there.
    Check(
        name="record_tasks_task_resolved",
        invariant="Every task id that a record names belongs to a task the version holds.",
        sql=(
            "SELECT r.external_id FROM record_tasks l JOIN records r ON r.record_id = l.record_id "
            "WHERE l.task_id IS NULL"
        ),
        remedy="Add the task to the dataset, or remove its id from the record.",
    ),
    Check(
        name="task_items_record_resolved",
        invariant="Every record item of a task names a record the version holds.",
        sql=(
            "SELECT t.external_id FROM task_items i JOIN tasks t ON t.task_id = i.task_id "
            "WHERE i.item_type = 'record' AND i.record_id IS NULL"
        ),
        remedy="Add the record to the dataset, or remove its id from the task.",
    ),
    Check(
        name="task_refs_record_resolved",
        invariant="Every record that a task payload references is a record the version holds.",
        sql=(
            "SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "
            "WHERE p.ref_kind = 'record' AND p.ref_id IS NULL"
        ),
        remedy="Add the record to the dataset, or remove its id from the payload of the task.",
    ),
    Check(
        name="task_refs_series_resolved",
        invariant="Every series that a task payload references is a series the version holds.",
        sql=(
            "SELECT t.external_id FROM task_refs p JOIN tasks t ON t.task_id = p.task_id "
            "WHERE p.ref_kind = 'time_series' AND p.ref_id IS NULL"
        ),
        remedy="Add the series to the dataset, or remove its id from the payload of the task.",
    ),
)
"""Every id that names another row finds one.

``task_from_tasks`` has no check here, and that one table keeps the caller's id as a string. A
streamed task skips the cross-task checks that :meth:`~timenet.dataset.TimeFDataset.add_task` runs,
so a dangling derivation can reach the writer. A surrogate id here resolves to null, which loses the
name of the missing task. The string keeps the name, so the reader can report it.
"""
