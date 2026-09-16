"""Identity: each row is named one time, and each surrogate id is dense.

Two rules make a row addressable. A surrogate id runs 0, 1, 2 with no gap and no repeat. A natural
key names one row and no more: the caller's id, a spec type, a descriptor key, a link, an item
position. The database declares no primary key and no unique index, so these checks stand in for
both.

The two generators below cover the repetitive part. Each keyed table gets the same density check,
and each table that keeps the caller's id gets the same duplicate check.
"""

from typing import Final

from timenet.control_plane.checks._check import Check
from timenet.control_plane.schema import EXTERNAL_IDS, KEYED_TABLES


def _dense_ids(table: str, column: str) -> Check:
    """Return the check that one table's ids are 0, 1, 2, with no gap and no repeat.

    A reader splits a corpus across workers with ``id % num_workers = worker_index``. That covers
    every row one time only while the ids run without a gap.

    Args:
        table: The table to check.
        column: Its surrogate id column.

    Returns:
        The named check.
    """
    return Check(
        name=f"{table}_ids_dense",
        invariant=f"{table}.{column} runs 0, 1, 2 with no gap and no repeat.",
        # A distinct count catches a repeat, and the row count against the largest id catches a
        # gap. Both are needed, because one repeat plus one gap leaves the row count unchanged.
        sql=(
            f"SELECT count(*) FROM {table} HAVING count(*) <> count(DISTINCT {column}) "  # noqa: S608
            f"OR count(*) - 1 <> max({column}) OR min({column}) <> 0"
        ),
        remedy=f"Report this build. The writer assigns {column}, so a gap is a defect of the loader.",
    )


def _unique_external_id(table: str) -> Check:
    """Return the check that no two rows in one table share an ``external_id``.

    Args:
        table: The table to check.

    Returns:
        The named check.
    """
    return Check(
        name=f"{table}_external_id_unique",
        invariant=f"No two rows of {table} have the same external_id.",
        sql=f"SELECT external_id FROM {table} GROUP BY external_id HAVING count(*) > 1",  # noqa: S608
        remedy=f"Give each entry of {table} its own id. Add an entity one time only.",
    )


IDENTITY: Final = (
    *(_dense_ids(table, column) for table, column in KEYED_TABLES.items()),
    *(_unique_external_id(table) for table in EXTERNAL_IDS),
    Check(
        name="record_time_series_link_unique",
        invariant="A record names each of its series one time.",
        sql=(
            "SELECT record_id, time_series_id FROM record_time_series "
            "GROUP BY record_id, time_series_id HAVING count(*) > 1"
        ),
        remedy="Add each series to a record one time only.",
    ),
    Check(
        name="time_series_chunks_index_unique",
        invariant="Each chunk index of a series belongs to one chunk.",
        sql=(
            "SELECT time_series_id, chunk_idx FROM time_series_chunks "
            "GROUP BY time_series_id, chunk_idx HAVING count(*) > 1"
        ),
        remedy="Report this build. The values plane placed two chunks at one index.",
    ),
    Check(
        name="values_artifacts_file_unique",
        invariant="Each file of the values plane is declared one time.",
        sql="SELECT chunk_file FROM values_artifacts GROUP BY chunk_file HAVING count(*) > 1",
        remedy="Report this build. The values plane declared one file twice.",
    ),
    Check(
        name="specs_type_unique",
        invariant="Each spec type belongs to one spec.",
        sql="SELECT spec_type FROM specs GROUP BY spec_type HAVING count(*) > 1",
        remedy="Two specs share a type. Make them equal, or give each one its own type.",
    ),
    Check(
        name="annotation_descriptors_key_unique",
        invariant="Each annotation key is declared one time in the schema.",
        sql="SELECT key FROM annotation_descriptors GROUP BY key HAVING count(*) > 1",
        remedy="Declare each annotation key one time in the dataset schema.",
    ),
    Check(
        name="record_tasks_link_unique",
        invariant="A task names each of its records one time.",
        sql="SELECT record_id, task_id FROM record_tasks GROUP BY record_id, task_id HAVING count(*) > 1",
        remedy="List each record one time in the record ids of the task.",
    ),
    Check(
        name="task_items_position_unique",
        invariant="Each role, item type and position of a task holds one item.",
        sql=(
            "SELECT task_id, role, item_type, position FROM task_items "
            "GROUP BY task_id, role, item_type, position HAVING count(*) > 1"
        ),
        remedy="Give each item of a task its own position.",
    ),
)
"""Each row is addressable: the surrogate ids are dense, and no natural key names two rows."""
