"""Shared fixed SQL fragments for public object identities and literal searches."""

OBJECT_ID_SELECTS = {
    "Dataset": "SELECT dataset_key AS object_key, 'Dataset' AS object_type, dataset_id AS object_id FROM datasets",
    "Record": "SELECT record_key AS object_key, 'Record' AS object_type, record_id AS object_id FROM records",
    "Source": "SELECT source_key AS object_key, 'Source' AS object_type, source_id AS object_id FROM sources",
    "Signal": "SELECT signal_key AS object_key, 'Signal' AS object_type, signal_id AS object_id FROM signals",
    "Task": "SELECT task_key AS object_key, 'Task' AS object_type, task_id AS object_id FROM tasks",
}
OBJECT_IDS_SQL = " UNION ALL ".join(OBJECT_ID_SELECTS.values())


def like_literal(value: str) -> str:
    """Escape a user string for a literal SQL LIKE match.

    Returns:
        The literal-safe LIKE fragment.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
