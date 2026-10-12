"""Projection budgets applied before materializing large variable-width fields."""

from timenet.errors import TimeFValidationError


# A page has at most 200 rows. Keeping each text field below 16 KiB leaves headroom
# under the viewer's 8 MiB transport cap, including JSON escaping and row metadata.
MAX_TEXT = 4096
MAX_METADATA = 16384
MAX_RELATIONSHIPS = 200


def validate_page(limit: int, offset: int = 0) -> int:
    """Validate shared public inspection pagination.

    Returns:
        The validated page size.

    Raises:
        TimeFValidationError: If the size or offset is outside the supported range.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RELATIONSHIPS:
        raise TimeFValidationError("limit must be between 1 and 200")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise TimeFValidationError("offset must be a nonnegative integer")
    return limit


def bounded_text(value: str | None) -> str | None:
    """Return a bounded preview, preserving absence."""
    return None if value is None else value[:MAX_TEXT]


def annotation_value_sql(alias: str = "") -> str:
    """Return a bounded display projection for every supported annotation value kind.

    The prefix comes only from fixed SQL in the reader, never from request input.
    """
    fields = ("text_value", "integer_value", "float_value", "boolean_value", "text_list_value")
    values = ", ".join(f"cast({alias}{name} AS VARCHAR)" for name in fields)
    return f"substring(coalesce({values}), 1, {MAX_TEXT + 1})"
