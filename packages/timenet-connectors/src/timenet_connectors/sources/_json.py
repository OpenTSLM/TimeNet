"""JSON decoding rules shared by source readers."""

from timenet.errors import TimeNetConnectorError


def distinct_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject repeated fields instead of silently overwriting source data.

    Returns:
        The decoded object in source order.

    Raises:
        TimeNetConnectorError: If an object repeats a field name.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TimeNetConnectorError(f"duplicate key {key!r}")
        result[key] = value
    return result
