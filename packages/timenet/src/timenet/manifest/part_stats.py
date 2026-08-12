"""The :class:`PartStat`: per-part skip metadata for a sharded control table."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PartStat:
    """One control-table part: its path, row count, and first/last sort key.

    ``first_key`` and ``last_key`` hold the part's boundary key in the encoded id space (raw bytes for
    a ``uuid16`` column, the string otherwise), or ``None`` for an empty part.
    """

    path: str
    n_rows: int
    first_key: tuple[object, ...] | None
    last_key: tuple[object, ...] | None
