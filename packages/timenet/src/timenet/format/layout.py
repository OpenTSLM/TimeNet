"""Values-plane byte targets, chosen by the shape of a corpus' training item.

A corpus whose item is a whole record and a corpus whose item is one window of a record want
opposite layouts, so no single target serves both. A connector declares the layout it wants and the
writer applies it. Neither side imports the other.
"""

from dataclasses import dataclass

from timenet.errors import TimeFValidationError
from timenet.format.constants import DEFAULT_CHUNK_MAX_BYTES, DEFAULT_ROW_GROUP_TARGET_BYTES


@dataclass(frozen=True)
class ValuesLayout:
    """The chunk and row-group byte targets a writer applies to the values plane.

    The chunk target bounds how much a reader decodes to serve one item. The row-group target
    bounds how much it must fetch to reach that chunk. Both matter, so set them together.
    """

    chunk_max_bytes: int
    """Maximum uncompressed values size of one logical chunk."""
    row_group_target_bytes: int
    """Flush a row group once its buffered values pass this size."""

    def __post_init__(self) -> None:
        """Reject a target that no writer can honor.

        Raises:
            TimeFValidationError: If either target is not positive.
        """
        if self.chunk_max_bytes <= 0 or self.row_group_target_bytes <= 0:
            raise TimeFValidationError(
                "values layout targets must be positive, got "
                f"chunk_max_bytes={self.chunk_max_bytes} and "
                f"row_group_target_bytes={self.row_group_target_bytes}"
            )


DEFAULT_VALUES_LAYOUT = ValuesLayout(
    chunk_max_bytes=DEFAULT_CHUNK_MAX_BYTES,
    row_group_target_bytes=DEFAULT_ROW_GROUP_TARGET_BYTES,
)
"""1 MiB chunks in 4 MiB row groups, for a corpus whose item is a whole record.

This is what the writer has always written, and the layout to keep when in doubt. Smaller chunks
help only the shuffled read of a windowed item. They make every other read, and the size on disk,
worse.
"""

WINDOWED_VALUES_LAYOUT = ValuesLayout(chunk_max_bytes=64 * 2**10, row_group_target_bytes=512 * 2**10)
"""64 KiB chunks in 512 KiB row groups, for a corpus whose item is one window of a record.

The small chunks let a reader decode one window instead of the record around it, which is what a
shuffled read does on every item. The price is a larger version on disk and a slower sequential or
whole-record read.
"""
