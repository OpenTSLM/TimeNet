"""Values-plane byte targets, chosen by the shape of a corpus' training item.

A corpus whose item is a whole record and a corpus whose item is one window of a record want
opposite layouts. The gap is large enough that one global value makes one of the two much worse.
The connector contract declares a layout and the writer applies it, and neither imports the other.

These targets are byte budgets, not durations. A per-sample-rate budget measured slightly better
again on the windowed corpus, 22.3 s against 26.5 s shuffled. It needs a writer concept that does
not exist, and :data:`WINDOWED_VALUES_LAYOUT` is within 20 per cent of it.
"""

from dataclasses import dataclass

from timenet.errors import TimeFValidationError
from timenet.format.constants import DEFAULT_CHUNK_MAX_BYTES, DEFAULT_ROW_GROUP_TARGET_BYTES


@dataclass(frozen=True)
class ValuesLayout:
    """The chunk and row-group byte targets a writer applies to the values plane.

    The chunk target bounds how much a reader decodes to serve one item. The row-group target
    bounds how much it must fetch to reach that chunk. An isolation run puts the cost of a shuffled
    read in the row group: 64 KiB chunks inside 4 MiB row groups still cost 87.7 s, where 1 MiB
    chunks inside 1 MiB row groups cost 48.9 s.
    """

    chunk_max_bytes: int
    """Maximum uncompressed values size of one logical chunk."""
    row_group_target_bytes: int
    """Flush a row group once its buffered values pass this size."""

    def __post_init__(self) -> None:
        """Reject a target that no writer can honour.

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

This is what the writer has always written. Measured on an ECG-QA-shaped corpus, one 12-lead record
per item. Every layout read within noise sequentially and disk stayed flat between 80 and 89 MB.
The shuffled read prefers the coarse layout: 0.56 s here against 1.56 s at 64 KiB / 512 KiB.

This is also the layout to keep for a corpus that has not been measured. Everything except the
shuffled read of a windowed item gets 20 to 25 per cent worse as the chunks shrink.
"""

WINDOWED_VALUES_LAYOUT = ValuesLayout(chunk_max_bytes=64 * 2**10, row_group_target_bytes=512 * 2**10)
"""64 KiB chunks in 512 KiB row groups, for a corpus whose item is one window of a record.

Measured on a Sleep-EDF-shaped corpus of 40 overnight recordings, one scored 30 s epoch per item.
Against :data:`DEFAULT_VALUES_LAYOUT`, the shuffled read drops from 115 s to 26.5 s and one epoch
decodes 3.9 MB instead of 32.1 MB.

It costs 824 MB on disk instead of 684 MB. The sequential read goes from 2.38 s to 2.89 s and the
whole-record read from 2.02 s to 2.51 s.
"""
