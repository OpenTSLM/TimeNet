"""An LRU of loaded control-table parts, so a reader keeps only touched parts resident."""

from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class _PartCache:
    """Bounded cache of parsed Parquet parts, keyed by relative path (most-recent-first)."""

    def __init__(self, root: Path, capacity: int) -> None:
        """Bind the cache to a version root and a maximum number of resident parts.

        Args:
            root: The version directory; parts resolve relative to it.
            capacity: The most parts to keep resident.
        """
        self._root = root
        self._capacity = max(1, capacity)
        self._parts: OrderedDict[str, pa.Table] = OrderedDict()

    def get(self, rel: str) -> pa.Table:
        """Return the parsed part at ``rel``, loading and caching it on a miss.

        Args:
            rel: The part's path relative to the version root.

        Returns:
            The part as an Arrow table.
        """
        table = self._parts.get(rel)
        if table is None:
            table = pq.read_table(self._root / rel)
            self._parts[rel] = table
            if len(self._parts) > self._capacity:
                self._parts.popitem(last=False)
        else:
            self._parts.move_to_end(rel)
        return table
