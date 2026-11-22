"""A bounded cache for connector-owned physical reads."""

from collections.abc import Callable, Hashable
from threading import Lock
from typing import Generic, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class SharedReadCache(Generic[K, V]):
    """Cache one decoded physical unit within a conversion.

    Keys identify physical units by equality and must be stable and unique within this
    cache, including any decoder parameters that affect the result. Use the same key
    and cache for callbacks in different logical sources that read the same unit.
    Create a new cache for each conversion. Readers for the same key must be equivalent.

    Reads and decoding are serialized. A reader must not call this cache recursively:
    the lock is not reentrant. On a miss, the old entry is released before decoding;
    failures propagate and leave the cache empty. Callers may retain returned values,
    so this bounds only the cache's own reference, not total memory.
    """

    def __init__(self) -> None:
        self._entry: tuple[K, V] | None = None
        self._lock = Lock()

    def get(self, key: K, reader: Callable[[], V]) -> V:
        """Return the unit for key, decoding it on a miss.

        Returns:
            The cached or newly decoded unit.
        """
        with self._lock:
            if self._entry is not None and self._entry[0] == key:
                return self._entry[1]
            self._entry = None
            value = reader()
            self._entry = (key, value)
            return value
