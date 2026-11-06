"""Share decoded recordings between lazy channel loaders with a bounded cache."""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from threading import Lock
from typing import Generic, TypeVar

import pyarrow as pa

from timenet.errors import TimeFValidationError


Key = TypeVar("Key", bound=Hashable)


class SharedChannelLoader(Generic[Key]):
    """Share one recording read across its channels, within a conversion.

    ``read`` validates the source and returns immutable Arrow arrays in channel order. Channels
    may have different lengths and types. This helper does not cast values or check source changes.
    Keys must identify the source and all options affecting its decoded values.

    The cache retains at most ``max_records`` recordings, not a fixed number of bytes. On a miss,
    it evicts the least recently used recording before decoding its replacement. A failed read
    leaves that entry evicted and is retried on the next request. Arrays held by callers may
    outlive eviction, so this is not a bound on total process memory.

    Reads and cache updates are serialized. The reader must not call back into this loader.
    """

    def __init__(self, read: Callable[[Key], Sequence[pa.Array]], *, max_records: int = 1) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise TimeFValidationError("max_records must be a positive integer")
        self._read = read
        self._max_records = max_records
        self._cache: OrderedDict[Key, tuple[pa.Array, ...]] = OrderedDict()
        self._lock = Lock()

    def channel(self, key: Key, index: int) -> Callable[[], pa.Array]:
        """Return a lazy channel callback suitable for ``Signal.from_loader``.

        Args:
            key: The recording key passed to the reader.
            index: The zero-based channel position in the reader's result.

        Returns:
            A no-argument callback that returns the channel's Arrow array.

        Raises:
            TimeFValidationError: If the index is invalid. The callback also raises this when
                the reader returns invalid channels or the requested channel does not exist.
        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TimeFValidationError("channel index must be a nonnegative integer")

        def load() -> pa.Array:
            with self._lock:
                if key not in self._cache:
                    if len(self._cache) == self._max_records:
                        self._cache.popitem(last=False)
                    result = self._read(key)
                    if (
                        not isinstance(result, Sequence)
                        or not result
                        or any(not isinstance(channel, pa.Array) for channel in result)
                    ):
                        raise TimeFValidationError(f"{key}: the reader must return one or more Arrow channel arrays")
                    self._cache[key] = tuple(result)
                self._cache.move_to_end(key)
                channels = self._cache[key]
                if index >= len(channels):
                    raise TimeFValidationError(f"{key}: channel {index} is out of range for {len(channels)} channels")
                return channels[index]

        return load

    def clear(self) -> None:
        """Release cached recordings; existing callbacks remain usable and read them again."""
        with self._lock:
            self._cache.clear()
