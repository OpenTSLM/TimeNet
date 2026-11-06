"""Share decoded recordings between lazy channel loaders with a bounded cache."""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from threading import Lock
from typing import Generic, TypeVar

import pyarrow as pa

from timenet.errors import TimeFValidationError


RecordingKey = TypeVar("RecordingKey", bound=Hashable)


class SharedChannelLoader(Generic[RecordingKey]):
    """Share one recording read across its channels, within a conversion.

    ``read_channels`` validates the source and returns immutable Arrow arrays in channel order. Channels
    may have different lengths and types. This helper does not cast values or check source changes.
    Keys must identify the source and all options affecting its decoded values.

    The cache retains at most ``max_recordings`` recordings, not a fixed number of bytes. On a miss,
    it evicts the least recently used recording before decoding its replacement. A failed read
    leaves that entry evicted and is retried on the next request. Arrays held by callers may
    outlive eviction, so this is not a bound on total process memory.

    Reads and cache updates are serialized. The reader must not call back into this loader.
    """

    def __init__(self, read_channels: Callable[[RecordingKey], Sequence[pa.Array]], *, max_recordings: int = 1) -> None:
        if isinstance(max_recordings, bool) or not isinstance(max_recordings, int) or max_recordings < 1:
            raise TimeFValidationError("max_recordings must be a positive integer")
        self._read_channels = read_channels
        self._max_recordings = max_recordings
        self._cache: OrderedDict[RecordingKey, tuple[pa.Array, ...]] = OrderedDict()
        self._lock = Lock()

    def channel_loader(self, recording_key: RecordingKey, channel_index: int) -> Callable[[], pa.Array]:
        """Return a lazy channel callback suitable for ``Signal.from_loader``.

        Args:
            recording_key: The recording key passed to the reader.
            channel_index: The zero-based position in the channels returned by the reader.

        Returns:
            A no-argument callback that returns the channel's Arrow array.

        Raises:
            TimeFValidationError: If the channel index is invalid. The callback also raises this when
                the reader returns invalid channels or the requested channel does not exist.
        """
        if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
            raise TimeFValidationError("channel index must be a nonnegative integer")

        def load() -> pa.Array:
            with self._lock:
                channels = self._channels(recording_key)
                if channel_index >= len(channels):
                    raise TimeFValidationError(
                        f"{recording_key}: channel {channel_index} is out of range for {len(channels)} channels"
                    )
                return channels[channel_index]

        return load

    def _channels(self, recording_key: RecordingKey) -> tuple[pa.Array, ...]:
        """Return cached or freshly decoded channels while the caller holds the lock.

        Raises:
            TimeFValidationError: If the reader does not return nonempty Arrow channels.
        """
        if recording_key not in self._cache:
            if len(self._cache) == self._max_recordings:
                self._cache.popitem(last=False)
            decoded_channels = self._read_channels(recording_key)
            if (
                not isinstance(decoded_channels, Sequence)
                or not decoded_channels
                or any(not isinstance(channel, pa.Array) for channel in decoded_channels)
            ):
                raise TimeFValidationError(f"{recording_key}: the reader must return one or more Arrow channel arrays")
            self._cache[recording_key] = tuple(decoded_channels)
        self._cache.move_to_end(recording_key)
        return self._cache[recording_key]

    def clear(self) -> None:
        """Release cached recordings; existing callbacks remain usable and read them again."""
        with self._lock:
            self._cache.clear()
