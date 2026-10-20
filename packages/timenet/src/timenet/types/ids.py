"""Generate entity ids.

TimeF entity ids (records, series, tasks, and the storage ids the writer gives other rows) default to
a canonical UUIDv7 string. UUIDv7 (RFC 9562 §5.7) puts a 48-bit millisecond Unix timestamp in the
high bits, so ids sort by creation time. Within one process they sort in the order they were made.
``uuid.uuid7()`` arrives only in Python 3.14, so this module generates the value.
"""

import secrets
import time
from typing import TypeAlias
import uuid


EntityId: TypeAlias = str
"""The id of one TimeF entity, a UUIDv7 string."""


_RANDOM_BITS = 74
_last_uuid7 = (0, 0)
"""The millisecond and random bits of the last UUIDv7 this process made."""


def uuid7() -> uuid.UUID:
    """Return a UUIDv7 (RFC 9562 §5.7): 48-bit ms timestamp, then random bits.

    Within one millisecond, each call adds one to the random bits of the previous call (RFC 9562
    §6.2, method 1). So the UUIDs this process makes sort in the order it made them.

    Returns:
        A version-7 :class:`uuid.UUID`, greater than every earlier one from this process.
    """
    global _last_uuid7  # noqa: PLW0603 - the process-wide ordering state
    unix_ms = time.time_ns() // 1_000_000
    random = secrets.randbits(_RANDOM_BITS)
    last_ms, last_random = _last_uuid7
    if unix_ms <= last_ms:
        unix_ms, random = divmod((last_ms << _RANDOM_BITS) + last_random + 1, 1 << _RANDOM_BITS)
    _last_uuid7 = (unix_ms, random)
    value = (unix_ms & 0xFFFFFFFFFFFF) << 80  # 48-bit timestamp in the high bits
    value |= 0x7 << 76  # version 7
    value |= (random >> 62) << 64  # rand_a
    value |= 0b10 << 62  # RFC 4122 variant
    value |= random & ((1 << 62) - 1)  # rand_b
    return uuid.UUID(int=value)


def new_id() -> str:
    """Return a new canonical UUIDv7 string, the default id for every TimeF entity.

    Returns:
        The canonical 36-character UUIDv7 string (time-ordered).
    """
    return str(uuid7())
