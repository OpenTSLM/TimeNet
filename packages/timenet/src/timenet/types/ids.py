"""Generate entity ids.

TimeF entity ids (records, series, annotations, tasks) default to a canonical UUIDv7 string. UUIDv7
(RFC 9562 §5.7) puts a 48-bit millisecond Unix timestamp in the high bits, so ids sort by creation
time. ``uuid.uuid7()`` arrives only in Python 3.14, so this module generates the value.
"""

import secrets
import time
import uuid


def uuid7() -> uuid.UUID:
    """Return a UUIDv7 (RFC 9562 §5.7): 48-bit ms timestamp, then random bits.

    Returns:
        A version-7 :class:`uuid.UUID`. Two calls in the same millisecond differ in their random bits.
    """
    unix_ms = time.time_ns() // 1_000_000
    value = (unix_ms & 0xFFFFFFFFFFFF) << 80  # 48-bit timestamp in the high bits
    value |= 0x7 << 76  # version 7
    value |= secrets.randbits(12) << 64  # rand_a
    value |= 0b10 << 62  # RFC 4122 variant
    value |= secrets.randbits(62)  # rand_b
    return uuid.UUID(int=value)


def new_id() -> str:
    """Return a new canonical UUIDv7 string, the default id for every TimeF entity.

    Returns:
        The canonical 36-character UUIDv7 string (time-ordered).
    """
    return str(uuid7())
