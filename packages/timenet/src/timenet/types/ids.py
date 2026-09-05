"""Generate entity ids.

TimeF entity ids (records, series, annotations, tasks) default to a canonical UUIDv7 string. UUIDv7
(RFC 9562 §5.7) puts a 48-bit millisecond Unix timestamp in the high bits, so ids sort by creation
time. The writer sorts rows by id. The shared timestamp prefix then forms long runs that compress
well, and the writer can store canonical-UUID id columns as 16 raw bytes. ``uuid.uuid7()`` arrives
only in Python 3.14, so this module generates the value.
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


def is_canonical_uuid(value: str) -> bool:
    """Return whether ``value`` is a canonical UUID string that round-trips exactly.

    Only canonical values qualify for compact 16-byte storage. A canonical value is lowercase and
    hyphenated, exactly as :func:`str` renders a :class:`uuid.UUID`. Its round-trip back to a string
    is byte-for-byte identical.

    Args:
        value: The id to test.

    Returns:
        ``True`` if ``value`` is a canonical UUID string.
    """
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


def id_to_bytes(value: str) -> bytes:
    """Return the 16 raw bytes of a canonical-UUID id string.

    Args:
        value: A canonical UUID string.

    Returns:
        The UUID's 16-byte representation.
    """
    return uuid.UUID(value).bytes


def id_from_bytes(raw: bytes) -> str:
    """Return the canonical UUID string for 16 raw bytes.

    Args:
        raw: A 16-byte UUID.

    Returns:
        The canonical UUID string.
    """
    return str(uuid.UUID(bytes=raw))
