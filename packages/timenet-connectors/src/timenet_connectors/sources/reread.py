"""Compare validated source rereads with the content checked during conversion."""

from collections.abc import Callable
from typing import TypeVar

from timenet.errors import TimeNetConnectorError


_Payload = TypeVar("_Payload")
_Fingerprint = TypeVar("_Fingerprint")


def checked_reread(
    load: Callable[[], _Payload],
    *,
    fingerprint: Callable[[_Payload], _Fingerprint],
    expected: _Fingerprint,
    context: str,
) -> _Payload:
    """Return a validated payload only if its fingerprint matches the conversion scan.

    The caller owns decoding, validation, source locations, and fingerprint semantics.
    Fingerprints must compare by value with a scalar boolean result. Neither callback's
    errors are intercepted: a failed read does not establish that the content changed.
    This function does not cache or copy the payload, or check a partially consumed stream.

    Raises:
        TimeNetConnectorError: If the validated content differs from the scan.
    """
    payload = load()
    if fingerprint(payload) != expected:
        raise TimeNetConnectorError(f"{context}: content differs from the conversion scan")
    return payload
