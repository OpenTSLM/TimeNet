"""Validation of stored irregular timestamp arrays."""

import pyarrow as pa
import pyarrow.compute as pc

from timenet.errors import TimeFFormatError
from timenet.types.payloads import PayloadDescriptor


def validate_offsets(offsets: pa.Array, descriptor: PayloadDescriptor) -> None:
    """Reject timestamps that violate their declared length, endpoints, or ordering.

    Raises:
        TimeFFormatError: If the timestamp array is inconsistent.
    """
    if offsets.type != pa.int64() or offsets.null_count or len(offsets) != descriptor.length:
        raise TimeFFormatError("timestamp payload has an invalid dtype, null value, or length")
    if not len(offsets) or (offsets[0].as_py(), offsets[-1].as_py()) != (descriptor.first_us, descriptor.last_us):
        raise TimeFFormatError("timestamp payload has offsets disagreeing with its axis endpoints")
    if len(offsets) > 1:
        decreasing = pc.less(offsets.slice(1), offsets.slice(0, len(offsets) - 1))  # ty: ignore[unresolved-attribute]
        if pc.any(decreasing).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFFormatError("timestamp payload has decreasing offsets")
