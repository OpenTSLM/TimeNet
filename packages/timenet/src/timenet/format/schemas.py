"""The Arrow schema of a TimeF values shard, parameterized by id storage type.

A shard's column layout is fixed. Only its ``time_series_id`` column varies: it is stored as
``pa.string()``, or as ``pa.binary(16)`` when every value is a canonical UUID. ``pa.binary(16)`` uses
16 raw bytes instead of a 36-character string. The writer picks the type from the data and writes it
into the shard schema. ``binary(16)`` decodes back to the canonical string, so callers always see
string ids.

Every other id lives in the control database as the caller's own string. The shard is the one place
where a logical id still needs a storage decision.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types.ids import id_to_bytes


#: The logical ids that cross-reference TimeF records, series, annotations and tasks. A shard
#: column holds ``time_series_id``.
LOGICAL_IDS: tuple[str, ...] = (
    "record_id",
    "time_series_id",
    "annotation_id",
    "task_id",
    "source_id",
    "subject_id",
)

#: The compact on-disk form for a canonical-UUID id column: 16 raw bytes.
UUID16 = pa.binary(16)

IdTypes = dict[str, pa.DataType]


def default_id_types() -> IdTypes:
    """Return the all-``string`` id types.

    Returns:
        A mapping from every logical id to ``pa.string()``.
    """
    return {name: pa.string() for name in LOGICAL_IDS}


#: The default values-element type of a waveform shard. A module-level constant keeps
#: :func:`shard_schema`'s default free of a function call in its signature.
_DEFAULT_VALUES_TYPE = pa.float32()


def shard_schema(id_types: IdTypes, value_type: pa.DataType = _DEFAULT_VALUES_TYPE) -> pa.Schema:
    """Return the waveform shard schema.

    Args:
        id_types: The resolved id storage types.
        value_type: The element type of the ``values`` list (defaults to ``float32``).

    Returns:
        The Arrow schema.
    """
    return pa.schema(
        [
            ("time_series_id", id_types["time_series_id"]),
            ("spec_type", pa.string()),
            ("signal", pa.string()),
            ("chunk_idx", pa.int32()),
            ("n_values", pa.int32()),
            ("values", pa.list_(value_type)),
            ("time_offsets_us", pa.list_(pa.int64())),
        ]
    )


@dataclass(frozen=True)
class IdCodec:
    """Convert ids between their in-memory strings and their on-disk form.

    This is the one place that holds the ``str -> bytes`` conversion for a ``uuid16`` column. The
    writer builds a codec (:meth:`from_uuid16` or :meth:`from_id_types`) and the values plane calls
    it. The two halves of the shard contract cannot drift apart.
    """

    uuid16: frozenset[str]
    """The logical ids whose columns are stored as ``binary(16)``. The rest are ``pa.string()``."""

    @classmethod
    def from_uuid16(cls, uuid16: Iterable[str]) -> "IdCodec":
        """Build a codec from the set of logical ids stored as ``binary(16)``.

        Args:
            uuid16: The logical ids whose columns are ``binary(16)``.

        Returns:
            The codec for those columns.
        """
        return cls(frozenset(uuid16))

    @classmethod
    def from_id_types(cls, id_types: IdTypes) -> "IdCodec":
        """Build a codec from resolved Arrow id types.

        Args:
            id_types: Mapping from logical id to its Arrow type.

        Returns:
            The codec for those columns.
        """
        return cls(frozenset(name for name, arrow_type in id_types.items() if arrow_type == UUID16))

    def encode(self, logical: str, value: object) -> object:
        """Encode one id to 16 raw bytes for a ``uuid16`` column, else pass it through unchanged.

        Args:
            logical: The logical id the column holds (for example, ``"time_series_id"``).
            value: The id string, or ``None``.

        Returns:
            The 16-byte form for a ``uuid16`` column, else ``value`` unchanged.

        Raises:
            TimeFValidationError: If a ``uuid16`` column holds a non-canonical-UUID id.
        """
        if value is None or logical not in self.uuid16:
            return value
        try:
            return id_to_bytes(cast(str, value))
        except (ValueError, AttributeError, TypeError) as exc:
            raise TimeFValidationError(
                f"id {value!r} in the {logical!r} column is not a canonical UUID, but that column is "
                f"stored as uuid16 (binary(16))"
            ) from exc

    def encode_list(self, logical: str, values: Iterable[object]) -> list:
        """Encode a list of ids element-wise via :meth:`encode`.

        Args:
            logical: The logical id the column holds.
            values: The id strings.

        Returns:
            The encoded list.
        """
        return [self.encode(logical, value) for value in values]
