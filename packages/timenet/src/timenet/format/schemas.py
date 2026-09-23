"""Pinned Arrow schema of the Parquet value shards and the id codec that writes them.

Every id column holds one of the logical ids in :data:`LOGICAL_IDS`, stored as ``pa.string()`` or as
``pa.binary(16)`` when every value of that id is a canonical UUID. The writer picks the type per
logical id; the DuckDB control plane stores public ids as strings and needs no codec.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from timenet.errors import TimeFValidationError
from timenet.types.ids import id_to_bytes


#: The logical ids that cross-reference TimeF entities. Every id column holds exactly one of these.
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


#: Where each logical id lives in the records Parquet schema.
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

    This is the one place that holds the logical-id-per-column mapping and the ``bytes <-> str``
    conversion. The writer and the reader each build a codec (:meth:`from_uuid16` or
    :meth:`from_encoding`). Both call it, so the two halves of the format contract cannot drift apart.
    """

    uuid16: frozenset[str]
    """The logical ids whose columns are stored as ``binary(16)``. The rest are ``pa.string()``."""

    @classmethod
    def from_uuid16(cls, uuid16: Iterable[str]) -> "IdCodec":
        """Build a codec from the set of logical ids stored as ``binary(16)`` (the writer's entry point).

        Args:
            uuid16: The logical ids whose columns are ``binary(16)``.

        Returns:
            The codec for those columns.
        """
        return cls(frozenset(uuid16))

    def encode(self, logical: str, value: object) -> object:
        """Encode one id to 16 raw bytes for a ``uuid16`` column, else pass it through unchanged.

        A reference ID outside the entity ID space raises a contextual
        :class:`TimeFValidationError` naming the column and value. This makes the writer fail legibly
        even if referential validation is bypassed.

        Args:
            logical: The logical id the column holds (for example, ``"record_id"``).
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
                f"stored as uuid16 (binary(16)); a reference likely points outside the entity id space"
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
