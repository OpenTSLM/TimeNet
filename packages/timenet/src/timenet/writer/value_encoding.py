"""Data-dependent choice of the Parquet encoding for the shard values column.

No single encoding is right for every waveform, so the writer reads the data instead of pinning one:

- **dictionary** stores each distinct value once and spends a bit-packed index per sample. It wins
  whenever a plane has few distinct values. A physical conversion onto a fixed grid, such as
  wfdb's 0.001 mV step or an integer ADC scale, produces that kind of plane.
- **BYTE_STREAM_SPLIT** transposes each float into four byte planes and compresses each apart. It
  wins on smooth high-cardinality signals, where the sign and high-mantissa planes are nearly
  constant. It loses on quantized data, where the low mantissa byte is noise.
- **plain** stores each value as it is. The rule never selects it, and it stays available as a
  manual override.

The rule is cardinality on a sample: at most :data:`DICT_MAX_CARDINALITY` distinct values selects
dictionary, more selects BYTE_STREAM_SPLIT. That is the float rule. Bools bypass it and always
encode plain. Strings share the cardinality rule but never select BYTE_STREAM_SPLIT, which is a
float transpose. Past the threshold, a string encodes plain. Integers behave like strings:
dictionary up to the threshold, plain past it.

The ``value_encoding`` override covers two known gaps instead of more machinery:

- High-cardinality *quantized* data (say a 24-bit integer-scaled signal) suits neither branch. It
  has too many distinct values for a dictionary and too much low-bit noise for a byte split. The
  rule sends it to BYTE_STREAM_SPLIT.
- The sample is one modality's first row group. The writer sorts series by ``(spec_type, signal,
  time_series_id)``, so the sample comes from that modality's first signal or two. The rule judges
  a modality by its first signals only, even when the modality's signals differ sharply in
  cardinality.

The choice needs no reader support. Parquet records the applied encoding per column chunk in its
footer, so a decoder resolves it without consulting the manifest. The manifest records it anyway,
under ``value_encoding``, so a builder can see what happened without opening a shard footer.
"""

from collections.abc import Sequence
from enum import StrEnum

import numpy as np


AUTO = "auto"
"""``value_encoding`` argument selecting the encoding from the data. The writer's default."""


class ValueEncoding(StrEnum):
    """Encoding applied to a shard's ``values.list.element`` column, per ``spec_type``.

    The member *values* are the manifest tags, not Parquet's encoding names. pyarrow's
    ``use_dictionary`` list requests dictionary encoding rather than ``column_encoding``. Parquet
    reports the encoding in the footer under several names, so a one-to-one mapping to a Parquet
    name does not exist. :mod:`timenet.writer.encodings` translates in both directions.
    """

    DICTIONARY = "dictionary"
    BYTE_STREAM_SPLIT = "byte_stream_split"
    PLAIN = "plain"


SUPPORTED_VALUE_ENCODINGS: frozenset[ValueEncoding] = frozenset(ValueEncoding)
"""Every encoding a writer can apply and the self-check accepts."""

DICT_MAX_CARDINALITY = 1 << 16
"""Distinct sampled values up to which the rule selects dictionary encoding.

This is the point where a dictionary index stops fitting in 16 bits.
"""

_MIB = 1 << 20

ENCODING_SAMPLE_SOURCE_BYTES = 4 * _MIB
"""Leading bytes the encoding rule draws its sample from, independent of the row-group size.

The cap is fixed so that a larger ``row_group_target_bytes`` cannot shift the sampled cardinality.
Such a shift flips a borderline modality's encoding.
"""

SAMPLE_MAX_VALUES = 200_000
"""Values the cardinality rule looks at.

The sample comes from values the writer already buffered. A plane with more than
:data:`DICT_MAX_CARDINALITY` distinct values shows that within this many samples, so a look at more
values cannot change the branch.

The two constants belong together. A smaller sample reports a smaller distinct count for the same
data, so it needs a lower :data:`DICT_MAX_CARDINALITY`. Change one and recalibrate the other.
"""


def sample_values(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Reduce buffered per-chunk values to one bounded sample, with a stride within each chunk.

    Each chunk contributes an equal share of :data:`SAMPLE_MAX_VALUES`, taken at a stride that spans
    the whole chunk. A contiguous prefix under-counts the alphabet of a chunk whose amplitude
    drifts. A stride sees all of it. Chunks shorter than their share contribute everything. All
    chunks share one dtype, which the sample keeps.

    Args:
        arrays: One 1-D numeric array per buffered chunk, all the same dtype.

    Returns:
        The concatenated sample, at most :data:`SAMPLE_MAX_VALUES` long, in the chunks' dtype. An
        empty sample is float32, which no caller reads values out of.
    """
    if not arrays:
        return np.empty(0, dtype=np.float32)
    share = max(1, SAMPLE_MAX_VALUES // len(arrays))
    parts = [array[:: max(1, array.size // share)][:share] for array in arrays]
    sample = np.concatenate(parts)[:SAMPLE_MAX_VALUES]
    return sample if sample.size else np.empty(0, dtype=np.float32)


def distinct_bit_patterns(values: np.ndarray) -> int:
    """Return how many distinct value bit patterns ``values`` holds.

    Bit patterns rather than numeric values. A Parquet dictionary keys on the stored bytes, so
    ``-0.0`` and ``0.0`` occupy two slots even though they compare equal. A fixed-width numeric
    array is read through the same-width unsigned view. That view turns each stored pattern into one
    integer: a float32 array reads as uint32, an int16 array as uint16, and so on. Bools and
    strings have no byte-linearity to exploit and are counted directly.

    Args:
        values: A 1-D array of one fixed-width numeric dtype, bool, or object-of-str.

    Returns:
        The number of distinct patterns.
    """
    if values.size == 0:
        return 0
    contiguous = np.ascontiguousarray(values)
    if np.issubdtype(values.dtype, np.bool_):
        return int(np.unique(contiguous).size)
    if np.issubdtype(values.dtype, np.number):
        unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(values.dtype.itemsize)
        if unsigned is not None:
            return int(np.unique(contiguous.view(unsigned)).size)
    return int(np.unique(contiguous).size)


def encoding_for_cardinality(distinct: int, *, dtype: str = "float32") -> ValueEncoding:
    """Map a sampled distinct-value count and a values dtype to the encoding that stores them smaller.

    The cardinality rule selects dictionary up to :data:`DICT_MAX_CARDINALITY`. Past it, the choice
    is a dtype matter rather than a count matter. Floats transpose into byte planes
    (BYTE_STREAM_SPLIT). Strings and integers have no byte-plane meaning and fall back to plain.
    Bools never reach this function: they bypass the rule.

    Args:
        distinct: Distinct bit patterns counted in the sample.
        dtype: The values dtype as its string name (``"float32"``, ``"int16"``, ``"bool"``, ...).
            Defaults to ``"float32"``.

    Returns:
        Dictionary up to :data:`DICT_MAX_CARDINALITY`, BYTE_STREAM_SPLIT past it for floats, plain
        past it for other types. The rule never selects PLAIN for floats. PLAIN stays reachable for
        them only as an explicit override.
    """
    if distinct <= DICT_MAX_CARDINALITY:
        return ValueEncoding.DICTIONARY
    from timenet.writer.encodings import byte_stream_split_supported  # noqa: PLC0415

    return ValueEncoding.BYTE_STREAM_SPLIT if byte_stream_split_supported(dtype) else ValueEncoding.PLAIN


def select_value_encoding(arrays: Sequence[np.ndarray], *, dtype: str = "float32") -> ValueEncoding:
    """Pick the encoding for one modality from the values the writer has already buffered.

    This choice depends on the buffered values alone. A re-build of an unchanged source reaches the
    same encoding, and its shards stay byte-identical. A copy-on-write edit runs the rule again over
    whatever survives the edit, so a modality near the threshold can come out the other way.

    The rule is dtype-aware. Bools have no redundancy to spend a dictionary on, so they encode
    plain. Everything else counts the sample and defers the count-to-encoding mapping to
    :func:`encoding_for_cardinality`.

    Args:
        arrays: The buffered chunks' values, one array per chunk.
        dtype: The values dtype as its string name (``"float32"``, ``"int16"``, ``"bool"``,
            ``"str"`` ...). Defaults to ``"float32"``.

    Returns:
        The selected encoding. It is dictionary when there is nothing to count.
    """
    if dtype == "bool":
        return ValueEncoding.PLAIN
    return encoding_for_cardinality(distinct_bit_patterns(sample_values(arrays)), dtype=dtype)
