"""Data-dependent choice of the Parquet encoding for the shard values column.

No single encoding is right for every waveform, so the writer measures the data instead of pinning
one:

- **dictionary** stores each distinct value once and spends a bit-packed index per sample. It wins
  whenever a plane has few distinct values, which is what a physical conversion onto a fixed grid
  (wfdb's 0.001 mV step, an integer ADC scale) produces.
- **BYTE_STREAM_SPLIT** transposes each float into four byte planes and compresses each apart. It
  wins on smooth high-cardinality signals, where the sign and high-mantissa planes are nearly
  constant. It loses on quantized data, where the low mantissa byte is effectively noise that the
  split isolates into an incompressible plane and zstd's LZ77 stage can no longer match whole
  repeated 4-byte values across.
- **plain** wins on nothing measured so far. It stays available as a manual override.

The rule is cardinality on a sample: at most :data:`DICT_MAX_CARDINALITY` distinct values selects
dictionary, more selects BYTE_STREAM_SPLIT. ``benchmarks/value_encoding/`` holds the sweep the
constant comes from.

Two known gaps, both covered by the ``value_encoding`` override rather than by more machinery:

- High-cardinality *quantized* data (say a 24-bit integer-scaled signal) suits neither branch: too
  many distinct values for a dictionary, too much low-bit noise for a byte split. The rule sends it
  to BYTE_STREAM_SPLIT.
- The sample is one modality's first row group, and the writer sorts series by ``(spec_type,
  channel, time_series_id)``, so it comes from that modality's first channel or two. A modality whose
  channels differ sharply in cardinality is judged by its first ones. Paying a second pass over the
  values to avoid this would cost more than the cases it fixes.

The choice needs no reader support. Parquet records the applied encoding per column chunk in its
footer, so a decoder resolves it without consulting the manifest. The manifest records it anyway,
under ``value_encoding``, so a curator can see what happened without opening a shard footer.
"""

from collections.abc import Sequence
from enum import StrEnum

import numpy as np


AUTO = "auto"
"""``value_encoding`` argument selecting the encoding from the data; the writer's default."""


class ValueEncoding(StrEnum):
    """Encoding applied to a shard's ``values.list.element`` column, per ``spec_type``.

    The member *values* are the manifest tags, not Parquet's encoding names: dictionary encoding is
    requested through pyarrow's ``use_dictionary`` list rather than ``column_encoding``, and Parquet
    reports it in the footer under several names, so a one-to-one mapping to a Parquet name does not
    exist. :mod:`timenet.writer.encodings` translates in both directions.
    """

    DICTIONARY = "dictionary"
    BYTE_STREAM_SPLIT = "byte_stream_split"
    PLAIN = "plain"


SUPPORTED_VALUE_ENCODINGS: frozenset[ValueEncoding] = frozenset(ValueEncoding)
"""Every encoding a writer may apply and the self-check will accept."""

DICT_MAX_CARDINALITY = 1 << 16
"""Distinct sampled values up to which dictionary encoding is selected.

``benchmarks/value_encoding/`` measures the crossover at **72,600** sampled distinct values (bracket
66,174 to 79,620), so this constant is about 10% low. It stays where it is for two reasons. It is the
point where a dictionary index stops fitting in 16 bits, which is a mechanism boundary rather than a
round number. And the error is cheap in the direction it errs: over the band between the constant and
the measured crossover, the encoding it picks costs at most 2.1% versus the better one, while
choosing a dictionary well past the crossover costs 9% at 80k distinct and 27% at 107k. Undershooting
is nearly free, overshooting is not.
"""

SAMPLE_MAX_VALUES = 200_000
"""Values the cardinality rule looks at.

The sample comes from values the writer has already buffered, so the only cost is counting. Capped
because the count saturates: a plane with more than :data:`DICT_MAX_CARDINALITY` distinct values
reveals that within this many samples, and looking at more cannot change the branch.

:data:`DICT_MAX_CARDINALITY` is calibrated against this number and against the row-group target,
whichever is smaller. Drawing ``n`` values from an alphabet of ``K`` reveals about
``K(1 - e**(-n/K))`` of it, so a smaller sample reports a smaller count for the same data and would
need a lower threshold. Changing either constant means re-running the sweep.
"""


def sample_values(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Reduce buffered per-chunk values to one bounded sample, striding within each chunk.

    Each chunk contributes an equal share of :data:`SAMPLE_MAX_VALUES`, taken at a stride that spans
    the whole chunk. A contiguous prefix would under-count the alphabet of a chunk whose amplitude
    drifts; a stride sees all of it. Chunks shorter than their share contribute everything.

    Args:
        arrays: One 1-D float32 array per buffered chunk.

    Returns:
        The concatenated sample, at most :data:`SAMPLE_MAX_VALUES` long.
    """
    if not arrays:
        return np.empty(0, dtype=np.float32)
    share = max(1, SAMPLE_MAX_VALUES // len(arrays))
    parts = [array[:: max(1, array.size // share)][:share] for array in arrays]
    return np.concatenate(parts)[:SAMPLE_MAX_VALUES]


def distinct_bit_patterns(values: np.ndarray) -> int:
    """Return how many distinct float32 bit patterns ``values`` holds.

    Bit patterns rather than numeric values: a Parquet dictionary keys on the stored bytes, so
    ``-0.0`` and ``0.0`` occupy two slots even though they compare equal.

    Args:
        values: A 1-D float32 array.

    Returns:
        The number of distinct patterns.
    """
    if values.size == 0:
        return 0
    return int(np.unique(np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)).size)


def encoding_for_cardinality(distinct: int) -> ValueEncoding:
    """Map a sampled distinct-value count to the encoding that stores it smaller.

    Args:
        distinct: Distinct bit patterns counted in the sample.

    Returns:
        Dictionary up to :data:`DICT_MAX_CARDINALITY`, BYTE_STREAM_SPLIT above it. Never PLAIN: it
        won on nothing measured, and stays reachable only as an explicit override.
    """
    return ValueEncoding.DICTIONARY if distinct <= DICT_MAX_CARDINALITY else ValueEncoding.BYTE_STREAM_SPLIT


def select_value_encoding(arrays: Sequence[np.ndarray]) -> ValueEncoding:
    """Pick the encoding for one modality from the values the writer has already buffered.

    Deterministic in the buffered values alone, so re-curating an unchanged source reaches the same
    encoding and its shards stay byte-identical. A copy-on-write edit re-measures whatever survives
    the edit, which can only reach a different answer for a modality already sitting within a few
    percent of the threshold, where the two encodings are near enough in size not to matter.

    Args:
        arrays: The buffered chunks' float32 values, one array per chunk.

    Returns:
        The selected encoding, defaulting to dictionary when there is nothing to measure.
    """
    return encoding_for_cardinality(distinct_bit_patterns(sample_values(arrays)))
