"""One Record per published row: its series as lazy signals, its source row in typed metadata, and its domain."""

from functools import lru_cache, partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, Split, TimeSeriesSpec
from timenet_connectors.datasets.time_ra.rats40k.keys import CHANNEL_ORDER, AnnotationKey
from timenet_connectors.datasets.time_ra.rats40k.release import Variant
from timenet_connectors.datasets.time_ra.rats40k.rows import Row, check_channels, reread_channels
from timenet_connectors.metadata import RecordMetadata


# The release gives neither units nor a sampling cadence.
_SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")


class RATS40KRecordMetadata(RecordMetadata):
    """A RATS40K Record's metadata: the published row the Record holds.

    Keys repeat across splits, so variant, split and key together name the row. The release names no
    upstream series or subject, so the shared ``subject_id`` stays unset.
    """

    variant: Variant
    split: Split
    key: str


def record_metadata(record: Record) -> RATS40KRecordMetadata:
    """Read a RATS40K Record's typed metadata back.

    Returns:
        The published row the Record holds.

    Raises:
        TimeFValidationError: If the Record's metadata is not RATS40K metadata.
    """  # noqa: DOC502 - raised by from_json
    return RATS40KRecordMetadata.from_json(record.metadata)


class ChannelLoader:
    """Load one channel of a row lazily, rereading each row once for all its channels.

    A loader serves the Records of one load, like TRACE's ``WindowLoader``.
    """

    def __init__(self) -> None:
        # A Record's signals load one after another, so one cached row suffices. A RowRef is
        # hashable, unlike a Row, so it keys the cache.
        self._reread = lru_cache(maxsize=1)(reread_channels)

    def __call__(self, row: Row, name: str) -> pa.Array:
        """Return one channel of a row as float64 values.

        Returns:
            The channel's values, checked against the converted row.
        """
        channels = self._reread(row.ref)
        check_channels(channels, row)
        return pa.array(channels[name], type=pa.float64())


def build_record(row: Row, channels: ChannelLoader) -> Record:
    """Build a Record with one lazy signal per channel, its source row, and its domain annotation.

    Args:
        row: The validated row.
        channels: The loader the Record's signals read their values through.

    Returns:
        The row's Record, whose values stay in the source file until read.
    """
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=_SPEC,
            time_axis=OrdinalAxis(),
            n_values=length,
            loader=partial(channels, row, name),
        )
        for name, length in row.channel_lengths.items()
    )
    ref = row.ref
    metadata = RATS40KRecordMetadata(variant=ref.variant, split=ref.split, key=ref.key)
    record = Record(
        sources=(
            Source(
                name=row.domain,
                signals=signals,
                # TimeF does not keep signal order on read, so record the source's channel order.
                metadata={CHANNEL_ORDER: list(row.channel_lengths)},
            ),
        ),
        # Provenance only. The task reads the domain, so it alone is an annotation.
        metadata=metadata.to_json(),
    )
    record.annotate(Annotation(key=AnnotationKey.SOURCE_DOMAIN, value=row.domain))
    return record
