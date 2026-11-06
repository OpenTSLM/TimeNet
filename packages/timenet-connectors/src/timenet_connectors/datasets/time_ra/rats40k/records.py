"""Build records from checked source descriptors."""

from timenet.dataset import Record, Signal, SignalChannel, Source
from timenet.json import JsonValue
from timenet.types import Annotation
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.datasets.time_ra.rats40k.source import Observation, Row
from timenet_connectors.datasets.time_ra.rats40k.vocabulary import AXIS, SPEC, AnnotationKey, MetadataKey


def build_record(row: Row, loader: SharedChannelLoader[Observation]) -> Record:
    """Build a row's Record: one Source with a lazy signal per channel, and the row's domain annotation.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    signals = Signal.from_shared_loader(
        loader=loader,
        recording_key=row.observation,
        channels=tuple(
            SignalChannel(name=name, spec=SPEC, time_axis=AXIS, n_values=length)
            for name, length in row.observation.channel_lengths
        ),
    )
    source = Source(
        name=row.domain,
        signals=signals,
        # TimeF does not keep signal order on read, so record the source's channel order.
        metadata={MetadataKey.CHANNEL_ORDER: [name for name, _ in row.observation.channel_lengths]},
    )
    # Provenance only: the task reads the domain, so it alone is an annotation.
    metadata: dict[str, JsonValue] = {
        MetadataKey.VARIANT: row.observation.variant.value,
        MetadataKey.SPLIT: row.observation.split.value,
        MetadataKey.ROW_KEY: row.observation.key,
    }
    record = Record(sources=(source,), metadata=metadata)
    record.annotate(Annotation(key=AnnotationKey.SOURCE_DOMAIN, value=row.domain))
    return record
