"""Build EngineMT-QA records and dataset provenance from checked source facts."""

from collections.abc import Sequence

from timenet.dataset import Record, Signal, SignalChannel, Source
from timenet.json import JsonValue
from timenet.types import (
    Annotation,
)
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.datasets.pandalin98.enginemt_qa.source import PairLocation
from timenet_connectors.datasets.pandalin98.enginemt_qa.vocabulary import (
    AXIS,
    CHANNELS,
    N_STEPS,
    SOURCE_NAME,
    SPEC,
    AnnotationKey,
    MetadataKey,
)


def build_record(row_id: int, reader: SharedChannelLoader[int]) -> Record:
    """Return a row's Record with 33 lazy signals and its published row ID."""
    signals = Signal.from_shared_loader(
        loader=reader,
        recording_key=row_id,
        channels=tuple(SignalChannel(name=name, spec=SPEC, time_axis=AXIS, n_values=N_STEPS) for name in CHANNELS),
    )
    return Record(sources=(Source(name=SOURCE_NAME, signals=signals),), metadata={MetadataKey.ROW_ID: row_id})


def unlabelled_annotation(places: Sequence[PairLocation]) -> Annotation:
    """Return the source locations of published pairs that have no answer label."""
    pairs: list[JsonValue] = [place.metadata for place in places]
    return Annotation(
        key=AnnotationKey.UNLABELLED_PAIRS,
        value=len(places),
        description="How many closed pairs have an empty published answer. They are missing labels and have no task.",
        metadata={"pairs": pairs},
    )
