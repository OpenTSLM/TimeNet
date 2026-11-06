"""Build TRACE records from checked windows and their shared measurement reader."""

from functools import partial

from timenet.dataset import Record, Signal, SignalChannel, Source
from timenet.dataset.axis import IrregularAxis
from timenet_connectors.datasets.trace.weather_events.source import Measurement, Window, WindowReader
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel, MetadataKey


_STATION_SOURCE = "station"
"""The name of each Record's one Source. It is generic: the station id is a strong prior on the event type."""


def _source(measurement: Measurement, reader: WindowReader) -> Source:
    """Build a new seven-channel station Source, whose values stay in the source file.

    Each call returns new Signals, because a Signal belongs to one Record.

    Returns:
        A Source with lazy signals on one shared time axis.
    """
    location, axis = measurement.location, measurement.axis
    offsets_loader = partial(reader.time_offsets, location) if isinstance(axis, IrregularAxis) else None
    signals = Signal.from_shared_loader(
        loader=reader.shared,
        recording_key=location,
        channels=tuple(
            SignalChannel(
                name=channel.value,
                spec=channel.spec,
                time_axis=axis,
                time_offsets_loader=offsets_loader,
                n_values=measurement.n_values,
            )
            for channel in Channel
        ),
    )
    return Source(
        name=_STATION_SOURCE,
        signals=signals,
        # TimeF does not keep signal order on read, so record the source's channel order.
        metadata={MetadataKey.CHANNEL_ORDER: [channel.value for channel in Channel]},
    )


def build_record(window: Window, reader: WindowReader) -> Record:
    """Build the window's seven-channel station Record, whose values stay in the source file.

    The Record has no start time: the timestamps are local clock time in an unstated zone, so it has no
    absolute anchor. The published first and last timestamps stay in its metadata.

    Returns:
        A Record with lazy signals on one shared time axis.
    """
    return Record(sources=(_source(window.measurement, reader),), metadata=window.metadata())
