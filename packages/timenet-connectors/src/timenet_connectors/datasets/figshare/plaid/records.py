"""One Record per CSV file: lazy current and voltage signals, the file's entry as metadata, and its events."""

from functools import lru_cache, partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimePoint, TimeSeriesSpec, ureg
from timenet_connectors.datasets.figshare.plaid.entries import AggregatedEntry, Appliance, SubmeteredEntry
from timenet_connectors.datasets.figshare.plaid.keys import EVENT_KEYS
from timenet_connectors.datasets.figshare.plaid.measurements import Measurement, read_values
from timenet_connectors.datasets.figshare.plaid.release import SAMPLE_RATE_HZ, Subset
from timenet_connectors.metadata import RecordMetadata


DEVICE = "NI-9215"
"""The acquisition card that digitized both channels: current from a Fluke i200 clamp, voltage from a Pico
TA041 probe."""

# The source writes three decimals; float64 keeps them as written.
_SPECS = {
    "current": TimeSeriesSpec(spec_type="electric_current", name="Current", unit_value=ureg.ampere, dtype="float64"),
    "voltage": TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.volt, dtype="float64"),
}

_EVENT_DESCRIPTION = (
    "A labelled event of an aggregated file: the appliance, by type as the metadata writes it, switched on or "
    "off at the sample its on or off index names."
)


class PlaidRecordMetadata(RecordMetadata):
    """A PLAID Record's metadata: the CSV file the Record holds, and its metadata entry as written.

    The shared ``subject_id`` is the household the authors' GlobalSIP notebook groups by, the campaign then
    the location, such as ``July, 2013_house1``.
    """

    subset: Subset
    file: str
    """The file's path in the release, such as ``submetered/1.csv``."""
    collection_time: str
    location: str
    header_notes: str | None = None
    """The campaign's note. The 2013 files say they are not all well calibrated."""
    length: str
    """The duration the entry states, such as ``2.00s``."""
    status: str
    """A submetered appliance's operating mode, or an aggregated file's sequence of events, as written."""
    appliances: tuple[Appliance, ...]
    """The submetered file's one appliance, or the aggregated file's appliances in entry order."""


def record_metadata(record: Record) -> PlaidRecordMetadata:
    """Read a PLAID Record's typed metadata back.

    Returns:
        The metadata: the household as ``subject_id``, the file, and its entry.

    Raises:
        TimeFValidationError: If the Record's metadata is not PLAID metadata.
    """  # noqa: DOC502 - raised by from_json
    return PlaidRecordMetadata.from_json(record.metadata)


class ValueLoader:
    """Load one column of one CSV file lazily, reading each file once for both its signals.

    A loader serves the Records of one load. A Record's signals load one after another, so one cached file
    suffices.
    """

    def __init__(self) -> None:
        self._read = lru_cache(maxsize=1)(read_values)

    def __call__(self, measurement: Measurement, column: str) -> pa.Array:
        """Return one column of a file.

        Returns:
            The column's float64 values.
        """
        return pa.array(self._read(measurement.path, measurement.n_values)[column], type=pa.float64())


def _metadata(measurement: Measurement) -> PlaidRecordMetadata:
    entry = measurement.entry
    if isinstance(entry, SubmeteredEntry):
        appliances = (entry.appliance,)
    else:
        appliances = tuple(switched.appliance for switched in entry.appliances)
    return PlaidRecordMetadata(
        subject_id=entry.household,
        subset=measurement.subset,
        file=measurement.file,
        collection_time=entry.header.collection_time,
        location=entry.location,
        header_notes=entry.header.notes,
        length=entry.length,
        status=entry.status,
        appliances=appliances,
    )


def build_record(measurement: Measurement, values: ValueLoader) -> Record:
    """Build a file's Record: one Source of lazy current and voltage signals, and its entry as metadata.

    The appliance type stays in the metadata, which is never model input: it is the appliance task's target.

    Args:
        measurement: The file and its entry.
        values: The loader the signals read their values through.

    Returns:
        The Record, whose values stay in the file until read.
    """
    time_axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
    signals = tuple(
        Signal.from_loader(
            name=column,
            spec=spec,
            time_axis=time_axis,
            n_values=measurement.n_values,
            loader=partial(values, measurement, column),
        )
        for column, spec in _SPECS.items()
    )
    metadata = _metadata(measurement)
    return Record(sources=(Source(name=DEVICE, signals=signals),), metadata=metadata.to_json())


def annotate_events(record: Record, entry: AggregatedEntry) -> tuple[Annotation, ...]:
    """Attach an aggregated file's labelled events to its Record, one point annotation each.

    An event at sample ``i`` sits at the time the 30 kHz axis gives sample ``i``, floored to whole
    microseconds. ``RegularAxis.index_at_or_after`` gives the sample back exactly.

    Args:
        record: The file's Record.
        entry: The file's entry.

    Returns:
        The attached annotations, in event order, which the event task names as its targets.
    """
    time_axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
    return tuple(
        record.annotate(
            Annotation(
                key=EVENT_KEYS[event.kind],
                value=event.appliance_type,
                span=TimePoint.micros(time_axis.time_offset_us(event.sample_index)),
                description=_EVENT_DESCRIPTION,
            )
        )
        for event in entry.events()
    )
