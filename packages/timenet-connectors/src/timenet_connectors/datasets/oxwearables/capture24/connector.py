"""Convert the pinned CAPTURE-24 archive to records and activity classification tasks.

Each participant has one record with wrist acceleration and activity labels.
Tasks cover 10-second windows in the two benchmark label schemes.
"""

from collections.abc import Callable, Iterator
from enum import StrEnum, unique
from functools import partial
from pathlib import Path

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    ClassificationTask,
    LabelSet,
    TimeInterval,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.datasets.oxwearables.capture24 import source_data
from timenet_connectors.datasets.oxwearables.capture24.source_data import (
    LabelDictionary,
    LabelRuns,
    Recording,
    Scheme,
    Segment,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The URL of the 6.9 GB CAPTURE-24 archive."""
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ARCHIVE_FILENAME = "capture24.zip"
_ARCHIVE_ROOT = "capture24"
"""The top-level directory inside the archive."""

_RECORD_FILE_KEY = "file"
"""The metadata key for a recording's file name."""
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float32"
)


@unique
class _Key(StrEnum):
    """An annotation key for CAPTURE-24 records or sources."""

    SUBJECT_IDS = "subject_ids"
    AGE_BAND = "age_band"
    """The participant's age range in years."""
    SEX = "sex"
    RECORDING_START_LOCAL = "recording_start_local"
    """The first sample's published timestamp with a randomized date and shifted time."""
    ACTIVITY = "activity"
    """An activity annotation for consecutive samples with the same raw label."""


_AGE_BAND_DESCRIPTION = "The participant's age band in years."
_RECORDING_START_DESCRIPTION = (
    "The first sample's timestamp as the authors published it, with no zone. For anonymity they randomized "
    "the date and shifted the time by a small random amount."
)


# The label sets of the two schemes that benchmark.py scores.
_ACTIVITY_TYPES_WILLETTS_SPECIFIC_2018 = LabelSet(
    list(source_data.BENCHMARK_CLASSES[Scheme.WILLETTS_SPECIFIC_2018]),
    annotations=[
        Annotation("name", value="willetts_specific_2018"),
        Annotation(
            "description",
            value="Activity type, in the 10 classes of the WillettsSpecific2018 scheme (Willetts et al., 2018).",
        ),
    ],
)
_MOVEMENT_BEHAVIORS_WALMSLEY_2020 = LabelSet(
    list(source_data.BENCHMARK_CLASSES[Scheme.WALMSLEY_2020]),
    annotations=[
        Annotation("name", value="walmsley_2020"),
        Annotation(
            "description",
            value=(
                "Movement behavior: sleep, sedentary, light or moderate-vigorous activity, as in the "
                "Walmsley2020 scheme (Walmsley et al., 2020)."
            ),
        ),
    ],
)

_BENCHMARK_LABEL_SETS = {
    Scheme.WILLETTS_SPECIFIC_2018: _ACTIVITY_TYPES_WILLETTS_SPECIFIC_2018,
    Scheme.WALMSLEY_2020: _MOVEMENT_BEHAVIORS_WALMSLEY_2020,
}
"""The label sets for the two benchmark schemes."""


_ReadAcceleration = Callable[[Recording], source_data.Acceleration]


class _OneRecording:
    """Cache one recording's acceleration values at a time.

    The cache releases the previous recording before reading a different one.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._values: source_data.Acceleration | None = None

    def __call__(self, recording: Recording) -> source_data.Acceleration:
        """Return cached acceleration values or read the requested recording.

        Returns:
            One float32 array per acceleration axis.
        """
        if self._values is None or self._path != recording.participant.path:
            # Release the previous recording before reading the next.
            self._path, self._values = None, None
            self._values = source_data.read_acceleration(recording)
            self._path = recording.participant.path
        return self._values


def _load_segment(acceleration: _ReadAcceleration, recording: Recording, segment: Segment, axis: int) -> pa.Array:
    """Extract one segment's acceleration values for the given axis index.

    Returns:
        The segment's float32 values as an Arrow array.
    """
    values = acceleration(recording)[axis]
    return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=source_data.AXIS_TYPE)


def _activity_annotations(runs: LabelRuns, labels: LabelDictionary) -> Iterator[Annotation]:
    """Convert raw label runs to activity annotations.

    Yields:
        An annotation with a raw label, its time span, and its class in each of the six schemes as metadata.
    """
    for start_us, end_us, code in zip(runs.starts_us.tolist(), runs.ends_us.tolist(), runs.codes.tolist(), strict=True):
        yield Annotation(
            _Key.ACTIVITY,
            value=labels.raw_labels[code],
            span=TimeInterval.micros(start_us, end_us),
            metadata={scheme.value: name for scheme, name in labels.scheme_classes_of_raw[code].items()},
        )


def _record(recording: Recording, labels: LabelDictionary, acceleration: _ReadAcceleration) -> Record:
    """Build a participant record with one source per segment.

    Each source has x, y, and z signals and raw activity annotations.
    Acceleration values load from the recording file on demand.

    Returns:
        The participant record with signals and annotations.
    """
    sources = []
    for index, (segment, runs) in enumerate(zip(recording.segments, recording.label_runs, strict=True)):
        time_axis = RegularAxis.from_rate_hz(source_data.SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = [
            Signal.from_loader(
                name=column,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(_load_segment, acceleration, recording, segment, axis),
            )
            for axis, column in enumerate(source_data.AXES)
        ]
        # Pass all runs to the Source constructor at once. Adding one run at a time copies the
        # annotation tuple for every run.
        sources.append(
            Source(
                f"{source_data.DEVICE} segment {index}",
                signals=signals,
                annotations=list(_activity_annotations(runs, labels)),
            )
        )
    participant = recording.participant
    return Record(
        sources=sources,
        metadata={_RECORD_FILE_KEY: participant.path.name},
        annotations=[
            Annotation(_Key.SUBJECT_IDS, value=[participant.pid]),
            Annotation(_Key.AGE_BAND, value=participant.age_band.value, description=_AGE_BAND_DESCRIPTION),
            Annotation(_Key.SEX, value=participant.sex.value),
            Annotation(
                _Key.RECORDING_START_LOCAL,
                value=recording.start_local,
                description=_RECORDING_START_DESCRIPTION,
            ),
        ],
    )


def _scheme_tasks(scheme: Scheme, recording: Recording, record: Record) -> Iterator[ClassificationTask]:
    """Yield classification tasks for a participant's selected windows in one scheme.

    Yields:
        A task with the window's time span and class in time order.
    """
    label_set = _BENCHMARK_LABEL_SETS[scheme]
    starts_us = recording.window_starts_us.tolist()
    classes = recording.window_classes[scheme].tolist()
    for start_us, class_index in zip(starts_us, classes, strict=True):
        yield ClassificationTask(
            inputs=[record],
            scope=TimeInterval.micros(start_us, start_us + source_data.WINDOW_US),
            targets=source_data.BENCHMARK_CLASSES[scheme][class_index],
            label_set=label_set,
            split=recording.participant.split,
        )


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from its pinned Oxford Research Archive zip file."""

    async def download_async(self, cache_dir: Path) -> Path:  # ruff: ignore[no-self-use] - the connector interface
        """Download and extract the pinned CAPTURE-24 archive.

        The archive must match ``ARCHIVE_SHA256`` before extraction.
        Later calls reuse the extracted directory.

        Returns:
            The extracted ``capture24`` directory.
        """
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=ARCHIVE_SHA256)
        return find_dir_containing(root, f"{_ARCHIVE_ROOT}/{source_data.METADATA_FILE}")

    def convert(self, root: Path, /) -> TimeFDataset:
        """Convert CAPTURE-24 recordings to records and window classification tasks.

        Acceleration values load from the files on demand.
        Keep the files available and unchanged until all signals have loaded.

        Args:
            root: The extracted ``capture24`` directory.

        Returns:
            A dataset with participant records and streamed tasks for both benchmark schemes.

        Raises:
            TimeFValidationError: If ``root`` is not a directory.
            TimeNetConnectorError: If a metadata table or recording is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
        if not root.is_dir():
            raise TimeFValidationError(f"CAPTURE-24 loads from one directory, got {root}")
        labels = source_data.read_labels(root / source_data.DICTIONARY_FILE)
        # One participant's samples are held at a time: the first pass keeps only small results.
        recordings = [
            source_data.read_recording(participant, labels) for participant in source_data.read_participants(root)
        ]
        acceleration = _OneRecording()
        records = [_record(recording, labels, acceleration) for recording in recordings]

        dataset = TimeFDataset(metadata=self.metadata())
        for record in records:
            dataset.add_record(record)

        def tasks() -> Iterator[ClassificationTask]:
            # As the authors' benchmark.py: one benchmark per scheme, over every participant's windows.
            for scheme in _BENCHMARK_LABEL_SETS:
                for recording, record in zip(recordings, records, strict=True):
                    yield from _scheme_tasks(scheme, recording, record)

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = Capture24Connector
