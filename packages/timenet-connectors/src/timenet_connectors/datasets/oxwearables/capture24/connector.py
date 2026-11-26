"""Convert the pinned CAPTURE-24 archive to records and activity classification tasks.

Each gap-free segment of a participant's recording becomes one record with wrist acceleration and
activity labels.
Tasks cover 10-second windows in the two benchmark label schemes.
"""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    Split,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.datasets.oxwearables.capture24 import source_data
from timenet_connectors.datasets.oxwearables.capture24.source_data import (
    LabelDictionary,
    LabelRuns,
    LabelScheme,
    Participant,
    Recording,
    Segment,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


ARCHIVE_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
"""The URL of the 6.9 GB CAPTURE-24 archive."""
_ARCHIVE_FILENAME = "capture24.zip"
_ARCHIVE_ROOT = "capture24"
"""The top-level directory inside the archive."""

_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float32"
)


_AGE_BAND_DESCRIPTION = "The participant's age band in years."
_RECORDING_START_DESCRIPTION = (
    "The first sample's timestamp as the authors published it, with no zone. For anonymity they randomized "
    "the date and shifted the time by a small random amount."
)


# The label sets of the two schemes that benchmark.py scores.
_ACTIVITY_TYPES_WILLETTS_SPECIFIC_2018 = LabelSet(
    list(source_data.SCHEME_CLASSES[LabelScheme.WILLETTS_SPECIFIC_2018]),
    annotations=[
        Annotation("name", value="willetts_specific_2018"),
        Annotation(
            "description",
            value="Activity type, in the 10 classes of the WillettsSpecific2018 scheme (Willetts et al., 2018).",
        ),
    ],
)
# The class definitions of Walmsley et al., Br J Sports Med 2021.
# MET is the energy use of an activity relative to quiet sitting.
_WALMSLEY_2020_DEFINITIONS = {
    "light": "Waking behavior below 3 METs that is not sedentary behavior.",
    "moderate-vigorous": "All behavior at 3 METs or more.",
    "sedentary": "Waking behavior at 1.5 METs or less in a sitting, lying or reclining posture.",
    "sleep": "Non-waking behavior.",
}
_MOVEMENT_BEHAVIORS_WALMSLEY_2020 = LabelSet(
    [
        Label(name=name, description=_WALMSLEY_2020_DEFINITIONS[name])
        for name in source_data.SCHEME_CLASSES[LabelScheme.WALMSLEY_2020]
    ],
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
    LabelScheme.WILLETTS_SPECIFIC_2018: _ACTIVITY_TYPES_WILLETTS_SPECIFIC_2018,
    LabelScheme.WALMSLEY_2020: _MOVEMENT_BEHAVIORS_WALMSLEY_2020,
}

# The splits of the paper and of benchmark.py.
_TRAIN_PARTICIPANTS = range(1, 101)
_TEST_PARTICIPANTS = range(101, 152)


class _SignalKey(NamedTuple):
    """One signal of a recording: the first row of its segment, and its axis index."""

    segment_first_row: int
    axis: int


class _AccelerationReader:
    """Read the acceleration values of a segment for the signals that load lazily.

    It keeps the last recording it read, to optimize sequential reads.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._values: source_data.AccelerationValues | None = None
        self._loaded: set[_SignalKey] = set()
        """The signals that loaded since the recording was read."""

    def read_segment(self, recording: Recording, segment: Segment, axis: int) -> pa.Array:
        """Return one segment's values on one axis.

        Returns:
            The segment's float32 values as an Arrow array.
        """
        if self._values is None or self._path != recording.participant.path:
            # Release the previous recording before reading the next.
            self._path, self._values = None, None
            self._values = source_data.read_acceleration(recording)
            self._path = recording.participant.path
            self._loaded = set()
        values = self._values[axis]
        self._loaded.add(_SignalKey(segment.first_row, axis))
        if len(self._loaded) == len(source_data.AXES) * len(recording.segments):
            self._path, self._values = None, None
        return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=source_data.AXIS_TYPE)


def _activity_key(scheme: LabelScheme) -> str:
    """Return the annotation key of a run's class in one scheme, such as ``activity_walmsley_2020``."""
    return f"activity_{scheme.name.lower()}"


def _activity_annotations(runs: LabelRuns, labels: LabelDictionary) -> Iterator[Annotation]:
    """Convert raw label runs to activity annotations.

    Yields:
        For each run, an annotation with its raw label, then one with its class in each of the six schemes.
    """
    for start_us, end_us, code in zip(runs.starts_us.tolist(), runs.ends_us.tolist(), runs.codes.tolist(), strict=True):
        span = TimeInterval.micros(start_us, end_us)
        yield Annotation("activity", value=labels.raw_labels[code], span=span)
        for scheme, name in labels.scheme_classes_of_raw[code].items():
            description = f"The run's activity class in the {scheme.value} label scheme."
            yield Annotation(_activity_key(scheme), value=name, span=span, description=description)


def _build_records(recording: Recording, labels: LabelDictionary, acceleration: _AccelerationReader) -> list[Record]:
    """Build one Record per continuous segment of a recording, all on one shared time origin.

    Each Record has one Source with x, y, and z signals and the segment's activity annotations.
    Acceleration values load from the recording file on demand.

    Returns:
        The Records in segment order.
    """
    # The parts of one recording share one origin, so the time between them stays exact.
    origin = TimeOrigin()
    participant = recording.participant
    records = []
    for index, segment in enumerate(recording.segments):
        time_axis = RegularAxis.from_rate_hz(source_data.SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = [
            Signal.from_loader(
                name=column,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(acceleration.read_segment, recording, segment, axis),
            )
            for axis, column in enumerate(source_data.AXES)
        ]
        source = Source(
            source_data.DEVICE,
            signals=signals,
            annotations=list(_activity_annotations(segment.runs, labels)),
            metadata={"device": source_data.DEVICE, "placement": source_data.PLACEMENT},
        )
        records.append(
            Record(
                start_time=origin,
                sources=[source],
                metadata={"file": participant.path.name, "segment": index},
                annotations=[
                    Annotation("subject_ids", value=[participant.pid]),
                    Annotation("age_band", value=participant.age_band.value, description=_AGE_BAND_DESCRIPTION),
                    Annotation("sex", value=participant.sex.value),
                    Annotation(
                        "recording_start_local",
                        value=recording.start_local,
                        description=_RECORDING_START_DESCRIPTION,
                    ),
                ],
            )
        )
    return records


def _split_of(participant: Participant) -> Split:
    """Return the benchmark split of a participant.

    Raises:
        TimeNetConnectorError: If the participant is in neither split.
    """
    number = int(participant.pid.removeprefix("P"))
    if number in _TRAIN_PARTICIPANTS:
        return Split.TRAIN
    if number in _TEST_PARTICIPANTS:
        return Split.TEST
    metadata_file = participant.path.parent / source_data.METADATA_FILE
    raise TimeNetConnectorError(f"{metadata_file}: {participant.pid} is in neither the train split nor the test split")


def _make_classification_tasks(
    label_scheme: LabelScheme, recording: Recording, records: list[Record]
) -> Iterator[ClassificationTask]:
    """Yield classification tasks for a participant's selected windows in one scheme.

    Yields:
        A task with the window's time span and class in time order, on the Record of its segment.
    """
    segment_starts_us = [segment.offset_us for segment in recording.segments]
    segment_indices = np.searchsorted(segment_starts_us, recording.window_starts_us, side="right") - 1
    starts_us = recording.window_starts_us.tolist()
    classes = recording.window_classes[label_scheme].tolist()
    for start_us, class_index, segment_index in zip(starts_us, classes, segment_indices.tolist(), strict=True):
        yield ClassificationTask(
            inputs=[records[segment_index]],
            scope=TimeInterval.micros(start_us, start_us + source_data.WINDOW_US),
            targets=source_data.SCHEME_CLASSES[label_scheme][class_index],
            label_set=_BENCHMARK_LABEL_SETS[label_scheme],
            split=_split_of(recording.participant),
        )


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from its pinned Oxford Research Archive zip file."""

    async def download_async(self, cache_dir: Path) -> Path:
        """Download and extract the CAPTURE-24 archive that the dataset card pins by its SHA-256.

        Returns:
            The extracted ``capture24`` directory.

        Raises:
            TimeNetConnectorError: If the card's ``source_revision`` is not ``sha256:`` and a digest.
        """
        revision = self.metadata().source_revision or ""
        sha256 = revision.removeprefix("sha256:")
        if sha256 == revision:
            raise TimeNetConnectorError(f"the dataset card's source_revision must start with sha256:, got {revision!r}")
        root = await ensure_archive(ARCHIVE_URL, cache_dir, filename=_ARCHIVE_FILENAME, sha256=sha256)
        return find_dir_containing(root, f"{_ARCHIVE_ROOT}/{source_data.METADATA_FILE}")

    def convert(self, root: Path, /) -> TimeFDataset:
        """Convert CAPTURE-24 recordings to records and window classification tasks.

        Args:
            root: The extracted ``capture24`` directory.

        Returns:
            A dataset with participant records and streamed tasks for both benchmark schemes.

        Raises:
            TimeNetConnectorError: If a metadata table or recording is invalid.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the read helpers
        labels = source_data.read_labels(root / source_data.DICTIONARY_FILE)
        participants = source_data.read_participants(root)
        recordings = [source_data.read_recording(participant, labels) for participant in participants]
        acceleration = _AccelerationReader()
        records = [_build_records(recording, labels, acceleration) for recording in recordings]
        dataset = TimeFDataset(metadata=self.metadata())
        for parts in records:
            for record in parts:
                dataset.add_record(record)

        def tasks() -> Iterator[ClassificationTask]:
            # As the authors' benchmark.py: one benchmark per scheme, over every recording's windows.
            for label_scheme in _BENCHMARK_LABEL_SETS:
                for recording, parts in zip(recordings, records, strict=True):
                    yield from _make_classification_tasks(label_scheme, recording, parts)

        dataset.set_task_stream((ClassificationTask,), tasks)
        return dataset


CONNECTOR = Capture24Connector
