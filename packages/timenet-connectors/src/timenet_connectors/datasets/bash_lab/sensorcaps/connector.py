"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The connector downloads the original JSONL files at a pinned commit through the HuggingFace base,
indexes rows by byte offset, and keeps just one decoded sensor row cached across the six channel
loaders. The prompt of each task is the row's system message. Its input annotations are the activity
summary and, when present, the sensor location. The 12 derived features are Record annotations only,
so the Records keep their text in memory. The connector reads the files again for the tasks.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import json
from typing import Annotated, ClassVar

import pyarrow as pa
from pydantic import Field, FiniteFloat, JsonValue, Strict, TypeAdapter, ValidationError

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef


DECLARED_SAMPLING_RATE_HZ = 10
_SOURCE_SAMPLING_RATE_HZ = 20
_SOURCE_WINDOW_SAMPLES = 120
_EXPORTED_SAMPLES = 60
# Apply the shared SensorCaps timing interpretation to all five partitions: upstream selects
# round(linspace(0, 119, 60)) from a 20 Hz window. These are
# reconstructed relative offsets, not acquisition timestamps or a regular 10 Hz cadence.
# https://github.com/BASHLab/LLaSA/blob/main/llasa_v2_data_generation.py
_TIME_OFFSETS_US = pa.array(
    [
        round(index * (_SOURCE_WINDOW_SAMPLES - 1) / (_EXPORTED_SAMPLES - 1)) * (1_000_000 // _SOURCE_SAMPLING_RATE_HZ)
        for index in range(_EXPORTED_SAMPLES)
    ],
    type=pa.int64(),
)
_AXIS = IrregularAxis(first_us=0, last_us=(_SOURCE_WINDOW_SAMPLES - 1) * (1_000_000 // _SOURCE_SAMPLING_RATE_HZ))
_SPECS = (
    # Upstream rounds IMU values to six decimal places. Use compact float32 storage, accepting
    # normal binary floating-point rounding rather than implying exact decimal preservation.
    TimeSeriesSpec(spec_type="accelerometer", name="Accelerometer", unit_value=None, dtype="float32"),
    TimeSeriesSpec(spec_type="gyroscope", name="Gyroscope", unit_value=None, dtype="float32"),
)
_AXES = ("x", "y", "z")
_CHANNELS = tuple(
    (spec, f"{spec.spec_type}_{axis}", sensor, dimension)
    for sensor, spec in enumerate(_SPECS)
    for dimension, axis in enumerate(_AXES)
)
# The published PAMAP2 arrays are shifted by one raw column relative to XYZ groups.
# Verified against all 9,672 examples and UCI's per-IMU column layout:
# https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
# Keep the published values/order, but identify their actual physical channels.
_PAMAP2_CHANNELS = (
    (_SPECS[0], "accelerometer_16g_y", 0, 0),
    (_SPECS[0], "accelerometer_16g_z", 0, 1),
    (_SPECS[0], "accelerometer_6g_x", 0, 2),
    (_SPECS[1], "gyroscope_y", 1, 0),
    (_SPECS[1], "gyroscope_z", 1, 1),
    (
        TimeSeriesSpec(spec_type="magnetometer", name="Magnetometer", unit_value=None, dtype="float32"),
        "magnetometer_x",
        1,
        2,
    ),
)
FEATURE_HEADINGS = tuple(
    f"{sensor} {statistic}"
    for statistic in ("moving average", "FFT", "minimum", "maximum", "median", "variance")
    for sensor in ("Gyroscope", "Accelerometer")
)
"""The derived features every v2 user message carries after its sensor arrays, in source order.

The LLaSA generator writes these 12 headings in this order. For PAMAP2, it writes the sensor location
after them. The parser rejects any other heading, so a change of the source format fails the conversion:
https://github.com/BASHLab/LLaSA/blob/37ea514bef969d81edc721846689a120d2abeba4/llasa_v2_data_generation.py#L103-L114
https://github.com/BASHLab/LLaSA/blob/37ea514bef969d81edc721846689a120d2abeba4/llasa_v2_data_generation_pamap2.py#L92-L104
"""
INPUT_ANNOTATION_KEYS = frozenset({"activity_summary", "sensor_location"})
"""The Record annotations that a task gives a model: the activity label and the sensor location.

The derived features stay on the Record only. They are about 36,000 characters of NumPy text per row,
and the generator computed them from the 120-reading source window, not from the 60 stored readings.
"""
_FIELDS = frozenset({"Summary", "Accelerometer", "Gyroscope", "Location"})
_ROLES = ("system", "user", "assistant")
_XYZValues = Annotated[list[Annotated[FiniteFloat, Strict()]], Field(min_length=3, max_length=3)]
_SENSOR_VALUES = TypeAdapter(Annotated[list[_XYZValues], Field(min_length=1)])


@dataclass(frozen=True)
class _RowRef:
    file: HubFileRef
    index: int
    offset: int

    @property
    def key(self) -> tuple[str, int]:
        return self.file.part, self.index

    @property
    def context(self) -> str:
        return f"{self.file.path} ({self.file.part}, row {self.index})"


@dataclass(frozen=True)
class _ParsedRow:
    system: str
    caption: str
    summary: str
    location: str | None
    features: Mapping[str, str]
    """The original text of each derived feature, keyed by its heading in :data:`FEATURE_HEADINGS`."""
    values: tuple[list[list[float]], list[list[float]]]
    """The accelerometer's and then the gyroscope's readings."""


def _sensor_values(text: str, name: str) -> list[list[float]]:
    """Decode and validate one time-major numeric xyz array.

    Returns:
        Validated numeric triples in source order.

    Raises:
        TimeFValidationError: If the decoded values are empty, ragged, or nonnumeric.
    """
    try:
        # Finite published values do not imply complete raw measurements: PAMAP2's
        # missing readings appear to have been zero-filled before upstream resampling.
        return _SENSOR_VALUES.validate_json(text)
    except ValidationError as exc:
        raise TimeFValidationError(f"{name}: {exc}") from exc


def _user_sections(user: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split the user message into its structured fields and its derived features.

    Returns:
        Named source fields, and each derived feature's original text keyed by its heading.

    Raises:
        TimeFValidationError: If a section is repeated, a derived feature is missing, or a line
            belongs to no known section.
    """
    sections: dict[str, str] = {}
    feature_lines: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in user.splitlines(keepends=True):
        heading, separator, value = line.partition(":")
        if separator and heading in _FIELDS:
            if heading in sections:
                raise TimeFValidationError(f"repeated {heading} section")
            sections[heading] = value.strip()
            current = None
        elif separator and heading in FEATURE_HEADINGS:
            if heading in feature_lines:
                raise TimeFValidationError(f"repeated {heading} section")
            current = feature_lines[heading] = [value]
        elif current is not None and (not line.strip() or line[0].isspace()):
            # NumPy prints a long statistic over indented continuation lines.
            current.append(line)
        else:
            raise TimeFValidationError(f"unexpected line in the user message: {line[:40]!r}")
    missing = [heading for heading in FEATURE_HEADINGS if heading not in feature_lines]
    if missing:
        raise TimeFValidationError(f"missing derived features: {missing}")
    # Keep each statistic's original text, including complex numbers and NumPy ellipses.
    features = {heading: "".join(feature_lines[heading]).strip() for heading in FEATURE_HEADINGS}
    return sections, features


def _message_contents(row: JsonValue) -> tuple[str, str, str]:
    """Validate and extract the ordered system, user, and assistant messages.

    Returns:
        The three nonempty message contents in source order.

    Raises:
        TimeFValidationError: If messages are missing, empty, or out of order.
    """
    messages = row.get("messages") if isinstance(row, dict) else None
    if not isinstance(messages, list) or len(messages) != len(_ROLES):
        raise TimeFValidationError("expected system, user, and assistant messages")
    contents: list[str] = []
    for message, role in zip(messages, _ROLES, strict=True):
        content = message.get("content") if isinstance(message, dict) else None
        if (
            not isinstance(message, dict)
            or message.get("role") != role
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise TimeFValidationError(f"expected a nonempty {role} message")
        contents.append(content)
    return contents[0], contents[1], contents[2]


def _section_values(sections: dict[str, str]) -> tuple[list[list[float]], list[list[float]]]:
    """Validate structured fields and decode the paired sensor arrays.

    Returns:
        Accelerometer and gyroscope readings in source order.

    Raises:
        TimeFValidationError: If fields are missing or invalid for the pinned release.
    """
    if not sections.get("Summary"):
        raise TimeFValidationError("missing or empty Summary section")
    for heading in ("Accelerometer", "Gyroscope"):
        if heading not in sections:
            raise TimeFValidationError(f"missing {heading} section")
    accelerometer = _sensor_values(sections["Accelerometer"], "Accelerometer")
    gyroscope = _sensor_values(sections["Gyroscope"], "Gyroscope")
    if len(accelerometer) != len(gyroscope):
        raise TimeFValidationError("Accelerometer and Gyroscope lengths differ")
    if len(accelerometer) != _EXPORTED_SAMPLES:
        raise TimeFValidationError(f"expected {_EXPORTED_SAMPLES} readings per sensor in the pinned v2 release")
    if "Location" in sections and not sections["Location"]:
        raise TimeFValidationError("empty Location section")
    return accelerometer, gyroscope


def _parse_row(raw: bytes, context: str) -> _ParsedRow:
    """Parse a source row, adding file and row context to decoding failures.

    Returns:
        Captioning instructions, targets, descriptive fields, and validated sensor arrays.

    Raises:
        TimeFValidationError: If the JSON, messages, or sensor sections are malformed.
    """
    try:
        row: JsonValue = json.loads(raw)
        system, user, caption = _message_contents(row)
        sections, features = _user_sections(user)
        values = _section_values(sections)
        return _ParsedRow(
            system,
            caption,
            sections["Summary"],
            sections.get("Location"),
            features,
            values,
        )
    except (ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        # Source text is untrusted; retain the narrow decoding failure and identify its exact row.
        raise TimeFValidationError(f"{context}: {exc}") from exc


def _iter_rows(file: HubFileRef) -> Iterator[tuple[_RowRef, _ParsedRow]]:
    """Read a JSONL file one row at a time, retaining seekable binary offsets.

    Yields:
        Each lightweight row reference and its transient decoded contents.
    """
    with file.path.open("rb") as handle:
        index = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                return
            ref = _RowRef(file, index, offset)
            yield ref, _parse_row(raw, ref.context)
            index += 1


class _ValueLoader:
    """Share one bounded sensor-row cache across a dataset's lazy channel loaders."""

    def __init__(self) -> None:
        self.cached: tuple[_RowRef, tuple[list[list[float]], list[list[float]]]] | None = None

    def __call__(self, ref: _RowRef, sensor: int, axis: int) -> pa.Array:
        cached = self.cached
        if cached is None or cached[0] != ref:
            with ref.file.path.open("rb") as handle:
                handle.seek(ref.offset)
                values = _parse_row(handle.readline(), ref.context).values
            self.cached = ref, values
        else:
            values = cached[1]
        # Preserve the HF benchmark values. Do not infer nulls from zeros or substitute
        # original PAMAP2 readings: resampling has already mixed zero-filled gaps into
        # the published arrays. See README.md for the missing-data limitation.
        # Use the local values even if another consumer replaces the shared cache during this read.
        return pa.array((sample[axis] for sample in values[sensor]), type=pa.float32())


def _load_time_offsets() -> pa.Array:
    """Return the shared reconstructed offsets in integer microseconds.

    Returns:
        One relative offset per exported sensor reading.
    """
    return _TIME_OFFSETS_US


def _feature_key(heading: str) -> str:
    """Return the annotation key of a derived feature, such as ``gyroscope_moving_average``.

    Returns:
        The heading in lowercase, with underscores for spaces.
    """
    return heading.lower().replace(" ", "_")


def _iter_tasks(files: tuple[HubFileRef, ...], records: Mapping[tuple[str, int], Record]) -> Iterator[AnswerTask]:
    """Re-read original system prompts and caption targets without retaining the corpus.

    Yields:
        A caption task with the system message as prompt, its registered input Record, and the activity
        summary and sensor location as input annotations.
    """
    for file in files:
        for ref, row in _iter_rows(file):
            record = records[ref.key]
            yield AnswerTask(
                inputs=(record,),
                targets=(row.caption,),
                prompt=row.system,
                # The summary and the location, in Record order. The derived features stay on the Record.
                input_annotations=tuple(a for a in record.annotations if a.key in INPUT_ANNOTATION_KEYS),
                split=Split.TRAIN,
            )


class SensorCapsConnector(BaseHuggingFaceConnector):
    """Build SensorCaps with lazy values and streamed captions."""

    HF_REPO = "BASH-Lab/SensorCaps"
    SOURCE_REVISION = "4242e07feaacb0ee3aff86182a62914fa7abacef"
    # Pinned inventory of the original captioning files.
    SOURCE_ROWS: ClassVar[Mapping[str, int]] = {
        "hhar_v2": 9166,
        "motion_v2": 4534,
        "pamap2_v2": 9672,
        "shoaib_v2": 10500,
        "uci_v2": 2088,
    }
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        partition: f"data/{partition}-00000-of-00001.jsonl" for partition in SOURCE_ROWS
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Index local rows into six-channel Records and a re-iterable caption stream.

        Returns:
            A dataset that keeps its hierarchy, its annotations, and file-backed loaders. The
            annotations hold the derived-feature text of every row, about 1.3 billion characters for
            the full release.

        Raises:
            TimeFValidationError: If references or source rows violate the SensorCaps mapping.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            ("source_repository", self.HF_REPO),
            ("source_revision", self.SOURCE_REVISION),
            ("declared_sampling_rate_hz", DECLARED_SAMPLING_RATE_HZ),
            ("source_sampling_rate_hz", _SOURCE_SAMPLING_RATE_HZ),
            ("source_window_samples", _SOURCE_WINDOW_SAMPLES),
            ("time_axis_reconstruction", "round(linspace(0, 119, 60)) / 20 seconds relative to window start"),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        records: dict[tuple[str, int], Record] = {}
        loader = _ValueLoader()
        for file in raw_refs:
            count = 0
            for ref, row in _iter_rows(file):
                record = self._make_record(ref, row, loader)
                dataset.add_record(record=record)
                records[ref.key] = record
                count += 1
            # Every pinned count is nonzero, so this also rejects an empty partition.
            expected = self.SOURCE_ROWS[file.part]
            if count != expected:
                raise TimeFValidationError(f"{file.path} ({file.part}): expected {expected} rows, found {count}")
        dataset.set_task_stream([AnswerTask], partial(_iter_tasks, tuple(raw_refs), records))
        return dataset

    def _make_record(self, ref: _RowRef, row: _ParsedRow, loader: _ValueLoader) -> Record:
        """Build one six-channel IMU Record with its source provenance.

        Returns:
            A Record with lazy sensor values, provenance metadata, the row's task context, and its
            derived features as annotations.
        """
        channels = _PAMAP2_CHANNELS if ref.file.part == "pamap2_v2" else _CHANNELS
        signals = tuple(
            Signal.from_loader(
                name=name,
                spec=spec,
                time_axis=_AXIS,
                time_offsets_loader=_load_time_offsets,
                n_values=len(row.values[sensor]),
                loader=partial(loader, ref, sensor, dimension),
            )
            for spec, name, sensor, dimension in channels
        )
        record = Record(
            sources=(Source(name="IMU", signals=signals),),
            # Provenance only. The task context and the features below are annotations.
            metadata={
                "source_dataset": ref.file.part.rsplit("_", 1)[0],
                "source_partition": ref.file.part,
                "source_file": self.SOURCE_FILES[ref.file.part],
                "source_row": ref.index,
            },
        )
        context: dict[str, str] = {"activity_summary": row.summary}
        if row.location is not None:
            context["sensor_location"] = row.location
        for key, value in context.items():
            record.annotate(Annotation(key=key, value=value))
        # The derived features are Record annotations only. They are not task inputs, because they are long
        # NumPy text that the generator computed from the 120-reading source window, not from the 60 stored
        # readings.
        for heading in FEATURE_HEADINGS:
            record.annotate(Annotation(key=_feature_key(heading), value=row.features[heading]))
        return record


CONNECTOR = SensorCapsConnector
