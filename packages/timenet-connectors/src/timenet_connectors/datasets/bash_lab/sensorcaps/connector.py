"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The connector downloads the original JSONL files at a pinned commit through the HuggingFace base,
indexes rows by byte offset, and keeps just one decoded sensor row cached across the six channel
loaders, so the complete corpus is never held in memory. Tasks use the original system message
followed by the original derived-feature text as their prompt, with the activity summary and optional
sensor location as input annotations. Feature text is streamed rather than retained across the corpus.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Annotated, ClassVar, Self

import pyarrow as pa
from pydantic import Field, FiniteFloat, Strict, TypeAdapter, ValidationError, model_validator

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.readers.base import DataRow, NonBlankText, PublishedEnum
from timenet_connectors.sources.readers.jsonl import JsonlLocation, JsonlRowSource


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
_XYZValues = Annotated[list[Annotated[FiniteFloat, Strict()]], Field(min_length=3, max_length=3)]
_SENSOR_VALUES = TypeAdapter(Annotated[list[_XYZValues], Field(min_length=1)])


class _Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


_EXPECTED_ROLES = (_Role.SYSTEM, _Role.USER, _Role.ASSISTANT)


class _Message(DataRow):
    role: PublishedEnum[_Role]
    content: NonBlankText


class _ChatRow(DataRow):
    messages: Annotated[list[_Message], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def check_message_order(self) -> Self:
        """Require the published system, user and assistant order.

        Returns:
            The row after its message order has been checked.

        Raises:
            ValueError: If a role is out of order, for Pydantic to report as source validation.
        """
        for message, role in zip(self.messages, _EXPECTED_ROLES, strict=True):
            if message.role is not role:
                raise ValueError(f"expected a {role.value} message")
        return self


@dataclass(frozen=True)
class _RowRef:
    file: HubFileRef
    location: JsonlLocation

    @property
    def index(self) -> int:
        return self.location.line - 1

    @property
    def key(self) -> tuple[str, int]:
        return self.file.part, self.index

    @property
    def context(self) -> str:
        return f"{self.file.path} ({self.file.part}, row {self.index})"

    def read_row(self) -> "_ParsedRow":
        """Reread a typed source row and interpret its SensorCaps sections.

        Returns:
            Validated messages, features and paired sensor arrays.
        """
        entry = JsonlRowSource(self.file.path, _ChatRow).read_row(self.location)
        return _ParsedRow.from_source(entry.row, context=self.context)


@dataclass(frozen=True)
class _ParsedRow:
    system: str
    caption: str
    summary: str
    location: str | None
    features: str
    values: tuple[list[list[float]], list[list[float]]]

    @classmethod
    def from_source(cls, row: _ChatRow, *, context: str) -> Self:
        """Interpret validated chat messages using the pinned SensorCaps source rules.

        Returns:
            Captioning instructions, targets, descriptive fields and validated sensor arrays.

        Raises:
            TimeFValidationError: If the sensor sections violate the SensorCaps mapping.
        """
        try:
            system, user, assistant = row.messages
            sections, features = _user_sections(user.content)
            values = _section_values(sections)
            return cls(
                system.content, assistant.content, sections["Summary"], sections.get("Location"), features, values
            )
        except (ValueError, OverflowError, RecursionError) as exc:
            raise TimeFValidationError(f"{context}: {exc}") from exc


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


def _user_sections(user: str) -> tuple[dict[str, str], str]:
    """Separate structured fields from verbatim multiline feature text.

    Returns:
        Named source fields and the remaining original feature text.

    Raises:
        TimeFValidationError: If a structured field is repeated.
    """
    sections: dict[str, str] = {}
    feature_lines: list[str] = []
    for line in user.splitlines(keepends=True):
        heading, separator, value = line.partition(":")
        if separator and heading in {"Summary", "Accelerometer", "Gyroscope", "Location"}:
            if heading in sections:
                raise TimeFValidationError(f"repeated {heading} section")
            sections[heading] = value.strip()
        else:
            # NumPy statistics can span lines and contain complex numbers or ellipses.
            # Preserve their original text, including headings and whitespace.
            feature_lines.append(line)
    return sections, "".join(feature_lines)


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


def _iter_rows(file: HubFileRef) -> Iterator[tuple[_RowRef, _ParsedRow]]:
    """Interpret streamed typed rows without retaining their message text.

    Yields:
        Lightweight source locations and transient SensorCaps contents.
    """
    source = JsonlRowSource(file.path, _ChatRow)
    with source.open_rows() as rows:
        for entry in rows:
            ref = _RowRef(file, entry.location)
            yield ref, _ParsedRow.from_source(entry.row, context=ref.context)


class _ValueLoader:
    """Share one bounded sensor-row cache across a dataset's lazy channel loaders."""

    def __init__(self) -> None:
        self.cached: tuple[_RowRef, tuple[list[list[float]], list[list[float]]]] | None = None

    def __call__(self, ref: _RowRef, sensor: int, axis: int) -> pa.Array:
        cached = self.cached
        if cached is None or cached[0] != ref:
            values = ref.read_row().values
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


def _iter_tasks(files: tuple[HubFileRef, ...], records: Mapping[tuple[str, int], Record]) -> Iterator[AnswerTask]:
    """Re-read original system prompts and caption targets without retaining the corpus.

    Yields:
        A caption task with its instruction prompt and registered input Record.
    """
    for file in files:
        for ref, row in _iter_rows(file):
            record = records[ref.key]
            context = [
                annotation
                for annotation in record.annotations
                if annotation.key in {"activity_summary", "sensor_location"}
            ]
            yield AnswerTask(
                inputs=(record,),
                targets=(row.caption,),
                prompt=f"{row.system}\n\n{row.features}" if row.features.strip() else row.system,
                input_annotations=tuple(context),
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
            A dataset retaining its hierarchy, small annotations, and file-backed loaders.

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
            A Record with lazy sensor values and annotations from the original row.
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
        record = Record(sources=(Source(name="IMU", signals=signals),))
        source = ref.file.part.rsplit("_", 1)[0]
        metadata: dict[str, str | int] = {
            "source_dataset": source,
            "source_partition": ref.file.part,
            "source_file": self.SOURCE_FILES[ref.file.part],
            "source_row": ref.index,
            "activity_summary": row.summary,
        }
        if row.location is not None:
            metadata["sensor_location"] = row.location
        for key, value in metadata.items():
            record.annotate(Annotation(key=key, value=value))
        return record


CONNECTOR = SensorCapsConnector
