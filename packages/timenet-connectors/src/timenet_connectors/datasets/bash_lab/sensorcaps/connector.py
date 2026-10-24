"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The row-based HuggingFace base accumulates the complete corpus and loses file provenance. This
connector instead downloads original JSONL files, indexes rows by byte offset, and keeps just one
decoded sensor row cached across the six channel loaders. Tasks use the original system message
followed by the original derived-feature text as their prompt, with the activity summary and
optional sensor location as input annotations. Feature text is streamed rather than retained
across the corpus.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import Annotated

import pyarrow as pa
from pydantic import Field, FiniteFloat, JsonValue, Strict, TypeAdapter, ValidationError

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.sources.huggingface_hub import hub_snapshot


HF_REPO = "BASH-Lab/SensorCaps"
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
# Pin the original JSONL files together: a mutable branch can silently change rebuilt datasets.
SOURCE_REVISION = "4242e07feaacb0ee3aff86182a62914fa7abacef"
# Pinned inventory of the original captioning files.
SOURCE_ROWS = {"hhar_v2": 9166, "motion_v2": 4534, "pamap2_v2": 9672, "shoaib_v2": 10500, "uci_v2": 2088}
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
_ROLES = ("system", "user", "assistant")
_XYZValues = Annotated[list[Annotated[FiniteFloat, Strict()]], Field(min_length=3, max_length=3)]
_SENSOR_VALUES = TypeAdapter(Annotated[list[_XYZValues], Field(min_length=1)])


def _source_filename(partition: str) -> str:
    """Return the original repository-relative filename for a source partition.

    Returns:
        The partition's JSONL path within the pinned snapshot.
    """
    return f"data/{partition}-00000-of-00001.jsonl"


@dataclass(frozen=True)
class SensorCapsFileRef:
    """One local JSONL file and its upstream source partition."""

    path: Path
    partition: str
    expected_rows: int | None = None
    """Pinned inventory count for downloads; local fixtures may omit it."""

    @property
    def filename(self) -> str:
        """Return the original repository-relative filename."""
        return _source_filename(self.partition)


@dataclass(frozen=True)
class _RowRef:
    file: SensorCapsFileRef
    index: int
    offset: int

    @property
    def key(self) -> tuple[str, int]:
        return self.file.partition, self.index

    @property
    def context(self) -> str:
        return f"{self.file.path} ({self.file.partition}, row {self.index})"


@dataclass(frozen=True)
class _ParsedRow:
    system: str
    caption: str
    summary: str
    location: str | None
    features: str
    values: tuple[list[list[float]], list[list[float]]]


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


def _iter_rows(file: SensorCapsFileRef) -> Iterator[tuple[_RowRef, _ParsedRow]]:
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


def _make_record(ref: _RowRef, row: _ParsedRow, loader: _ValueLoader) -> Record:
    """Build one six-channel IMU Record with its source provenance.

    Returns:
        A Record with lazy sensor values and annotations from the original row.
    """
    channels = _PAMAP2_CHANNELS if ref.file.partition == "pamap2_v2" else _CHANNELS
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
    source = ref.file.partition.rsplit("_", 1)[0]
    metadata: dict[str, str | int] = {
        "source_dataset": source,
        "source_partition": ref.file.partition,
        "source_file": ref.file.filename,
        "source_row": ref.index,
        "activity_summary": row.summary,
    }
    if row.location is not None:
        metadata["sensor_location"] = row.location
    for key, value in metadata.items():
        record.annotate(Annotation(key=key, value=value))
    return record


def _iter_tasks(
    files: tuple[SensorCapsFileRef, ...], records: Mapping[tuple[str, int], Record]
) -> Iterator[AnswerTask]:
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


class SensorCapsConnector(BaseConnector[SensorCapsFileRef]):
    """Build SensorCaps with lazy values and streamed captions."""

    def download(self, cache_dir: Path) -> list[SensorCapsFileRef]:  # noqa: PLR6301 — BaseConnector interface
        """Download the original JSONL files at one immutable Hub revision.

        Returns:
            Lightweight local file references, with their source partitions intact.

        Raises:
            ImportError: If the connector's Hugging Face dependency is missing.
        """
        try:
            snapshot = hub_snapshot(
                HF_REPO,
                SOURCE_REVISION,
                cache_dir,
                tuple(_source_filename(partition) for partition in SOURCE_ROWS),
            )
        except ImportError as exc:
            raise ImportError(
                "SensorCaps needs huggingface_hub from its requirements.txt; "
                "run the build without --no-isolation or install the connector requirements with uv"
            ) from exc
        return [
            SensorCapsFileRef(
                snapshot / _source_filename(partition),
                partition,
                count,
            )
            for partition, count in SOURCE_ROWS.items()
        ]

    def convert(self, raw_refs: list[SensorCapsFileRef]) -> TimeFDataset:
        """Index local rows into six-channel Records and a re-iterable caption stream.

        Returns:
            A dataset retaining its hierarchy, small annotations, and file-backed loaders.

        Raises:
            TimeFValidationError: If references or source rows violate the SensorCaps mapping.
        """
        if not raw_refs or len({ref.partition for ref in raw_refs}) != len(raw_refs):
            raise TimeFValidationError("SensorCaps requires nonempty, distinct partition references")
        if any(ref.partition not in SOURCE_ROWS for ref in raw_refs):
            raise TimeFValidationError("unknown SensorCaps source partition")
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            ("source_repository", HF_REPO),
            ("source_revision", SOURCE_REVISION),
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
                record = _make_record(ref, row, loader)
                dataset.add_record(record=record)
                records[ref.key] = record
                count += 1
            if not count:
                raise TimeFValidationError(f"{file.path} ({file.partition}): empty source partition")
            if file.expected_rows is not None and count != file.expected_rows:
                raise TimeFValidationError(
                    f"{file.path} ({file.partition}): expected {file.expected_rows} rows, found {count}"
                )
        dataset.set_task_stream([AnswerTask], partial(_iter_tasks, tuple(raw_refs), records))
        return dataset


CONNECTOR = SensorCapsConnector
