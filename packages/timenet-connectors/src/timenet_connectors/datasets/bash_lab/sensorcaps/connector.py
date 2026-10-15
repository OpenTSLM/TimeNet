"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The row-based HuggingFace base accumulates the complete corpus and loses file provenance. This
connector instead downloads original JSONL files, indexes rows by byte offset, and keeps just one
decoded sensor row cached across the six channel loaders. Full generation messages travel through
streamed task metadata, never through the writer's accumulated annotation list.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec
from timenet_connectors.sources.huggingface_hub import hub_snapshot


HF_REPO = "BASH-Lab/SensorCaps"
# This repository commit contains both caption versions; v1/v2 select files within the same snapshot.
# Pin the original JSONL files together: a mutable branch can silently change rebuilt datasets.
SOURCE_REVISION = "4242e07feaacb0ee3aff86182a62914fa7abacef"
# Counts are shared across caption versions; PAMAP2 was released only for v2.
SOURCE_ROWS = {"hhar": 9166, "motion": 4534, "pamap2": 9672, "shoaib": 10500, "uci": 2088}
_SPECS = (
    # Upstream rounds IMU values to six decimal places. Use compact float32 storage, accepting
    # normal binary floating-point rounding rather than implying exact decimal preservation.
    TimeSeriesSpec(spec_type="accelerometer", name="Accelerometer", unit_value=None, dtype="float32"),
    TimeSeriesSpec(spec_type="gyroscope", name="Gyroscope", unit_value=None, dtype="float32"),
)
_AXES = ("x", "y", "z")
_ROLES = ("system", "user", "assistant")


class _BuildOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    caption_version: Literal["v1", "v2"] = "v2"


def partition_rows(caption_version: Literal["v1", "v2"]) -> dict[str, int]:
    """Select source partitions for one caption version.

    Returns:
        Expected counts keyed by original source/version partition.
    """
    return {
        f"{source}_{caption_version}": count
        for source, count in SOURCE_ROWS.items()
        if source != "pamap2" or caption_version == "v2"
    }


@dataclass(frozen=True)
class SensorCapsFileRef:
    """One local JSONL file and its upstream source/version partition."""

    path: Path
    partition: str
    expected_rows: int | None = None
    """Pinned inventory count for downloads; local fixtures may omit it."""

    @property
    def filename(self) -> str:
        """Return the original repository-relative filename."""
        return f"data/{self.partition}-00000-of-00001.jsonl"


@dataclass(frozen=True)
class _RowRef:
    file: SensorCapsFileRef
    index: int
    offset: int

    @property
    def id(self) -> str:
        return f"sensorcaps-{self.file.partition.replace('_', '-')}-{self.index:06d}"

    @property
    def context(self) -> str:
        return f"{self.file.path} ({self.file.partition}, row {self.index})"


@dataclass(frozen=True)
class _ParsedRow:
    system: str
    user: str
    caption: str
    summary: str
    location: str | None
    values: tuple[list[list[float]], list[list[float]]]


def _sensor_values(text: str, name: str) -> list[list[float]]:
    """Decode and validate one time-major numeric xyz array.

    Returns:
        Validated numeric triples in source order.

    Raises:
        TimeFValidationError: If the decoded values are empty, ragged, or nonnumeric.
    """
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise TimeFValidationError(f"{name} must be a nonempty list of xyz triples")
    for sample in values:
        if not isinstance(sample, list) or len(sample) != len(_AXES):
            raise TimeFValidationError(f"{name} must contain exactly three values per timestep")
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in sample):
            raise TimeFValidationError(f"{name} must contain only finite numbers")
    return values


def _parse_row(raw: bytes, context: str) -> _ParsedRow:
    """Parse a source row, adding file and row context to decoding failures.

    Returns:
        Messages, descriptive fields, and validated sensor arrays for just this row.

    Raises:
        TimeFValidationError: If the JSON, messages, or sensor sections are malformed.
    """
    try:
        row = json.loads(raw)
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) != len(_ROLES):
            raise TimeFValidationError("expected system, user, and assistant messages")
        contents: list[str] = []
        for message, role in zip(cast("list[dict[str, Any]]", messages), _ROLES, strict=True):
            if (
                not isinstance(message, dict)
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise TimeFValidationError(f"expected a nonempty {role} message")
            contents.append(message["content"])
        system, user, caption = contents
        sections: dict[str, str] = {}
        for line in user.splitlines():
            heading, separator, value = line.partition(":")
            if separator and heading in {"Summary", "Accelerometer", "Gyroscope", "Location"}:
                if heading in sections:
                    raise TimeFValidationError(f"repeated {heading} section")
                sections[heading] = value.strip()
        if not sections.get("Summary"):
            raise TimeFValidationError("missing or empty Summary section")
        for heading in ("Accelerometer", "Gyroscope"):
            if heading not in sections:
                raise TimeFValidationError(f"missing {heading} section")
        accelerometer = _sensor_values(sections["Accelerometer"], "Accelerometer")
        gyroscope = _sensor_values(sections["Gyroscope"], "Gyroscope")
        if len(accelerometer) != len(gyroscope):
            raise TimeFValidationError("Accelerometer and Gyroscope lengths differ")
        if "Location" in sections and not sections["Location"]:
            raise TimeFValidationError("empty Location section")
        return _ParsedRow(
            system, user, caption, sections["Summary"], sections.get("Location"), (accelerometer, gyroscope)
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
        # Use the local values even if another consumer replaces the shared cache during this read.
        return pa.array((sample[axis] for sample in values[sensor]), type=pa.float32())


def _iter_tasks(files: tuple[SensorCapsFileRef, ...], records: Mapping[str, Record]) -> Iterator[AnswerTask]:
    """Re-read captions and full generation provenance without retaining the corpus.

    Yields:
        A caption task referencing the registered Record and original generation messages.
    """
    for file in files:
        for ref, row in _iter_rows(file):
            yield AnswerTask(
                id=f"{ref.id}-caption",
                inputs=(records[ref.id],),
                targets=(row.caption,),
                metadata={"system_message": row.system, "user_message": row.user},
            )


class SensorCapsConnector(BaseConnector[SensorCapsFileRef]):
    """Build the selected SensorCaps caption version with lazy values and streamed captions."""

    def __init__(self) -> None:
        super().__init__()
        try:
            self._options = _BuildOptions.model_validate(self.metadata().build_options)
        except ValueError as exc:
            raise TimeFValidationError(f"{self._card_path()}: invalid SensorCaps build_options: {exc}") from exc

    @property
    def partitions(self) -> dict[str, int]:
        """Expected source partitions for the card's selected caption version."""
        return partition_rows(self._options.caption_version)

    def download(self, cache_dir: Path) -> list[SensorCapsFileRef]:
        """Download the selected original JSONL files at one immutable Hub revision.

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
                tuple(f"data/{partition}-00000-of-00001.jsonl" for partition in self.partitions),
            )
        except ImportError as exc:
            raise ImportError(
                "SensorCaps needs huggingface_hub from its requirements.txt; "
                "run the build without --no-isolation or install the connector requirements with uv"
            ) from exc
        return [
            SensorCapsFileRef(
                snapshot / f"data/{partition}-00000-of-00001.jsonl",
                partition,
                count,
            )
            for partition, count in self.partitions.items()
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
        if any(ref.partition not in self.partitions for ref in raw_refs):
            raise TimeFValidationError("unknown SensorCaps source partition for the selected caption version")
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            ("source_repository", HF_REPO),
            ("source_revision", SOURCE_REVISION),
            ("declared_sampling_rate_hz", 10),
            ("caption_version", self._options.caption_version),
        ):
            dataset.annotate(Annotation(key=key, value=value, id=f"sensorcaps-{key}"))
        records: dict[str, Record] = {}
        loader = _ValueLoader()
        for file in raw_refs:
            count = 0
            for ref, row in _iter_rows(file):
                axis = OrdinalAxis()
                signals = tuple(
                    Signal.from_loader(
                        id=f"{ref.id}-{spec.spec_type}-{name}",
                        name=f"{spec.spec_type}_{name}",
                        spec=spec,
                        time_axis=axis,
                        n_values=len(row.values[sensor]),
                        loader=partial(loader, ref, sensor, dimension),
                    )
                    for sensor, spec in enumerate(_SPECS)
                    for dimension, name in enumerate(_AXES)
                )
                record = Record(
                    record_id=ref.id,
                    sources=(Source(id=f"{ref.id}-imu", name="IMU", signals=signals),),
                )
                source, version = file.partition.rsplit("_", 1)
                metadata: dict[str, str | int] = {
                    "source_dataset": source,
                    "caption_version": version,
                    "source_partition": file.partition,
                    "source_file": file.filename,
                    "source_row": ref.index,
                    "activity_summary": row.summary,
                }
                if row.location is not None:
                    metadata["sensor_location"] = row.location
                for key, value in metadata.items():
                    record.annotate(Annotation(key=key, value=value, id=f"{ref.id}-{key}"))
                dataset.add_record(record=record)
                records[record.id] = record
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
