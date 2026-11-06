"""Validate source rows and retain checked descriptors for lazy rereads."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import hashlib
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
from pydantic import Field

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import (
    CLASSES,
    KINDS,
    KNOWLEDGE_SEPARATOR,
    QUANTITY_PATTERN,
    SERIES_MARKER,
    SOURCE_ANSWER,
    STEP_LABEL_DESCRIPTIONS,
    Family,
    Part,
    SeriesKind,
    SourceTask,
)
from timenet_connectors.sources.jsonl import JsonlLocation, iter_jsonl_rows, read_jsonl_row
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.sources.rows import NonBlankText, SourceRow
from timenet_connectors.sources.signals import numeric_signal


class RowLocation(NamedTuple):
    """Where a row lives: its file, part and position in the file, and its byte range."""

    location: JsonlLocation
    part: Part

    def __str__(self) -> str:
        return self.location.context


class ProcessAnnotation(NamedTuple):
    """A task annotation of a row: a process label or the ChatTS answer."""

    key: str
    value: str


class SeriesKey(NamedTuple):
    """What tells distinct series apart: rows with an equal key share a Record."""

    kind: SeriesKind
    digest: bytes
    """A hash of the series' float64 values."""


@dataclass(frozen=True, slots=True)
class Row:
    """A row's checked fields and where its series is, but not its values."""

    location: RowLocation
    task: SourceTask
    question: str
    label: str
    cot: str
    annotations: tuple[ProcessAnnotation, ...]
    """The task annotations: each process label that is not null, then the ChatTS answer."""
    n_values: int
    series_key: SeriesKey


def _digest(values: pa.Array) -> bytes:
    return hashlib.blake2b(values.to_numpy().tobytes(), digest_size=16).digest()


def _parse_task(value: str, part: Part) -> SourceTask:
    """Read a row's task and check that it belongs to the row's file.

    Returns:
        The task.

    Raises:
        TimeNetConnectorError: If the task is not text, is unknown, or is of the other family's file.
    """
    name = value
    try:
        task = SourceTask(name)
    except ValueError:
        raise TimeNetConnectorError(f"unknown task {name!r}") from None
    if task.kind.family is not part.family:
        raise TimeNetConnectorError(f"a {part.family} file holds a {name!r} row")
    return task


def _check_answer(task: SourceTask, question: str, label: str) -> None:
    """Check a row's question and label against its task.

    Raises:
        TimeNetConnectorError: If the question lacks its series marker, the label is not one the task allows, or an
            inferential calculation label is not a number.
    """
    if question.count(SERIES_MARKER) != 1:
        raise TimeNetConnectorError(f"a question must hold {SERIES_MARKER!r} exactly once")
    if task in CLASSES and label not in CLASSES[task]:
        raise TimeNetConnectorError(f"label {label!r} is not one of {CLASSES[task]}")
    if task is SourceTask.INFERENTIAL_CALCULATION and not QUANTITY_PATTERN.fullmatch(label):
        raise TimeNetConnectorError(f"an inferential calculation label must be a number, got {label!r}")


class _PublishedRow(SourceRow):
    """The metadata columns shared by the two published VeriTime families."""

    task: NonBlankText
    question: NonBlankText
    label: NonBlankText
    cot: NonBlankText
    step1_label: NonBlankText | None
    step2_label: NonBlankText
    step4_label: NonBlankText | None
    step6_label: NonBlankText
    timeseries2: object


class _ScenarioRow(_PublishedRow):
    """Scenario metadata with its source answer and one nested numeric series."""

    output: NonBlankText
    timeseries2: list[list[int | float]] = Field(min_length=1, max_length=1)


class _KnowledgeRow(_PublishedRow):
    """Knowledge metadata with its comma-separated published measurements."""

    timeseries2: NonBlankText


def _parse_series(row: _ScenarioRow | _KnowledgeRow, kind: SeriesKind) -> pa.Array:
    """Decode a typed row's series and validate its lossless float64 samples.

    Returns:
        The numeric samples in source order.

    Raises:
        TimeNetConnectorError: If the encoded series or its numeric values are invalid.
    """
    if isinstance(row, _ScenarioRow):
        values = row.timeseries2[0]
    else:
        try:
            values = [float(token) for token in row.timeseries2.split(KNOWLEDGE_SEPARATOR)]
        except ValueError as exc:
            raise TimeNetConnectorError(
                f"a knowledge timeseries2 must hold numbers separated by {KNOWLEDGE_SEPARATOR!r}"
            ) from exc
    if not values:
        raise TimeNetConnectorError("timeseries2 must hold at least one value")
    return numeric_signal(values, "timeseries2", spec=KINDS[kind].spec)


def _row_model(part: Part) -> type[_ScenarioRow] | type[_KnowledgeRow]:
    """Return the typed source schema for a file's family."""
    return _ScenarioRow if part.family is Family.SCENARIO else _KnowledgeRow


def _parse_annotations(value: _PublishedRow) -> tuple[ProcessAnnotation, ...]:
    """Return non-null process labels and the scenario's published source answer."""
    labels = [
        ProcessAnnotation(name, text) for name in STEP_LABEL_DESCRIPTIONS if (text := getattr(value, name)) is not None
    ]
    if isinstance(value, _ScenarioRow):
        labels.append(ProcessAnnotation(SOURCE_ANSWER, value.output))
    return tuple(labels)


def _parse_row(value: _ScenarioRow | _KnowledgeRow, location: RowLocation) -> Row:
    """Return a checked row descriptor while leaving series values in the source file."""
    task = _parse_task(value.task, location.part)
    _check_answer(task, value.question, value.label)
    values = _parse_series(value, task.kind)
    return Row(
        location,
        task,
        value.question,
        value.label,
        value.cot,
        _parse_annotations(value),
        len(values),
        SeriesKey(task.kind, _digest(values)),
    )


def read_rows(path: Path, part: Part) -> Iterator[Row]:
    """Yield checked descriptors with physical JSONL byte ranges for lazy rereading.

    Raises:
        TimeNetConnectorError: If a source row is invalid.
    """
    for located in iter_jsonl_rows(path, _row_model(part)):
        place = located.location
        location = RowLocation(place, part)
        try:
            yield _parse_row(located.value, location)
        except (ValueError, OverflowError, RecursionError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{location}: {exc}") from exc


def _read_series(row: Row) -> pa.Array:
    """Return validated measurements from the row's recorded location.

    Raises:
        TimeNetConnectorError: If the source row cannot be read or its measurements are invalid.
    """
    location = row.location
    value = read_jsonl_row(location.location, _row_model(location.part)).value
    try:
        return _parse_series(value, row.series_key.kind)
    except (ValueError, OverflowError, RecursionError, TimeNetConnectorError) as exc:
        raise TimeNetConnectorError(f"{location}: {exc}") from exc


def load_series(row: Row) -> pa.Array:
    """Return measurements that match the length and digest checked during conversion."""
    return checked_reread(
        partial(_read_series, row),
        fingerprint=lambda values: (len(values), _digest(values)),
        expected=(row.n_values, row.series_key.digest),
        context=str(row.location),
    )


def to_part(name: str) -> Part:
    """Read the name of a downloaded file's part.

    Returns:
        The part.

    Raises:
        TimeNetConnectorError: If the name is not one of the release's four parts.
    """
    try:
        return Part(name)
    except ValueError:
        raise TimeNetConnectorError(
            f"{name!r} is not a VeriTime part, expected one of {[p.value for p in Part]}"
        ) from None
