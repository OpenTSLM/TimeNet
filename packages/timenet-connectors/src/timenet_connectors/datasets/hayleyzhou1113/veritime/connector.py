"""Convert the pinned VeriTime JSON Lines files into lazy Records and one reasoning task per row.

Each distinct series becomes a Record whose signal rereads its first row on demand. Each row becomes a
task whose prompt is the authors' question and whose rationale is the published chain of thought.
README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that names
it in ``input_annotations``, and no VeriTime task names one.
"""

from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from fractions import Fraction
from functools import partial
import hashlib
import math
from pathlib import Path
import re
from typing import Any, ClassVar, NamedTuple, TypedDict

import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ScalarPredictionTask,
    Split,
    Task,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.json_stream import JsonCursor, read_entry


_OUTPUT = "output"
_SERIES_MARKER = "<ts><ts/>"
"""Where a question takes its series. The authors' scripts replace it with the series as text."""
_KNOWLEDGE_SEPARATOR = ", "
_MICROSECONDS_PER_SECOND = 1_000_000
# An inferential calculation label: a count, except one two-part row whose label is a decimal amplitude.
_QUANTITY = re.compile(r"[0-9]+(\.[0-9]+)?")


@unique
class _Family(StrEnum):
    """The paper's two task categories, one folder of the release each."""

    SCENARIO = "scenario"
    KNOWLEDGE = "knowledge"


@unique
class _Part(StrEnum):
    """One file of the release: a family's train or test rows."""

    SCENARIO_TRAIN = "scenario_train"
    SCENARIO_TEST = "scenario_test"
    KNOWLEDGE_TRAIN = "knowledge_train"
    KNOWLEDGE_TEST = "knowledge_test"

    @property
    def family(self) -> _Family:
        """Return the family whose rows the file holds."""
        return _Family(self.value.split("_")[0])

    @property
    def timenet_split(self) -> Split:
        """Return the TimeNet split of the file's rows: the authors' split."""
        return Split(self.value.split("_")[1])


@unique
class _SeriesKind(StrEnum):
    """What a series measures: a ChatTS series, or one knowledge task's signal."""

    SCENARIO = "scenario"
    CTU = "ctu"
    ECG = "ecg"
    EMG = "emg"
    RCW = "rcw"

    @property
    def family(self) -> _Family:
        """Return the family whose rows publish series of the kind."""
        return _Family.SCENARIO if self is _SeriesKind.SCENARIO else _Family.KNOWLEDGE


@unique
class _SourceTask(StrEnum):
    """A row's ``task``, as the release spells it: the paper's seven tasks."""

    ANOMALY_DETECTION = "Anomaly detection"
    SCENARIO_ATTRIBUTION = "Scenario attribution"
    INFERENTIAL_CALCULATION = "Inferential calculation"
    CTU = "CTU"
    ECG = "ECG"
    EMG = "EMG"
    RCW = "RCW"

    @property
    def kind(self) -> _SeriesKind:
        """Return the kind of series the task's rows publish."""
        scenario = (_SourceTask.ANOMALY_DETECTION, _SourceTask.SCENARIO_ATTRIBUTION)
        if self in scenario or self is _SourceTask.INFERENTIAL_CALCULATION:
            return _SeriesKind.SCENARIO
        return _SeriesKind(self.name.lower())

    @property
    def vocabulary_key(self) -> str:
        """Return the key of the dataset annotation that holds the task's classes, such as ``ecg_vocabulary``.

        A classification task states it as its ``target_schema``.
        """
        return f"{self.value.lower().replace(' ', '_')}_vocabulary"


_CLASSES = {
    # The authors' extract_label.py takes Yes or No from the ChatTS answer.
    _SourceTask.ANOMALY_DETECTION: ("Yes", "No"),
    # Each knowledge question lists its classes as options a), b), and so on, in this order.
    _SourceTask.CTU: ("Desktop", "Laptop"),
    _SourceTask.ECG: ("normal sinus rhythm", "atrial fibrillation", "other cardiac rhythms", "noise"),
    _SourceTask.EMG: ("Healthy", "Myopathy", "Neuropathy"),
    _SourceTask.RCW: ("Right Whale Present", "No Right Whale"),
}
"""The classes of each classification task, as the release spells them. Scenario attribution offers its own
options in each question and inferential calculation answers a count, so neither has one."""

_STEP_LABELS = {
    "step1_label": "The process label of step 1, the task intent, as published.",
    "step2_label": "The process label of step 2, the key patterns separated by semicolons, as published.",
    "step4_label": "The process label of step 4, the preliminary answer, as published.",
    "step6_label": "The process label of step 6, the final answer, as published.",
}
"""The process labels the authors' RL rewards score, and the description of each task annotation."""
_SOURCE_ANSWER = "source_answer"
_SOURCE_ANSWER_DESCRIPTION = "The ChatTS answer the authors extracted the label from, as published."
_DESCRIPTIONS = {**_STEP_LABELS, _SOURCE_ANSWER: _SOURCE_ANSWER_DESCRIPTION}
_REQUIRED = ("task", "question", "label", "cot", "timeseries2", *_STEP_LABELS)
_NULLABLE = ("step1_label", "step4_label")
"""The process labels that the release publishes as null for some scenario rows."""


class _KindSpec(NamedTuple):
    """What a kind of series measures and how its values are spaced."""

    signal: str
    spec: TimeSeriesSpec
    period_us: Fraction | None
    """The time between values as published, or None where the release gives no rate."""


# The ECG and RCW questions state 300 Hz and 2 kHz, but the series are every second and every eighth
# sample of those, so 150 Hz and 250 Hz. The authors' GitHub copy keeps the originals in its `timeseries`
# column for test rows only, so the train rows are assumed to be decimated the same way. No low-pass
# filter ran first, so RCW is aliased above 125 Hz. Supporting evidence: the median R-R interval of normal
# sinus rhythm rows is 132.5 samples, 68 bpm at 150 Hz, and the RCW average spectrum is flat (README.md).
# CTU states its 2-minute cadence. The ChatTS series are synthetic and EMG states no rate, so both are
# ordinal.
_KINDS: dict[_SeriesKind, _KindSpec] = {
    _SeriesKind.SCENARIO: _KindSpec(
        "series",
        TimeSeriesSpec(spec_type="chatts_series", name="ChatTS series", unit_value=None, dtype="float64"),
        None,
    ),
    # Each CTU series is z-normalised on its own (mean 0, standard deviation 0.9993), so its values are
    # dimensionless and not consumption, though the spec and signal keep the name of the quantity.
    _SeriesKind.CTU: _KindSpec(
        "consumption",
        TimeSeriesSpec(
            spec_type="electricity_consumption",
            name="Household electricity consumption",
            unit_value=None,
            dtype="float64",
        ),
        Fraction(120 * _MICROSECONDS_PER_SECOND),
    ),
    _SeriesKind.ECG: _KindSpec(
        "ecg",
        TimeSeriesSpec(spec_type="ecg", name="Single-lead ECG", unit_value=None, dtype="float64"),
        Fraction(_MICROSECONDS_PER_SECOND, 150),
    ),
    # The EMG question states the amplitude in mV; the others state no unit.
    _SeriesKind.EMG: _KindSpec(
        "emg",
        TimeSeriesSpec(spec_type="emg", name="Needle EMG", unit_value=ureg.millivolt, dtype="float64"),
        None,
    ),
    _SeriesKind.RCW: _KindSpec(
        "audio",
        TimeSeriesSpec(spec_type="hydrophone_audio", name="Hydrophone audio", unit_value=None, dtype="float64"),
        Fraction(_MICROSECONDS_PER_SECOND, 250),
    ),
}


class _Location(NamedTuple):
    """Where a row lives: its file, part and position in the file, and its byte range."""

    path: Path
    part: _Part
    index: int
    """The row's position in its file, from 0."""
    offset: int
    length: int
    """The row's UTF-8 byte range in its file."""

    def __str__(self) -> str:
        """Return where the row is, for error messages: its file and position."""
        return f"{self.path} (row {self.index})"


class _Label(NamedTuple):
    """A task annotation of a row: a process label or the ChatTS answer."""

    key: str
    value: str


class _SeriesKey(NamedTuple):
    """What tells distinct series apart: rows with an equal key share a Record."""

    kind: _SeriesKind
    digest: bytes
    """A hash of the series' float64 values."""


@dataclass(frozen=True, slots=True)
class _Row:
    """A row's checked fields and where its series is, but not its values."""

    location: _Location
    task: _SourceTask
    question: str
    label: str
    cot: str
    annotations: tuple[_Label, ...]
    """The task annotations: each process label that is not null, then the ChatTS answer."""
    n_values: int
    series_key: _SeriesKey


class _TaskFields(TypedDict):
    """The arguments every VeriTime task shares."""

    inputs: tuple[Record, ...]
    prompt: str
    rationale: str
    split: Split
    metadata: dict[str, JsonValue]


def _digest(values: list[float]) -> bytes:
    return hashlib.blake2b(array("d", values).tobytes(), digest_size=16).digest()


def _parse_series(published: Any, family: _Family) -> list[float]:
    """Decode a row's ``timeseries2`` in its family's format.

    A scenario row holds a list with one list of numbers. A knowledge row holds one string of numbers
    separated by ``", "``, which the authors' scripts paste into the question as it is.

    Returns:
        The series' values, in order.

    Raises:
        ValueError: If the value is not a nonempty series of finite numbers in the family's format.
        OverflowError: If an integer is too large for float64.
    """  # noqa: DOC502 - raised by float()
    if family is _Family.SCENARIO:
        if not (isinstance(published, list) and len(published) == 1 and isinstance(published[0], list)):
            raise ValueError("a scenario timeseries2 must be a list holding one list of numbers")
        # A JSON true is an int to Python, and is not a number here.
        if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in published[0]):
            raise ValueError("a scenario timeseries2 must hold numbers")
        values = [float(value) for value in published[0]]
    else:
        if not isinstance(published, str):
            raise ValueError("a knowledge timeseries2 must be a string of numbers")
        try:
            values = [float(token) for token in published.split(_KNOWLEDGE_SEPARATOR)]
        except ValueError:
            raise ValueError(
                f"a knowledge timeseries2 must hold numbers separated by {_KNOWLEDGE_SEPARATOR!r}"
            ) from None
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("timeseries2 must hold at least one value, and every value must be finite")
    return values


def _text(value: Any, name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{name} must be nonblank text, got {value!r}")


def _parse_task(value: Any, part: _Part) -> _SourceTask:
    """Read a row's task and check that it belongs to the row's file.

    Returns:
        The task.

    Raises:
        ValueError: If the task is not text, is unknown, or is of the other family's file.
    """
    name = _text(value, "task")
    try:
        task = _SourceTask(name)
    except ValueError:
        raise ValueError(f"unknown task {name!r}") from None
    if task.kind.family is not part.family:
        raise ValueError(f"a {part.family} file holds a {name!r} row")
    return task


def _check_answer(task: _SourceTask, question: str, label: str) -> None:
    """Check a row's question and label against its task.

    Raises:
        ValueError: If the question lacks its series marker, the label is not one the task allows, or an
            inferential calculation label is not a number.
    """
    if question.count(_SERIES_MARKER) != 1:
        raise ValueError(f"a question must hold {_SERIES_MARKER!r} exactly once")
    if task in _CLASSES and label not in _CLASSES[task]:
        raise ValueError(f"label {label!r} is not one of {_CLASSES[task]}")
    if task is _SourceTask.INFERENTIAL_CALCULATION and not _QUANTITY.fullmatch(label):
        raise ValueError(f"an inferential calculation label must be a number, got {label!r}")


def _parse_annotations(value: dict[str, Any], family: _Family) -> tuple[_Label, ...]:
    """Read a row's task annotations: its process labels and, on a scenario row, the ChatTS answer.

    A process label published as null has no annotation, since an annotation must say something.

    Returns:
        The annotations, in the order the labels have.

    Raises:
        ValueError: If a label or the answer is not nonblank text.
    """  # noqa: DOC502 - raised by _text
    labels = [
        _Label(name, _text(value[name], name))
        for name in _STEP_LABELS
        if not (value[name] is None and name in _NULLABLE)
    ]
    if family is _Family.SCENARIO:
        labels.append(_Label(_SOURCE_ANSWER, _text(value[_OUTPUT], _OUTPUT)))
    return tuple(labels)


def _parse_row(value: Any, location: _Location) -> _Row:
    """Check a published row and keep its small fields.

    Returns:
        The row, with its series' length and digest.

    Raises:
        ValueError: If a field is missing or malformed, the task is unknown or not of the row's file, the
            question lacks its series marker, the label is not one the task allows, or the series is
            malformed.
        OverflowError: If a series integer is too large for float64.
    """  # noqa: DOC502 - raised by float()
    if not isinstance(value, dict):
        raise ValueError("expected a row object")
    family = location.part.family
    required = (*_REQUIRED, _OUTPUT) if family is _Family.SCENARIO else _REQUIRED
    if missing := [name for name in required if name not in value]:
        raise ValueError(f"missing fields {missing}")
    task = _parse_task(value["task"], location.part)
    question, label, cot = (_text(value[name], name) for name in ("question", "label", "cot"))
    _check_answer(task, question, label)
    annotations = _parse_annotations(value, family)
    values = _parse_series(value["timeseries2"], family)
    series_key = _SeriesKey(task.kind, _digest(values))
    return _Row(location, task, question, label, cot, annotations, len(values), series_key)


def _read_rows(path: Path, part: _Part) -> Iterator[_Row]:
    """Check every row of one file, in file order, and keep each row's byte range.

    Yields:
        Each row.

    Raises:
        TimeFValidationError: If the file isn't JSON values or a row is malformed, with file and position.
    """
    with JsonCursor.open(path) as cursor:
        index = 0
        while True:
            try:
                if not cursor.peek():
                    return
                value, offset, length = cursor.value()
                row = _parse_row(value, _Location(path, part, index, offset, length))
            except (ValueError, OverflowError, RecursionError) as exc:
                raise TimeFValidationError(f"{path} (row {index}): {exc}") from exc
            yield row
            index += 1


def _load_series(row: _Row) -> pa.Array:
    """Reread a row's series from its file.

    Returns:
        The values as float64, checked against the converted row.

    Raises:
        TimeFFormatError: If the row no longer holds the series it held at conversion, as when the file changed.
    """
    location = row.location
    value = read_entry(location.path, location.offset, location.length)
    try:
        values = _parse_series(value.get("timeseries2"), location.part.family)
    except (AttributeError, ValueError, OverflowError, RecursionError) as exc:
        raise TimeFFormatError(f"{location}: {exc}") from exc
    if len(values) != row.n_values or _digest(values) != row.series_key.digest:
        raise TimeFFormatError(f"{location}: the series differs from the converted row; the file changed")
    return pa.array(values, type=pa.float64())


def _record(rows: list[_Row]) -> Record:
    """Build the Record of one series, whose signal rereads the first of the rows that publish it.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    first = rows[0]
    kind = _KINDS[first.series_key.kind]
    axis: TimeAxis = OrdinalAxis() if kind.period_us is None else RegularAxis(period_us=kind.period_us)
    signal = Signal.from_loader(
        name=kind.signal,
        spec=kind.spec,
        time_axis=axis,
        n_values=first.n_values,
        loader=partial(_load_series, first),
    )
    # Scenario rows can publish one series from either split.
    metadata: dict[str, JsonValue] = {
        "kind": first.series_key.kind.value,
        "rows": [{"part": row.location.part.value, "row": row.location.index} for row in rows],
    }
    return Record(sources=(Source(name=kind.spec.name, signals=(signal,)),), metadata=metadata)


def _task(row: _Row, record: Record) -> Task:
    """Build the row's task. The type follows the answer.

    A fixed class gives a ClassificationTask, an option's text an AnswerTask and a number a
    ScalarPredictionTask.

    Returns:
        The task, with the process labels and, on a scenario row, the ChatTS answer as annotations.
    """
    fields = _TaskFields(
        inputs=(record,),
        prompt=row.question,
        rationale=row.cot,
        split=row.location.part.timenet_split,
        metadata={"source_task": row.task.value, "part": row.location.part.value, "row": row.location.index},
    )
    task: Task
    if row.task in _CLASSES:
        task = ClassificationTask(targets=(row.label,), target_schema=row.task.vocabulary_key, **fields)
    elif row.task is _SourceTask.INFERENTIAL_CALCULATION:
        # A count, except one row whose two-part question also asks an amplitude: its label is a float.
        quantity = int(row.label) if row.label.isdecimal() else float(row.label)
        task = ScalarPredictionTask(targets=(quantity,), **fields)
    else:
        # The options differ by question, so there is no shared vocabulary. The target is the label as published.
        task = AnswerTask(targets=(row.label,), **fields)
    for label in row.annotations:
        task.annotate(Annotation(key=label.key, value=label.value, description=_DESCRIPTIONS[label.key]))
    return task


def _to_part(name: str) -> _Part:
    """Read the name of a downloaded file's part.

    Returns:
        The part.

    Raises:
        TimeFValidationError: If the name is not one of the release's four parts.
    """
    try:
        return _Part(name)
    except ValueError:
        raise TimeFValidationError(
            f"{name!r} is not a VeriTime part, expected one of {[p.value for p in _Part]}"
        ) from None


class VeriTimeConnector(BaseHuggingFaceConnector):
    """Build VeriTime from the pinned JSON Lines files of its Hugging Face release."""

    HF_REPO = "HayleyZhou1113/VeriTime"
    SOURCE_REVISION = "fc88cb1305a96a208093e13b69d766b19c84c99b"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        _Part.SCENARIO_TRAIN.value: "Dataset/Scenario/scenario_shuffled_train.jsonl",
        _Part.SCENARIO_TEST.value: "Dataset/Scenario/scenario_shuffled_test.jsonl",
        # "knowldege" is the release's own spelling.
        _Part.KNOWLEDGE_TRAIN.value: "Dataset/Knowledge/knowldege_shuffled_train.jsonl",
        _Part.KNOWLEDGE_TEST.value: "Dataset/Knowledge/knowledge_test.jsonl",
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per distinct series and a stream of one task per row.

        Pass one reads each file once, checks every row, and keeps its small fields, its byte range and a
        digest of its series. Rows that publish equal values of the same kind share one Record. Pass two
        builds the Records and the tasks. The series values stay in the files until written, so the files
        must stay available until then.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If a reference names no VeriTime part.
        """  # noqa: DOC502 - a file's rows are checked by _read_rows
        rows = [row for ref in raw_refs for row in _read_rows(ref.path, _to_part(ref.part))]
        by_series: dict[_SeriesKey, list[_Row]] = {}
        for row in rows:
            by_series.setdefault(row.series_key, []).append(row)
        records = {key: _record(series) for key, series in by_series.items()}

        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", self.SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value))
        for record in records.values():
            dataset.add_record(record=record)
        present = {row.task for row in rows}
        for task, classes in _CLASSES.items():
            if task in present:
                dataset.annotate(Annotation(key=task.vocabulary_key, value=list(classes)))

        def tasks() -> Iterator[Task]:
            for row in rows:
                yield _task(row, records[row.series_key])

        dataset.set_task_stream((ClassificationTask, AnswerTask, ScalarPredictionTask), tasks)
        return dataset


CONNECTOR = VeriTimeConnector
