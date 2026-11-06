"""Convert the pinned TSAQA parquet files into lazy Records and one question task per row.

Each row's series become Records: its input series, then each candidate series its question prints,
which the prompt shows as ``<series N>``. A true-or-false row becomes a ClassificationTask, a
multiple-choice or puzzle row an AnswerTask, each with the published answer as target. README.md
explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that
names it in ``input_annotations``.
"""

from array import array
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any, ClassVar, NamedTuple, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from timenet.dataset import Record, Signal, SignalChannel, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types import Annotation, AnswerTask, ClassificationTask, Label, LabelSet, Split, Task, TimeSeriesSpec
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.parquet import iter_parquet_rows
from timenet_connectors.sources.rows import SourceRow, validate_row
from timenet_connectors.sources.signals import numeric_signal


@unique
class _SplitFile(StrEnum):
    """A split file of the release, named as in the Hub repository without ``.parquet``."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

    @property
    def filename(self) -> str:
        return f"{self.value}.parquet"


_SPLITS = {_SplitFile.TRAIN: Split.TRAIN, _SplitFile.VALIDATION: Split.VALIDATION, _SplitFile.TEST: Split.TEST}
"""The TimeNet split of each split file."""


@unique
class _Task(StrEnum):
    """The six source tasks of the release, as its ``task`` column names them."""

    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    COMPARISON = "comparison"
    DATA_TRANSFORMATION = "data_transformation"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"


@unique
class _Role(StrEnum):
    """The role of a Record that is not a printed candidate series, which has its letter as its role."""

    INPUT = "input"
    SECOND_INPUT = "second_input"
    SEQUENCE = "sequence"
    """The series a true-or-false question prints, which has no letter."""


_TRUE_OR_FALSE_LABEL_SET = LabelSet(
    name="true_or_false",
    labels=(Label(name="T", description="True."), Label(name="F", description="False.")),
)
"""The label set that every true-or-false task shares."""
_CONTEXT_KEYS = ("meta_info", "domain", "dataset")
"""The published columns an input Record is annotated with, and its task gives the model as context."""
_PUBLISHED_LISTS = {key: f"published_{key}" for key in _CONTEXT_KEYS}
"""The annotation key of each context column for a row that publishes the column as a list."""
_READ_COLUMNS = ("task", "question_type", "input_ts", "question", "answer", *_CONTEXT_KEYS)
"""The columns a conversion reads. The release's ninth, ``raw_ts``, holds the series before normalization."""
_SERIES_COLUMNS = ["input_ts", "question"]
_DIGEST_COLUMNS = ("task", "input_ts", "question")
"""The columns that make a row's question: the same question about the same series, whatever its
context or published ``question_type`` says. The model sees neither."""

_COMPARISON_SERIES = 2
"""How many series a comparison row has."""
_MIN_CHOICES = 2
_LETTERS = ("A", "B", "C", "D")
"""A puzzle's patch letters, and the most letters a multiple-choice question offers."""
_TRUE_OR_FALSE_LETTERS = _TRUE_OR_FALSE_LABEL_SET.names
"""The letters of a true-or-false question: T stands for 'True.' and F for 'False.'"""
_INPUT_ROLES = (_Role.INPUT, _Role.SECOND_INPUT)
"""The roles of the input series, in order."""
_PRINTING_TASKS = {_Task.DATA_TRANSFORMATION, _Task.TEMPORAL_RELATIONSHIP}
"""The tasks whose questions print candidate series of their own."""


@unique
class _QuestionFormat(StrEnum):
    """The format a question has, read from its text: the paper's TF, MC or PZ."""

    TRUE_OR_FALSE = "true_or_false"
    MULTIPLE_CHOICE = "multiple_choice"
    PUZZLING = "puzzling"


_QUESTION_TYPES = {
    _QuestionFormat.TRUE_OR_FALSE: "true_or_false",
    _QuestionFormat.MULTIPLE_CHOICE: "multiple_choices",
    _QuestionFormat.PUZZLING: "ordering",
}
"""The ``question_type`` the release gives each question format when it names it right."""
_PRINTED = re.compile(r"^(?:([A-D]): )?(\[[^\n]*\])$", re.MULTILINE)
"""A series printed on a line of its own, behind a choice or patch letter or alone after a stem."""
_TRUE_OR_FALSE_LINES = "\nT: True.\nF: False.\n"


class _Printed(NamedTuple):
    """A series a question prints."""

    label: str | None
    """The choice or patch letter, ``None`` for a true-or-false question's sequence."""
    values: list[float]


class _QuestionReading(NamedTuple):
    """What a question's text says besides its series."""

    question_format: _QuestionFormat
    choices: dict[str, str]
    """The choice texts of a multiple-choice question whose choices are text, else empty."""
    ambiguous: bool
    """Whether another answer is as right as the published one."""


class _Contexts(NamedTuple):
    """The context a row publishes, as the annotations of its first input Record."""

    on_input: Mapping[str, str | tuple[str, ...]]
    """Each published text under its column name, or each published list under its ``published_*`` key."""


class _RowKey(NamedTuple):
    """A row's place in the release."""

    split: _SplitFile
    index: int
    """The row's position in its file, counted from 0."""


_TRANSFORMS = {
    "Fourier Transform": ("tsaqa_fourier_transform", "Fourier transform"),
    "Wavelet Transform": ("tsaqa_wavelet_transform", "Wavelet transform"),
    "First Order Difference": ("tsaqa_first_order_difference", "First-order difference"),
}
_TRANSFORM = re.compile(rf"the ({'|'.join(_TRANSFORMS)}) of the given time series")
_AXIS = OrdinalAxis()
_SERIES_SPEC = TimeSeriesSpec(spec_type="tsaqa_series", name="TSAQA series", unit_value=None, dtype="float64")
_TRANSFORM_SPECS = {
    transform: TimeSeriesSpec(spec_type=spec_type, name=f"{label} of a TSAQA series", unit_value=None, dtype="float64")
    for transform, (spec_type, label) in _TRANSFORMS.items()
}


@unique
class SourceIssue(StrEnum):
    """A problem in a published row, which the connector keeps as published and names in the task's metadata."""

    QUESTION_TYPE_MISMATCH = "question_type_mismatch"
    """The published ``question_type`` names another format than the question has."""
    REPEATED_QUESTION = "repeated_question"
    """Another row asks the same question about the same series, and expects the same answer."""
    REPEATED_ACROSS_SPLITS = "repeated_across_splits"
    """A copy of the row's question and series is in another split."""
    CONFLICTING_ANSWER = "conflicting_answer"
    """Another row asks the same question about the same series, and expects another answer."""
    AMBIGUOUS_ANSWER = "ambiguous_answer"
    """Another answer is as right as the published one: two patches, or a choice and the right one, are equal."""
    WRONG_TRANSFORM_TARGET = "wrong_transform_target"
    """The input of a data transformation question is all zeros, which every transform maps to zeros,
    and the published answer says otherwise."""


_TaskT = TypeVar("_TaskT", bound=Task)
_SOURCE_ISSUES = "source_issues"
"""The task metadata key that lists the row's source issues."""


def without_source_issues(
    tasks: Iterable[_TaskT], issues: Collection[SourceIssue] = tuple(SourceIssue)
) -> Iterator[_TaskT]:
    """Leave out the tasks whose row has any of ``issues``.

    ``issues`` is checked at the call. Each task is checked when the result reaches it: reading a task
    that has no ``source_issues`` list of issue names raises ``TimeFValidationError``.

    Args:
        tasks: The tasks to filter.
        issues: The issues that exclude a task. All of them by default.

    Returns:
        An iterator over each task whose row has none of ``issues``, in order.

    Raises:
        TimeFValidationError: If ``issues`` isn't a collection of issue names.
    """
    if isinstance(issues, str):
        raise TimeFValidationError(f"issues is a collection of issue names, got the text {issues!r}")
    if not isinstance(issues, Iterable):
        raise TimeFValidationError(f"issues is a collection of issue names, got {issues!r}")
    excluded: set[SourceIssue] = set()
    for issue in issues:
        # Check the type first: an entry such as a list can't be looked up in a set.
        if not isinstance(issue, str) or issue not in set(SourceIssue):
            raise TimeFValidationError(
                f"unknown source issue {issue!r}, expected one of {[issue.value for issue in SourceIssue]}"
            )
        excluded.add(SourceIssue(issue))
    return (task for task in tasks if excluded.isdisjoint(_task_issues(task)))


def _task_issues(task: Task) -> set[SourceIssue]:
    if _SOURCE_ISSUES not in task.metadata:
        raise TimeFValidationError(f"task metadata has no {_SOURCE_ISSUES!r} list, so its source issues are unknown")
    found = task.metadata[_SOURCE_ISSUES]
    message = f"task metadata {_SOURCE_ISSUES!r} must list issue names, got {found!r}"
    if not isinstance(found, list):
        raise TimeFValidationError(message)
    try:
        return {SourceIssue(name) for name in found}
    except ValueError as exc:
        raise TimeFValidationError(message) from exc


@dataclass(frozen=True)
class _Row:
    """A published row, validated, with its series' lengths and digest but not their values."""

    split: _SplitFile
    index: int
    """The row's position in its file, counted from 0."""
    task: _Task
    question_type: str
    """As published, which may disagree with :attr:`question_format`."""
    question_format: _QuestionFormat
    """As read from the question's text."""
    answer: str
    prompt: str
    """The question with a placeholder in place of each series it prints."""
    choices: Mapping[str, str]
    """A multiple-choice question's choice texts, empty when its choices are series."""
    printed_roles: tuple[str, ...]
    """The role of each printed series: its choice or patch letter, or ``sequence`` for a true-or-false one."""
    lengths: tuple[int, ...]
    """The length of each series in Record order: the inputs, then the printed series."""
    values_digest: bytes
    """The digest of every series' values in Record order, to tell a reread of the same values from another."""
    n_inputs: int
    contexts: _Contexts
    transform: str | None
    question_digest: bytes
    """Equal for rows that ask the same question about the same series, whatever their context says."""
    has_ambiguous_answer: bool
    has_wrong_target: bool

    @property
    def file(self) -> str:
        return self.split.filename

    @property
    def key(self) -> _RowKey:
        return _RowKey(self.split, self.index)


def _values_digest(series: Iterable[Sequence[float]]) -> bytes:
    """Digest the values of series in order, as float64 bytes.

    Returns:
        A 16-byte digest that differs when any value, or the order of the series, differs.
    """
    digest = hashlib.blake2b(digest_size=16)
    for values in series:
        digest.update(array("d", values).tobytes())
    return digest.digest()


class _SeriesRow(SourceRow):
    """A decoded source series, before exact numerical conversion."""

    series: list[int | float] = Field(min_length=1)


def _series_values(value: Any) -> list[float]:
    parsed = validate_row({"series": value}, _SeriesRow, location="series")
    return numeric_signal(parsed.series, "series", spec=_SERIES_SPEC).to_pylist()


def _input_series(task: _Task, text: str) -> list[list[float]]:
    """Decode a row's ``input_ts``: one series, or a list of two for comparison.

    Returns:
        The input series, as floats.

    Raises:
        TimeNetConnectorError: If a comparison's ``input_ts`` doesn't hold two series, or a series isn't finite numbers.
    """
    value = json.loads(text)
    if task is not _Task.COMPARISON:
        return [_series_values(value)]
    if not isinstance(value, list) or len(value) != _COMPARISON_SERIES:
        raise TimeNetConnectorError(f"a comparison input_ts holds {_COMPARISON_SERIES} series")
    return [_series_values(series) for series in value]


def _printed_series(task: _Task, question: str) -> list[_Printed]:
    """Find the series a question prints, with each one's label.

    Returns:
        Each printed series' label and values, in text order.
    """
    if task not in _PRINTING_TASKS:
        return []
    return [_Printed(match[1], _series_values(json.loads(match[2]))) for match in _PRINTED.finditer(question)]


def _input_contexts(task: _Task, text: Mapping[str, str]) -> _Contexts:
    """Read the context a row publishes.

    A comparison, data transformation or temporal relationship row publishes each context column as a
    list. No source says what each entry describes, so the connector keeps the lists whole, under their
    ``published_*`` keys. Any other row publishes one text for each column, under its own name.

    Returns:
        The annotations of the row's first input Record.

    Raises:
        TimeNetConnectorError: If a published list isn't a JSON list of texts, a comparison's doesn't hold two, or
            the three lists differ in length.
    """
    if task is not _Task.COMPARISON and task not in _PRINTING_TASKS:
        return _Contexts({key: text[key] for key in _CONTEXT_KEYS})
    expected = f"{_COMPARISON_SERIES} texts" if task is _Task.COMPARISON else "texts"
    lists: dict[str, tuple[str, ...]] = {}
    for key in _CONTEXT_KEYS:
        entries = json.loads(text[key])
        if not isinstance(entries, list) or not entries or not all(isinstance(entry, str) for entry in entries):
            raise TimeNetConnectorError(f"a {task} row's {key} is a JSON list of {expected}")
        if task is _Task.COMPARISON and len(entries) != _COMPARISON_SERIES:
            raise TimeNetConnectorError(f"a {task} row's {key} is a JSON list of {expected}")
        lists[key] = tuple(entries)
    lengths = {key: len(entries) for key, entries in lists.items()}
    if len(set(lengths.values())) != 1:
        raise TimeNetConnectorError(f"a {task} row's context lists differ in length: {lengths}")
    return _Contexts({_PUBLISHED_LISTS[key]: entries for key, entries in lists.items()})


def _question_letters(question: str, lines: list[tuple[str, str]]) -> tuple[_QuestionFormat, tuple[str, ...]]:
    """Read a question's format and the letters it offers.

    Returns:
        The format and its letters, in order.

    Raises:
        TimeNetConnectorError: If a multiple-choice question doesn't offer A to D in order.
    """
    if "\n\nPatches:\n" in question:
        return _QuestionFormat.PUZZLING, _LETTERS
    if _TRUE_OR_FALSE_LINES in question:
        return _QuestionFormat.TRUE_OR_FALSE, _TRUE_OR_FALSE_LETTERS
    letters = tuple(letter for letter, _ in lines)
    if not _MIN_CHOICES <= len(letters) <= len(_LETTERS) or letters != _LETTERS[: len(letters)]:
        raise TimeNetConnectorError(f"a multiple-choice question offers A to D in order, got {letters}")
    return _QuestionFormat.MULTIPLE_CHOICE, letters


def _check_answer(question_format: _QuestionFormat, answer: str, letters: tuple[str, ...]) -> None:
    """Check that the answer is one of the letters, or for a puzzle an order of all of them.

    Raises:
        TimeNetConnectorError: If the answer doesn't fit.
    """
    if question_format is _QuestionFormat.PUZZLING:
        fits = sorted(answer.split(",")) == sorted(letters)
    else:
        fits = answer in letters
    if not fits:
        raise TimeNetConnectorError(f"the answer {answer!r} does not fit the question's letters {letters}")


def _has_another_right_answer(
    question_format: _QuestionFormat, answer: str, candidates: Mapping[str, tuple[float, ...] | str]
) -> bool:
    """Tell whether another answer is as right as the published one, from the question's candidates.

    Returns:
        ``True`` if two patches, or the answer's choice and another, are equal.
    """
    if question_format is _QuestionFormat.PUZZLING:
        return len(set(candidates.values())) < len(candidates)
    if question_format is _QuestionFormat.MULTIPLE_CHOICE:
        return list(candidates.values()).count(candidates[answer]) > 1
    return False


def _read_question(task: _Task, question: str, answer: str, printed: list[_Printed]) -> _QuestionReading:
    """Read a question's format, its text choices, and whether another answer is as right as the published one.

    A puzzle has another right order when two patches are equal. A multiple-choice answer has another
    right letter when another choice prints the same series, or reads the same text. A true-or-false
    answer is never ambiguous.

    Returns:
        The format, the choice texts of a multiple-choice question whose choices are text, and the ambiguity.

    Raises:
        TimeNetConnectorError: If a multiple-choice question doesn't offer A to D in order, the answer is not one of the
            question's letters or an order of all four patches, or the printed series don't match the letters.
    """
    lines = re.findall(r"^([A-Z]): (.*)$", question.partition("\n\nChoices:\n")[2], re.MULTILINE)
    question_format, letters = _question_letters(question, lines)
    _check_answer(question_format, answer, letters)
    expected_labels = [None] if question_format is _QuestionFormat.TRUE_OR_FALSE else list(letters)
    labels = [item.label for item in printed]
    if task in _PRINTING_TASKS and labels != expected_labels:
        raise TimeNetConnectorError(f"a {task} question prints series labelled {expected_labels}, got {labels}")
    choices: dict[str, str] = {}
    if question_format is _QuestionFormat.MULTIPLE_CHOICE and task not in _PRINTING_TASKS:
        choices = dict(lines)
    candidates: dict[str, tuple[float, ...] | str] = dict(choices)
    candidates.update({item.label: tuple(item.values) for item in printed if item.label is not None})
    return _QuestionReading(question_format, choices, _has_another_right_answer(question_format, answer, candidates))


def _contradicts_zero_input(
    task: _Task, question_format: _QuestionFormat, answer: str, inputs: list[list[float]], printed: list[_Printed]
) -> bool:
    """Tell whether a data transformation answer contradicts an all-zero input.

    Fourier, wavelet and first-order difference transforms all turn zeros into zeros. So the right
    candidate of such an input is the one of zeros.

    Returns:
        ``True`` if the input is all zeros and the answer rejects the zero candidate or picks another.
    """
    if task is not _Task.DATA_TRANSFORMATION or any(inputs[0]):
        return False
    if question_format is _QuestionFormat.TRUE_OR_FALSE:
        return (answer == "T") != (not any(printed[0].values))
    chosen = next(item for item in printed if item.label == answer)
    return any(chosen.values)


def _prompt(question: str, first_series: int, printed: list[_Printed]) -> str:
    """Return the question with a placeholder, such as ``<series 2>``, in place of each series it prints.

    Args:
        question: The published question.
        first_series: The number of the first printed series, counted from 1 after the input series.
        printed: The series the question prints. A task that prints none keeps its question as published,
            so a text choice such as ``[13, 17]`` is not taken for a series.

    Returns:
        The prompt.
    """
    if not printed:
        return question
    numbers = itertools.count(first_series)

    def placeholder(match: re.Match[str]) -> str:
        label = f"{match[1]}: " if match[1] else ""
        return f"{label}<series {next(numbers)}>"

    return _PRINTED.sub(placeholder, question)


def _validate_row(values: Mapping[str, Any], split: _SplitFile, index: int) -> _Row:
    """Validate a published row against what a conversion needs of it, and keep its small fields.

    Returns:
        The row.

    Raises:
        TimeNetConnectorError: If the row is malformed, with its file and row.
    """
    try:
        text: dict[str, str] = {}
        for column in _READ_COLUMNS:
            if not isinstance(values[column], str) or not values[column].strip():
                raise TimeNetConnectorError(f"{column} is nonblank text")
            text[column] = values[column]
        question = text["question"]
        if text["task"] not in set(_Task):
            raise TimeNetConnectorError(f"unknown task {text['task']!r}")
        task = _Task(text["task"])
        if text["question_type"] not in _QUESTION_TYPES.values():
            raise TimeNetConnectorError(f"unknown question_type {text['question_type']!r}")
        inputs, printed = _input_series(task, text["input_ts"]), _printed_series(task, question)
        all_series = [*inputs, *(item.values for item in printed)]
        reading = _read_question(task, question, text["answer"], printed)
        return _Row(
            split=split,
            index=index,
            task=task,
            question_type=text["question_type"],
            question_format=reading.question_format,
            answer=text["answer"],
            prompt=_prompt(question, len(inputs) + 1, printed),
            choices=reading.choices,
            printed_roles=tuple(_Role.SEQUENCE if item.label is None else item.label for item in printed),
            lengths=tuple(len(values) for values in all_series),
            values_digest=_values_digest(all_series),
            n_inputs=len(inputs),
            contexts=_input_contexts(task, text),
            transform=_transform(task, question),
            question_digest=hashlib.blake2b(
                "\0".join(text[c] for c in _DIGEST_COLUMNS).encode(), digest_size=16
            ).digest(),
            has_ambiguous_answer=reading.ambiguous,
            has_wrong_target=_contradicts_zero_input(task, reading.question_format, text["answer"], inputs, printed),
        )
    except (TimeNetConnectorError, ValueError, KeyError, IndexError, RecursionError) as exc:
        raise TimeNetConnectorError(f"{split.filename} row {index}: {exc}") from exc


def _transform(task: _Task, question: str) -> str | None:
    """Read the transform a data transformation question names.

    Returns:
        The transform, ``None`` for any other task.

    Raises:
        TimeNetConnectorError: If a data transformation question names no known transform.
    """
    if task is not _Task.DATA_TRANSFORMATION:
        return None
    match = _TRANSFORM.search(question)
    if match is None:
        raise TimeNetConnectorError("a data transformation question names a known transform")
    return match[1]


def _read_rows(path: Path, split: _SplitFile) -> Iterator[_Row]:
    """Validate every row of one split file, in file order.

    Yields:
        Each row.

    Raises:
        TimeNetConnectorError: If the file can't be read as parquet.
    """
    try:
        for index, values in enumerate(iter_parquet_rows(path, columns=_READ_COLUMNS)):
            yield _validate_row(values, split, index)
    except (OSError, pa.ArrowException) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc


def _find_issues(rows: list[_Row]) -> dict[_RowKey, frozenset[SourceIssue]]:
    """Find each row's source issues, which depend on the copies of its question in every split.

    Returns:
        The issues of each row that has any.
    """
    copies: defaultdict[bytes, list[_Row]] = defaultdict(list)
    for row in rows:
        copies[row.question_digest].append(row)
    issues: dict[_RowKey, frozenset[SourceIssue]] = {}
    for row in rows:
        found: set[SourceIssue] = set()
        if row.question_type != _QUESTION_TYPES[row.question_format]:
            found.add(SourceIssue.QUESTION_TYPE_MISMATCH)
        if row.has_ambiguous_answer:
            found.add(SourceIssue.AMBIGUOUS_ANSWER)
        if row.has_wrong_target:
            found.add(SourceIssue.WRONG_TRANSFORM_TARGET)
        for other in copies[row.question_digest]:
            if other is row:
                continue
            found.add(SourceIssue.REPEATED_QUESTION if other.answer == row.answer else SourceIssue.CONFLICTING_ANSWER)
            if other.split != row.split:
                found.add(SourceIssue.REPEATED_ACROSS_SPLITS)
        if found:
            issues[row.key] = frozenset(found)
    return issues


class _SeriesRead(NamedTuple):
    """A row's location, parsing mode, and expected values for a lazy read."""

    split: _SplitFile
    index: int
    task: _Task
    lengths: tuple[int, ...]
    digest: bytes


class _SeriesLoader:
    """Load one series of a row lazily, holding one split file's series text and parsing each row once.

    Reading the Records in order reads each split file once. The writer sorts the series by spec type,
    so it returns to each file once per spec type, and the loader reads a file again then. A series
    loads only if its row's lengths and values digest equal those kept at conversion.
    """

    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths = paths
        self._columns = lru_cache(maxsize=1)(self._read_columns)
        # A row's Records load one after another, so one parsed row suffices.
        self._shared_channels = SharedChannelLoader(self._read_series)

    def _read_columns(self, split: _SplitFile) -> tuple[pa.StringArray, pa.StringArray]:
        try:
            table = pq.read_table(self._paths[split], columns=_SERIES_COLUMNS).combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self._paths[split]}: {exc}") from exc
        input_ts, question = (table.column(name).chunk(0) for name in _SERIES_COLUMNS)
        return input_ts, question

    def _read_series(self, request: _SeriesRead) -> tuple[pa.Array, ...]:
        input_ts, question = self._columns(request.split)
        context = f"{request.split.filename} row {request.index}"
        try:
            printed = _printed_series(request.task, str(question[request.index].as_py()))
            series = [
                *_input_series(request.task, str(input_ts[request.index].as_py())),
                *(item.values for item in printed),
            ]
        except (TimeNetConnectorError, ValueError, IndexError) as exc:
            raise TimeNetConnectorError(f"{context}: {exc}; the file changed since conversion") from exc
        if tuple(len(values) for values in series) != request.lengths or _values_digest(series) != request.digest:
            raise TimeNetConnectorError(f"{context}: the series differ from the converted row; the file changed")
        return tuple(pa.array(values, type=pa.float64()) for values in series)

    def signals(self, row: _Row) -> tuple[Signal, ...]:
        """Return the row's signals in source order without reading values."""
        request = _SeriesRead(row.split, row.index, row.task, row.lengths, row.values_digest)
        return Signal.from_shared_loader(
            loader=self._shared_channels,
            recording_key=request,
            channels=tuple(
                SignalChannel(
                    name="series",
                    spec=_SERIES_SPEC
                    if position < row.n_inputs or row.transform is None
                    else _TRANSFORM_SPECS[row.transform],
                    time_axis=_AXIS,
                    n_values=length,
                )
                for position, length in enumerate(row.lengths)
            ),
        )


def _records(row: _Row, loader: _SeriesLoader) -> tuple[Record, ...]:
    """Build a row's Records in the order its task gives them to a model, without reading any values.

    Returns:
        The row's Records.
    """
    roles = (*_INPUT_ROLES[: row.n_inputs], *row.printed_roles)
    records = []
    for position, (role, signal) in enumerate(zip(roles, loader.signals(row), strict=True)):
        metadata: dict[str, JsonValue] = {"file": row.file, "row": row.index, "role": str(role)}
        record = Record(sources=(Source(name="series", signals=(signal,)),), metadata=metadata)
        if position == 0:
            record.add_annotations(
                {key: value if isinstance(value, str) else list(value) for key, value in row.contexts.on_input.items()}
            )
        records.append(record)
    return tuple(records)


def _published_labels(row: _Row) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return the published ``task`` and ``question_type`` of a row, as annotation keys and values."""
    return ("task", row.task.value), ("question_type", row.question_type)


def _task(row: _Row, inputs: tuple[Record, ...], issues: frozenset[SourceIssue], dataset: TimeFDataset) -> Task:
    """Build a row's task: a true-or-false row is a classification task, any other an answer task.

    The model reads what Appendix E.1 lists: the published context, then the published ``task`` and
    ``question_type``, then the question. The context is on the first input Record: published texts
    under their column names, or published lists under their ``published_*`` keys. The question format
    read from the question stays in the metadata, for routing.

    Args:
        row: The row.
        inputs: The row's input Records.
        issues: The row's source issues.
        dataset: The dataset that shares each published task and question-type annotation.

    Returns:
        The task, its target the published answer.
    """
    annotations = itertools.chain.from_iterable(record.annotations for record in inputs)
    input_annotations = (
        *(a for a in annotations if a.key in _CONTEXT_KEYS or a.key in _PUBLISHED_LISTS.values()),
        *(dataset.annotate_shared(Annotation(key=key, value=value)) for key, value in _published_labels(row)),
    )
    metadata: dict[str, JsonValue] = {
        "task": row.task.value,
        "question_type": row.question_type,
        "question_format": row.question_format.value,
        "file": row.file,
        "row": row.index,
    }
    if row.transform is not None:
        metadata["transform"] = row.transform
    metadata[_SOURCE_ISSUES] = [issue.value for issue in sorted(issues)]
    if row.question_format is _QuestionFormat.TRUE_OR_FALSE:
        return ClassificationTask(
            inputs=inputs,
            prompt=row.prompt,
            targets=(row.answer,),
            split=_SPLITS[row.split],
            input_annotations=input_annotations,
            label_set=_TRUE_OR_FALSE_LABEL_SET,
            metadata=metadata,
        )
    if row.choices:
        metadata["answer_text"] = row.choices[row.answer]
    return AnswerTask(
        inputs=inputs,
        prompt=row.prompt,
        targets=(row.answer,),
        split=_SPLITS[row.split],
        input_annotations=input_annotations,
        metadata=metadata,
    )


class TSAQAConnector(BaseHuggingFaceConnector):
    """Build TSAQA from the pinned train, val and test parquet files of its Hugging Face release."""

    HF_REPO = "TSAQA/TSAQA-Benchmark"
    SOURCE_REVISION = "883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {split.value: split.filename for split in _SplitFile}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert each split file into lazy Records and a stream of one task per row.

        Every file is read once up front to check each row and keep its small fields. The source issues
        need all three splits, because a copy of a question may be in another one. The series values stay
        in the files until written, so the files must stay available until then.

        Args:
            raw_refs: The downloaded split files.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.

        Raises:
            TimeNetConnectorError: If a split file is missing, or a row is malformed.
        """
        paths = {ref.part: ref.path for ref in raw_refs}
        if paths.keys() != _SPLITS.keys():
            raise TimeNetConnectorError(
                f"TSAQA needs all its split files, missing {sorted(split.value for split in _SplitFile if split not in paths)}"
            )
        rows: list[_Row] = []
        for split in _SplitFile:
            rows.extend(_read_rows(paths[split], split))
        issues = _find_issues(rows)
        loader = _SeriesLoader(paths)
        records = [_records(row, loader) for row in rows]

        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (("source_repository", self.HF_REPO), ("source_revision", self.SOURCE_REVISION)):
            dataset.annotate(Annotation(key=key, value=value))
        for row_records in records:
            for record in row_records:
                dataset.add_record(record=record)

        def tasks() -> Iterator[Task]:
            for row, inputs in zip(rows, records, strict=True):
                yield _task(row, inputs, issues.get(row.key, frozenset()), dataset)

        dataset.set_task_stream((ClassificationTask, AnswerTask), tasks)
        return dataset


CONNECTOR = TSAQAConnector
