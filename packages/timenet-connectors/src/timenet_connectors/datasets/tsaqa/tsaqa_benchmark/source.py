"""Read and check the TSAQA parquet rows, find their source issues, and reread their series on demand.

All code that reads the source lives here. A checked row keeps its small fields and the lengths and
digest of its series, not their values. The series loader rereads a row's series from its file when
the writer loads them, and checks that they still match the row.
"""

from array import array
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, field_validator

from timenet.dataset import OrdinalAxis, Signal, SignalChannel
from timenet.errors import TimeNetConnectorError
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import (
    CONTEXT_KEYS,
    LETTERS,
    PRINTING_TASKS,
    PUBLISHED_LISTS,
    QUESTION_TYPES,
    SERIES_SPEC,
    SPLITS,
    TRANSFORM_SPECS,
    TRANSFORMS,
    TRUE_OR_FALSE_LETTERS,
    AnalysisTask,
    QuestionFormat,
    Role,
    SourceIssue,
    SplitFile,
)
from timenet_connectors.sources.parquet import iter_parquet_rows
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.sources.rows import NonBlankText, PublishedStrEnum, SourceRow, validate_row
from timenet_connectors.sources.signals import numeric_signal


class _PublishedRow(SourceRow):
    """Fields used from a published TSAQA row; raw_ts is not read."""

    task: PublishedStrEnum[AnalysisTask]
    question_type: NonBlankText
    input_ts: NonBlankText
    question: NonBlankText
    answer: NonBlankText
    meta_info: NonBlankText
    domain: NonBlankText
    dataset: NonBlankText

    @field_validator("question_type")
    @classmethod
    def _known_question_type(cls, value: str) -> str:
        if value not in QUESTION_TYPES.values():
            raise ValueError(f"unknown question_type {value!r}")
        return value


_READ_COLUMNS = tuple(_PublishedRow.model_fields)

_SERIES_COLUMNS = ["input_ts", "question"]

_DIGEST_COLUMNS = ("task", "input_ts", "question")
"""The columns that make a row's question: the same question about the same series, whatever its
context or published ``question_type`` says. The model sees neither."""

_COMPARISON_SERIES = 2
"""How many series a comparison row has."""

_MIN_CHOICES = 2

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

    question_format: QuestionFormat
    choices: dict[str, str]
    """The choice texts of a multiple-choice question whose choices are text, else empty."""
    ambiguous: bool
    """Whether another answer is as right as the published one."""


class _Contexts(NamedTuple):
    """The context a row publishes, as the annotations of its first input Record."""

    on_input: Mapping[str, str | tuple[str, ...]]
    """Each published text under its column name, or each published list under its ``published_*`` key."""


class RowKey(NamedTuple):
    """A row's place in the release."""

    split: SplitFile
    index: int
    """The row's position in its file, counted from 0."""


_TRANSFORM = re.compile(rf"the ({'|'.join(TRANSFORMS)}) of the given time series")


@dataclass(frozen=True)
class Row:
    """A published row, validated, with its series' lengths and digest but not their values."""

    split: SplitFile
    index: int
    """The row's position in its file, counted from 0."""
    task: AnalysisTask
    question_type: str
    """As published, which may disagree with :attr:`question_format`."""
    question_format: QuestionFormat
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
        """The source parquet filename."""
        return self.split.filename

    @property
    def key(self) -> RowKey:
        """The stable source row identity."""
        return RowKey(self.split, self.index)


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
    return numeric_signal(parsed.series, "series", spec=SERIES_SPEC).to_pylist()


def _is_finite(item: Any) -> bool:
    return isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)


def _input_series(task: AnalysisTask, text: str) -> list[list[float]]:
    """Decode a row's ``input_ts``: one series, or a list of two for comparison.

    Returns:
        The input series, as floats.

    Raises:
        TimeNetConnectorError: If a comparison's ``input_ts`` doesn't hold two series, or a series isn't finite numbers.
    """
    value = json.loads(text)
    if task is not AnalysisTask.COMPARISON:
        return [_series_values(value)]
    if not isinstance(value, list) or len(value) != _COMPARISON_SERIES:
        raise TimeNetConnectorError(f"a comparison input_ts holds {_COMPARISON_SERIES} series")
    return [_series_values(series) for series in value]


def _printed_series(task: AnalysisTask, question: str) -> list[_Printed]:
    """Find the series a question prints, with each one's label.

    Returns:
        Each printed series' label and values, in text order.
    """
    if task not in PRINTING_TASKS:
        return []
    return [_Printed(match[1], _series_values(json.loads(match[2]))) for match in _PRINTED.finditer(question)]


def _input_contexts(task: AnalysisTask, text: Mapping[str, str]) -> _Contexts:
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
    if task is not AnalysisTask.COMPARISON and task not in PRINTING_TASKS:
        return _Contexts({key: text[key] for key in CONTEXT_KEYS})
    expected = f"{_COMPARISON_SERIES} texts" if task is AnalysisTask.COMPARISON else "texts"
    lists: dict[str, tuple[str, ...]] = {}
    for key in CONTEXT_KEYS:
        entries = json.loads(text[key])
        if not isinstance(entries, list) or not entries or not all(isinstance(entry, str) for entry in entries):
            raise TimeNetConnectorError(f"a {task} row's {key} is a JSON list of {expected}")
        if task is AnalysisTask.COMPARISON and len(entries) != _COMPARISON_SERIES:
            raise TimeNetConnectorError(f"a {task} row's {key} is a JSON list of {expected}")
        lists[key] = tuple(entries)
    lengths = {key: len(entries) for key, entries in lists.items()}
    if len(set(lengths.values())) != 1:
        raise TimeNetConnectorError(f"a {task} row's context lists differ in length: {lengths}")
    return _Contexts({PUBLISHED_LISTS[key]: entries for key, entries in lists.items()})


def _question_letters(question: str, lines: list[tuple[str, str]]) -> tuple[QuestionFormat, tuple[str, ...]]:
    """Read a question's format and the letters it offers.

    Returns:
        The format and its letters, in order.

    Raises:
        TimeNetConnectorError: If a multiple-choice question doesn't offer A to D in order.
    """
    if "\n\nPatches:\n" in question:
        return QuestionFormat.PUZZLING, LETTERS
    if _TRUE_OR_FALSE_LINES in question:
        return QuestionFormat.TRUE_OR_FALSE, TRUE_OR_FALSE_LETTERS
    letters = tuple(letter for letter, _ in lines)
    if not _MIN_CHOICES <= len(letters) <= len(LETTERS) or letters != LETTERS[: len(letters)]:
        raise TimeNetConnectorError(f"a multiple-choice question offers A to D in order, got {letters}")
    return QuestionFormat.MULTIPLE_CHOICE, letters


def _check_answer(question_format: QuestionFormat, answer: str, letters: tuple[str, ...]) -> None:
    """Check that the answer is one of the letters, or for a puzzle an order of all of them.

    Raises:
        TimeNetConnectorError: If the answer doesn't fit.
    """
    if question_format is QuestionFormat.PUZZLING:
        fits = sorted(answer.split(",")) == sorted(letters)
    else:
        fits = answer in letters
    if not fits:
        raise TimeNetConnectorError(f"the answer {answer!r} does not fit the question's letters {letters}")


def _has_another_right_answer(
    question_format: QuestionFormat, answer: str, candidates: Mapping[str, tuple[float, ...] | str]
) -> bool:
    """Tell whether another answer is as right as the published one, from the question's candidates.

    Returns:
        ``True`` if two patches, or the answer's choice and another, are equal.
    """
    if question_format is QuestionFormat.PUZZLING:
        return len(set(candidates.values())) < len(candidates)
    if question_format is QuestionFormat.MULTIPLE_CHOICE:
        return list(candidates.values()).count(candidates[answer]) > 1
    return False


def _read_question(task: AnalysisTask, question: str, answer: str, printed: list[_Printed]) -> _QuestionReading:
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
    expected_labels = [None] if question_format is QuestionFormat.TRUE_OR_FALSE else list(letters)
    labels = [item.label for item in printed]
    if task in PRINTING_TASKS and labels != expected_labels:
        raise TimeNetConnectorError(f"a {task} question prints series labelled {expected_labels}, got {labels}")
    choices: dict[str, str] = {}
    if question_format is QuestionFormat.MULTIPLE_CHOICE and task not in PRINTING_TASKS:
        choices = dict(lines)
    candidates: dict[str, tuple[float, ...] | str] = dict(choices)
    candidates.update({item.label: tuple(item.values) for item in printed if item.label is not None})
    return _QuestionReading(question_format, choices, _has_another_right_answer(question_format, answer, candidates))


def _contradicts_zero_input(
    task: AnalysisTask, question_format: QuestionFormat, answer: str, inputs: list[list[float]], printed: list[_Printed]
) -> bool:
    """Tell whether a data transformation answer contradicts an all-zero input.

    Fourier, wavelet and first-order difference transforms all turn zeros into zeros. So the right
    candidate of such an input is the one of zeros.

    Returns:
        ``True`` if the input is all zeros and the answer rejects the zero candidate or picks another.
    """
    if task is not AnalysisTask.DATA_TRANSFORMATION or any(inputs[0]):
        return False
    if question_format is QuestionFormat.TRUE_OR_FALSE:
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


def _validate_row(values: Mapping[str, Any], split: SplitFile, index: int) -> Row:
    """Validate a published row against what a conversion needs of it, and keep its small fields.

    Returns:
        The row.

    Raises:
        TimeNetConnectorError: If the row is malformed, with its file and row.
    """
    row = validate_row(values, _PublishedRow, location=f"{split.filename} row {index}")
    try:
        text = {column: getattr(row, column) for column in _READ_COLUMNS}
        question = row.question
        task = row.task
        inputs, printed = _input_series(task, text["input_ts"]), _printed_series(task, question)
        all_series = [*inputs, *(item.values for item in printed)]
        reading = _read_question(task, question, text["answer"], printed)
        return Row(
            split=split,
            index=index,
            task=task,
            question_type=text["question_type"],
            question_format=reading.question_format,
            answer=text["answer"],
            prompt=_prompt(question, len(inputs) + 1, printed),
            choices=reading.choices,
            printed_roles=tuple(Role.SEQUENCE if item.label is None else item.label for item in printed),
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


def _transform(task: AnalysisTask, question: str) -> str | None:
    """Read the transform a data transformation question names.

    Returns:
        The transform, ``None`` for any other task.

    Raises:
        TimeNetConnectorError: If a data transformation question names no known transform.
    """
    if task is not AnalysisTask.DATA_TRANSFORMATION:
        return None
    match = _TRANSFORM.search(question)
    if match is None:
        raise TimeNetConnectorError("a data transformation question names a known transform")
    return match[1]


def _read_rows(path: Path, split: SplitFile) -> Iterator[Row]:
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


def read_rows(paths: Mapping[str, Path]) -> list[Row]:
    """Check that every split file is there, and read and check all their rows in one full scan.

    The source issues need all three splits, because a copy of a question may be in another one.

    Args:
        paths: The downloaded file of each split, by split name.

    Returns:
        Every row, train first, then validation, then test, each in file order.

    Raises:
        TimeNetConnectorError: If a split file is missing, or a row is malformed.
    """
    if paths.keys() != SPLITS.keys():
        raise TimeNetConnectorError(
            f"TSAQA needs all its split files, missing {sorted(split.value for split in SplitFile if split not in paths)}"
        )
    rows: list[Row] = []
    for split in SplitFile:
        rows.extend(_read_rows(paths[split], split))
    return rows


def find_issues(rows: list[Row]) -> dict[RowKey, frozenset[SourceIssue]]:
    """Find each row's source issues, which depend on the copies of its question in every split.

    Returns:
        The issues of each row that has any.
    """
    copies: defaultdict[bytes, list[Row]] = defaultdict(list)
    for row in rows:
        copies[row.question_digest].append(row)
    issues: dict[RowKey, frozenset[SourceIssue]] = {}
    for row in rows:
        found: set[SourceIssue] = set()
        if row.question_type != QUESTION_TYPES[row.question_format]:
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

    split: SplitFile
    index: int
    task: AnalysisTask
    lengths: tuple[int, ...]
    digest: bytes


class SeriesLoader:
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

    def _read_columns(self, split: SplitFile) -> tuple[pa.StringArray, pa.StringArray]:
        try:
            table = pq.read_table(self._paths[split], columns=_SERIES_COLUMNS).combine_chunks()
        except (OSError, pa.ArrowException) as exc:
            raise TimeNetConnectorError(f"{self._paths[split]}: {exc}") from exc
        input_ts, question = (table.column(name).chunk(0) for name in _SERIES_COLUMNS)
        return input_ts, question

    def _parse_series(self, request: _SeriesRead) -> list[Sequence[float]]:
        input_ts, question = self._columns(request.split)
        context = f"{request.split.filename} row {request.index}"
        try:
            printed = _printed_series(request.task, str(question[request.index].as_py()))
            series: list[Sequence[float]] = [
                *_input_series(request.task, str(input_ts[request.index].as_py())),
                *(item.values for item in printed),
            ]
        except (TimeNetConnectorError, ValueError, IndexError) as exc:
            raise TimeNetConnectorError(f"{context}: {exc}") from exc
        return series

    def _read_series(self, request: _SeriesRead) -> tuple[pa.Array, ...]:
        series = checked_reread(
            partial(self._parse_series, request),
            fingerprint=lambda series: (tuple(map(len, series)), _values_digest(series)),
            expected=(request.lengths, request.digest),
            context=f"{request.split.filename} row {request.index}",
        )
        return tuple(pa.array(values, type=pa.float64()) for values in series)

    def signals(self, row: Row) -> tuple[Signal, ...]:
        """Return the row's signals in source order without reading values."""
        request = _SeriesRead(row.split, row.index, row.task, row.lengths, row.values_digest)
        return Signal.from_shared_loader(
            loader=self._shared_channels,
            recording_key=request,
            channels=tuple(
                SignalChannel(
                    name="series",
                    spec=SERIES_SPEC
                    if position < row.n_inputs or row.transform is None
                    else TRANSFORM_SPECS[row.transform],
                    time_axis=OrdinalAxis(),
                    n_values=length,
                )
                for position, length in enumerate(row.lengths)
            ),
        )
