"""Read the published TSAQA parquet files: one validated row per published row, and the reread of its series.

Each split file is one parquet row group of nine string columns. ``input_ts``, ``meta_info``,
``domain``, ``dataset`` and ``raw_ts`` hold JSON for some tasks. A row keeps its small fields and the
lengths of its series, not their values, which a Record's signal rereads. ``raw_ts`` is never read: it
holds each series before normalization, which the authors neither describe nor give a model (README).
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.questions import (
    Question,
    checked_series,
    decode_json,
    parse_question,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.release import SourceSplit
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import (
    QuestionFormat,
    QuestionType,
    SourceTask,
    context_entries,
)


PUBLISHED_COLUMNS = (
    "task",
    "question_type",
    "input_ts",
    "meta_info",
    "question",
    "answer",
    "domain",
    "dataset",
    "raw_ts",
)
"""The release's columns, in file order. Every one is a string column."""
_READ_COLUMNS = ("task", "question_type", "input_ts", "meta_info", "question", "answer", "domain", "dataset")
_QUESTION_COLUMNS = ("task", "question_type", "input_ts", "question")
_SERIES_COLUMNS = ["input_ts", "question"]
_BATCH_ROWS = 16384

E = TypeVar("E", bound=StrEnum)


@dataclass(frozen=True)
class RowRef:
    """Where one row lives: its split file and its position in it, counted from 0."""

    split: SourceSplit
    row: int

    @property
    def context(self) -> str:
        """Return where the row is, for error messages: its file and row."""
        return f"{self.split.source_file} row {self.row}"


@dataclass(frozen=True)
class Row:
    """A row's published fields, validated, with its series' lengths but not their values. Build one with :meth:`parse`."""

    ref: RowRef
    task: SourceTask
    question_type: QuestionType
    """As published, which may disagree with :attr:`question`'s format."""
    question: Question
    answer: str
    input_lengths: tuple[int, ...]
    """The length of each series of ``input_ts``: two for comparison, one otherwise."""
    meta_info: str
    """As published: plain text, or JSON list text for a task of several series. See :func:`context_list`."""
    domain: str
    """As published, like :attr:`meta_info`."""
    dataset: str
    """As published, like :attr:`meta_info`."""
    question_digest: bytes
    """A digest of the row's question about its series: task, question type, question and ``input_ts``.
    Rows with equal digests ask the same question about the same series, whatever their context says:
    some UCR series are published under two dataset names."""

    @property
    def format(self) -> QuestionFormat:
        """Return the format the question has, read from its text."""
        return self.question.format

    @classmethod
    def parse(cls, values: Mapping[str, Any], ref: RowRef) -> "Row":
        """Validate a published row.

        Args:
            values: The row's columns, by name.
            ref: Where the row was read from.

        Returns:
            The row's fields, and its series' lengths.

        Raises:
            TimeFValidationError: If the row is malformed, with its file and row.
        """
        try:
            text = {column: _text(values.get(column), column) for column in _READ_COLUMNS}
            task = _enum(SourceTask, text["task"], "task")
            question, _ = parse_question(text["question"], task)
            entries = context_entries(task, question.format)
            return cls(
                ref=ref,
                task=task,
                question_type=_enum(QuestionType, text["question_type"], "question_type"),
                question=question,
                answer=_answer(text["answer"], question),
                input_lengths=tuple(len(series) for series in parse_inputs(text["input_ts"], task)),
                meta_info=_context(text["meta_info"], entries, "meta_info"),
                domain=_context(text["domain"], entries, "domain"),
                dataset=_context(text["dataset"], entries, "dataset"),
                question_digest=hashlib.blake2b(
                    "\0".join(text[column] for column in _QUESTION_COLUMNS).encode(), digest_size=16
                ).digest(),
            )
        except (ValueError, RecursionError) as exc:
            raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _text(value: Any, column: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TimeFValidationError(f"{column} is nonblank text")
    return value


def _enum(kind: type[E], value: str, column: str) -> E:
    if value not in set(kind):
        raise TimeFValidationError(f"{column} {value!r} is not one of {[member.value for member in kind]}")
    return kind(value)


def parse_inputs(text: str, task: SourceTask) -> tuple[list[float], ...]:
    """Decode a row's ``input_ts``: one series, or a list of two for comparison.

    Returns:
        The input series, in published order, as floats.

    Raises:
        TimeFValidationError: If ``input_ts`` isn't a nonempty list of finite numbers, or two for comparison.
    """
    value = decode_json(text, "input_ts")
    if task.input_series == 1:
        return (checked_series(value),)
    if not isinstance(value, list) or len(value) != task.input_series:
        raise TimeFValidationError(f"a {task} input_ts holds {task.input_series} series")
    return tuple(checked_series(series) for series in value)


def _answer(answer: str, question: Question) -> str:
    """Require the answer to be one of the question's letters, or, for a puzzle, an order of all four.

    Returns:
        The answer, as published.

    Raises:
        TimeFValidationError: If the answer doesn't fit the question.
    """
    if question.format is QuestionFormat.PUZZLING:
        if sorted(answer.split(",")) != sorted(question.letters):
            raise TimeFValidationError(f"a puzzle's answer orders {', '.join(question.letters)}, got {answer!r}")
    elif answer not in question.letters:
        raise TimeFValidationError(f"the answer {answer!r} is not one of the question's letters {question.letters}")
    return answer


def _context(text: str, entries: int | None, column: str) -> str:
    """Check a row's ``meta_info``, ``domain`` or ``dataset``: plain text, or a JSON list of so many entries.

    Returns:
        The text, as published.

    Raises:
        TimeFValidationError: If a list is expected and the value isn't one of nonblank texts of that length.
    """
    if entries is not None and len(context_list(text, column)) != entries:
        raise TimeFValidationError(f"{column} is a JSON list of {entries} entries")
    return text


def context_list(text: str, column: str) -> list[str]:
    """Decode a published ``meta_info``, ``domain`` or ``dataset`` that holds a JSON list.

    Args:
        text: The published text.
        column: The column's name, for the error message.

    Returns:
        The list's entries.

    Raises:
        TimeFValidationError: If the text isn't a JSON list of nonblank texts.
    """
    value = decode_json(text, column)
    if not isinstance(value, list) or not all(isinstance(entry, str) and entry.strip() for entry in value):
        raise TimeFValidationError(f"{column} is a JSON list of nonblank texts")
    return value


def check_schema(path: Path) -> None:
    """Require a split file to have the release's columns, in order, all strings.

    Raises:
        TimeFValidationError: If the schema differs.
    """
    schema = pq.read_schema(path)
    found = [(field.name, field.type) for field in schema]
    if found != [(column, pa.string()) for column in PUBLISHED_COLUMNS]:
        raise TimeFValidationError(f"{path}: expected the string columns {list(PUBLISHED_COLUMNS)}, found {found}")


def read_rows(path: Path, split: SourceSplit) -> Iterator[Row]:
    """Validate every row of one split file, in file order.

    Yields:
        Each row.

    Raises:
        TimeFValidationError: If the schema or any row is malformed, with file and row.
    """  # noqa: DOC502 - raised by check_schema and Row.parse
    check_schema(path)
    position = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS, columns=list(_READ_COLUMNS)):
        for values in batch.to_pylist():
            yield Row.parse(values, RowRef(split, position))
            position += 1


def read_series_columns(path: Path) -> tuple[pa.StringArray, pa.StringArray]:
    """Read a split file's ``input_ts`` and ``question`` columns, the text every series of its rows is in.

    The file is one row group, so a single row can't be read alone; a reader holds one file's two
    columns while it rereads that file's rows.

    Returns:
        The two columns, each as one array indexed by row.

    Raises:
        TimeFFormatError: If the file can't be read, as when it changed or vanished since conversion.
    """
    try:
        table = pq.read_table(path, columns=_SERIES_COLUMNS).combine_chunks()
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    input_ts, question = (table.column(name).chunk(0) for name in _SERIES_COLUMNS)
    if not isinstance(input_ts, pa.StringArray) or not isinstance(question, pa.StringArray):
        raise TimeFFormatError(f"{path}: input_ts and question are string columns")
    return input_ts, question
