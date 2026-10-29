"""Read the published VeriTime JSON Lines: their rows, the two series formats, the walk, and the reread.

Each file holds one JSON object per line. Walk it one value at a time and keep each value's UTF-8 byte
range, so a signal loader rereads just its row instead of holding every series in memory.

The two families publish ``timeseries2`` differently. A scenario row holds a list with one list of
numbers, the ChatTS series rounded to four decimals. A knowledge row holds one string of numbers, each
with exactly four decimals, separated by ``", "``, which the authors' scripts paste into the question
as it is.
"""

from array import array
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Annotated, Any, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    InstanceOf,
    Strict,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import Family, Part, SourceTask
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import CLASS_VOCABULARIES
from timenet_connectors.sources.json_stream import JsonCursor, read_entry


SERIES_MARKER = "<ts><ts/>"
"""Where a question takes its series. The authors' scripts replace it with the series as text."""

# Strict types: a JSON true or "1" must not pass as a number, and a blank string not as text.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
Number = Annotated[FiniteFloat, Strict()]

# A scenario series: one nonempty list of finite numbers, wrapped in a list.
_SCENARIO_SERIES = TypeAdapter(
    Annotated[list[Annotated[list[Number], Field(min_length=1)]], Field(min_length=1, max_length=1)]
)
_KNOWLEDGE_VALUE = re.compile(r"-?[0-9]+\.[0-9]{4}")
_KNOWLEDGE_SEPARATOR = ", "
# Every ChatTS question states its series' length this way.
_STATED_LENGTH = re.compile(r"with length of ([0-9]+):")
# An inferential calculation label: a count, except one two-part row whose label is a decimal amplitude.
_QUANTITY = re.compile(r"[0-9]+(\.[0-9]+)?")


@dataclass(frozen=True)
class RowRef:
    """Where one row lives: its file, its position in the file, and its byte range."""

    path: Path
    part: Part
    row: int
    """The row's position in its file, from 0."""
    offset: int
    length: int

    @property
    def context(self) -> str:
        """Return where the row is, for error messages: its file and position."""
        return f"{self.path} (row {self.row})"


class PublishedSeries(NamedTuple):
    """What a row keeps of its series: the length, and a digest that tells equal series apart."""

    n_values: int
    digest: str
    """A hash of the float64 values, so rows that publish the same series can share a Record."""

    @classmethod
    def of(cls, values: list[float]) -> Self:
        """Summarize decoded values.

        Returns:
            Their length and digest.
        """
        return cls(len(values), hashlib.blake2b(array("d", values).tobytes(), digest_size=16).hexdigest())


def parse_series(published: Any, family: Family) -> list[float]:
    """Decode a row's ``timeseries2`` in its family's format.

    Returns:
        The series' values, in order.

    Raises:
        TimeFValidationError: If the value violates the family's format.
    """
    if family is Family.SCENARIO:
        (values,) = _SCENARIO_SERIES.validate_python(published)
        return values
    if not isinstance(published, str):
        raise TimeFValidationError("a knowledge timeseries2 must be a string of numbers")
    tokens = published.split(_KNOWLEDGE_SEPARATOR)
    if not all(_KNOWLEDGE_VALUE.fullmatch(token) for token in tokens):
        raise TimeFValidationError(
            f"a knowledge timeseries2 must hold four-decimal numbers separated by {_KNOWLEDGE_SEPARATOR!r}"
        )
    values = [float(token) for token in tokens]
    # The pattern admits no inf or nan, but a long run of digits still overflows to inf.
    if not all(math.isfinite(value) for value in values):
        raise TimeFValidationError("a knowledge timeseries2 holds a value too large for float64")
    return values


class Row(BaseModel):
    """A row's published fields, validated, with its series' length and digest but not its values.

    Each field keeps the published name as its alias, so a validation error names the source field.
    The row also carries where it was read from, which validation takes as context, since the file's
    family decides the series' format. Use :meth:`parse` to build one.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    ref: InstanceOf[RowRef] = Field(exclude=True)
    """Where the row was read from. It comes from the validation context, never from the JSON."""
    task: Annotated[SourceTask, Strict(False)]
    question: Text
    """The question as the authors pose it, with one :data:`SERIES_MARKER` where the series goes."""
    output: Text | None = None
    """Scenario rows only: the ChatTS answer the label was extracted from."""
    label: Text
    """The ground-truth answer the authors score against."""
    cot: Text
    """The six-step chain of thought and the answer, as the authors fine-tune on it."""
    step1_label: Text | None
    step2_label: Text
    step4_label: Text | None
    step6_label: Text
    series: InstanceOf[PublishedSeries] = Field(alias="timeseries2")
    """The series' length and digest. The values stay in the file."""

    @model_validator(mode="before")
    @classmethod
    def _add_ref(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        # The JSON must not supply the reference, so a published "ref" key is rejected, not used.
        if "ref" in data:
            raise TimeFValidationError("a published row has no 'ref' key")
        return {**data, "ref": _context_ref(info)}

    @field_validator("series", mode="before")
    @classmethod
    def _series(cls, published: Any, info: ValidationInfo) -> PublishedSeries:
        return PublishedSeries.of(parse_series(published, _context_ref(info).part.family))

    @field_validator("question")
    @classmethod
    def _one_marker(cls, question: str) -> str:
        if question.count(SERIES_MARKER) != 1:
            raise TimeFValidationError(f"a question must hold {SERIES_MARKER!r} exactly once")
        return question

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        family = self.ref.part.family
        if self.task.family is not family:
            raise TimeFValidationError(f"a {family} file holds a {self.task.value!r} row")
        if (self.output is not None) != (family is Family.SCENARIO):
            raise TimeFValidationError("output must be present exactly on scenario rows")
        classes = CLASS_VOCABULARIES.get(self.task)
        if classes is not None and self.label not in classes:
            raise TimeFValidationError(f"label {self.label!r} is not one of {classes}")
        if self.task is SourceTask.INFERENTIAL_CALCULATION and not _QUANTITY.fullmatch(self.label):
            raise TimeFValidationError(f"an inferential calculation label must be a number, got {self.label!r}")
        if family is Family.SCENARIO:
            stated = _STATED_LENGTH.search(self.question)
            if stated is None or int(stated.group(1)) != self.series.n_values:
                raise TimeFValidationError(f"the question does not state the series' length, {self.series.n_values}")
        return self

    @property
    def quantity(self) -> int | float:
        """Return an inferential calculation row's label as the number it states.

        The label is a count, an int, in every row but one, whose two-part question also asks an
        amplitude and whose label is that amplitude, a float. Both are kept as published.

        Raises:
            TimeFValidationError: If the row is of another task.
        """
        if self.task is not SourceTask.INFERENTIAL_CALCULATION:
            raise TimeFValidationError(f"a {self.task.value!r} row has no quantity")
        return int(self.label) if self.label.isdecimal() else float(self.label)

    @classmethod
    def parse(cls, value: Any, ref: RowRef) -> Self:
        """Validate a published row against its file's family, and attach where it was read from.

        Returns:
            The row's fields, its series' length and digest, and its reference.

        Raises:
            TimeFValidationError: If the row is malformed, with file and position.
        """
        try:
            if not isinstance(value, dict):
                raise TimeFValidationError("expected a row object")
            return cls.model_validate(value, context={"ref": ref})
        # A pydantic ValidationError is a ValueError, so it gains the same context.
        except (ValueError, OverflowError, RecursionError) as exc:
            raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def _context_ref(info: ValidationInfo) -> RowRef:
    ref = (info.context or {}).get("ref")
    if not isinstance(ref, RowRef):
        raise TimeFValidationError("a row is validated with its RowRef as context")
    return ref


def read_rows(path: Path, part: Part) -> Iterator[Row]:
    """Validate every row of one file, in file order.

    Yields:
        Each validated row, with its reference.

    Raises:
        TimeFValidationError: If the file isn't JSON values or any row is malformed, with file and position.
    """
    with JsonCursor.open(path) as cursor:
        row = 0
        while True:
            try:
                if not cursor.peek():
                    return
                value, offset, length = cursor.value()
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise TimeFValidationError(f"{path} (row {row}): {exc}") from exc
            # Row.parse adds the same context to its own errors.
            yield Row.parse(value, RowRef(path, part, row, offset, length))
            row += 1


def reread_series(ref: RowRef) -> list[float]:
    """Reread a row and decode only its series.

    The row's other fields were validated at conversion and the tasks use those, so a reread parses
    only what it supplies. ``read_entry`` already fails if the byte range no longer holds one JSON
    value. :func:`check_series` then compares the values with the converted row.

    Returns:
        The series' values.

    Raises:
        TimeFFormatError: If the row no longer holds a valid series, as when the file changed.
    """
    value = read_entry(ref.path, ref.offset, ref.length)
    try:
        if not isinstance(value, dict):
            raise TimeFFormatError("expected a row object")
        return parse_series(value.get("timeseries2"), ref.part.family)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise TimeFFormatError(f"{ref.context}: {exc}") from exc


def check_series(values: list[float], row: Row) -> None:
    """Require reread values to be the converted row's series.

    Raises:
        TimeFFormatError: If they differ, as when the file changed since conversion.
    """
    if PublishedSeries.of(values) != row.series:
        raise TimeFFormatError(f"{row.ref.context}: the series differs from the converted row; the file changed")
