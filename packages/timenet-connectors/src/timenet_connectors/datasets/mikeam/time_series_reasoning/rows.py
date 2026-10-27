"""Read the published files one row per line: each row validated, with its line's offset for a reread.

A row keeps its texts as published, edge whitespace included, since the authors' prompts contain them,
and its metadata in published key order, since their prompt prints ``str(metadata)``. It keeps each
series' shape, not its values: a lazy loader rereads them from the line with :func:`reread_values`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, unique
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet_connectors.datasets.mikeam.time_series_reasoning.issues import SourceIssue
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile


# Nonblank, and kept unstripped.
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]
Uuid = Annotated[
    StrictStr, StringConstraints(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
]
# Every question offers four options; the authors' configs set num_classes: 4.
Options = Annotated[list[Text], Field(min_length=4, max_length=4)]


@unique
class ValueType(StrEnum):
    """The dtype a series is stored as, chosen so no published value changes.

    ``int32``, TimeF's widest integer dtype, for a series written in integers only; ``float64`` otherwise.
    """

    INT32 = "int32"
    FLOAT64 = "float64"

    @property
    def exact_ints(self) -> range:
        """Return the integers the dtype holds exactly."""
        return _EXACT_INTS[self]


_EXACT_INTS: dict[ValueType, range] = {
    ValueType.INT32: range(-(2**31), 2**31),
    ValueType.FLOAT64: range(-(2**53), 2**53 + 1),
}


@dataclass(frozen=True)
class RowRef:
    """Where one row lives: its file, its line, and the line's first byte."""

    path: Path
    file: SourceFile
    line: int
    """The row's line number, counted from 1."""
    offset: int

    @property
    def context(self) -> str:
        """Return where the row is, for error messages: its file and line."""
        return f"{self.path} (line {self.line})"


class SeriesShape(BaseModel):
    """A published series without its values: what a reread must find again."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    length: int
    dtype: ValueType
    digest: str
    """The sha256 of the values as JSON, which tells copies of a series apart across files."""

    @classmethod
    def of(cls, values: Any) -> Self:
        """Check a published series and describe it.

        Returns:
            Its length, value type and digest.

        Raises:
            TimeFValidationError: If it isn't a nonempty list of finite numbers that its value type
                holds exactly.
        """
        if not isinstance(values, list) or not values:
            raise TimeFValidationError("a series must be a nonempty list of numbers")
        ints = 0
        for value in values:
            # type(), not isinstance(): a JSON true decodes to bool, an int subclass, and isn't a number.
            if type(value) is int:
                ints += 1
            elif type(value) is not float or not math.isfinite(value):
                raise TimeFValidationError(f"a series holds {value!r}, which isn't a finite number")
        dtype = ValueType.INT32 if ints == len(values) else ValueType.FLOAT64
        exact = dtype.exact_ints
        if ints and any(type(value) is int and value not in exact for value in values):
            raise TimeFValidationError(f"a series holds an integer that {dtype} can't hold exactly")
        digest = hashlib.sha256(json.dumps(values).encode()).hexdigest()
        return cls(length=len(values), dtype=dtype, digest=digest)


class Row(BaseModel):
    """A row of ``TS_Dataset.jsonl``: one series and its texts, validated. Use :meth:`parse` to build one.

    Every other row repeats these fields for its series and adds its own, so the other row classes
    extend this one. The row also carries where it was read from, which validation takes as context.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    ref: InstanceOf[RowRef] = Field(exclude=True)
    """Where the row was read from. It comes from the validation context, never from the JSON."""
    uuid: Uuid
    description: Text
    description_short: Text
    description_tiny: Text
    characteristics: Text
    generator: Text
    """The authors' Python code that generates the series."""
    metadata: dict[Text, Text]
    """Free text such as ``start``, ``end``, ``units`` and ``frequency``, in published key order. The
    keys vary between rows."""
    series_shape: SeriesShape = Field(alias="series")
    """The series' shape. Its values stay in the file."""

    @model_validator(mode="before")
    @classmethod
    def _add_ref(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        # The JSON must not supply the reference, so a published "ref" key is rejected, not used.
        if "ref" in data:
            raise TimeFValidationError("a published row has no 'ref' key")
        return {**data, "ref": _context_ref(info)}

    @field_validator("series_shape", mode="before")
    @classmethod
    def _series_shape(cls, values: Any) -> SeriesShape:
        return SeriesShape.of(values)

    @classmethod
    def parse(cls, value: Any, ref: RowRef) -> Self:
        """Validate a published row, and attach where it was read from.

        Returns:
            The row.

        Raises:
            TimeFValidationError: If the row is malformed, with its file and line.
        """
        try:
            if not isinstance(value, dict):
                raise TimeFValidationError("expected a row object")
            return cls.model_validate(value, context={"ref": ref})
        # A pydantic ValidationError is a ValueError, so it gains the same context.
        except (ValueError, OverflowError, RecursionError) as exc:
            raise TimeFValidationError(f"{ref.context}: {exc}") from exc


@unique
class ScenarioCategory(StrEnum):
    """A description row's published ``scenario_category``: one of the release's 10 domains.

    The train and validation files hold the first 7, and the test file holds the last 3, which the
    authors hold out (the paper's Appendix B.1).
    """

    SALES_AND_MARKET_TRENDS = "Sales and Market Trends"
    ENVIRONMENTAL_AND_CLIMATE_DATA = "Environmental and Climate Data"
    ENERGY_AND_RESOURCE_CONSUMPTION = "Energy and Resource Consumption"
    RECREATIONAL_AND_ENTERTAINMENT_TRENDS = "Recreational and Entertainment Trends"
    TECHNOLOGICAL_AND_DIGITAL_TRENDS = "Technological and Digital Trends"
    TRANSPORT_AND_TRAFFIC_TRENDS = "Transport and Traffic Trends"
    WILDLIFE_AND_NATURE_OBSERVATION = "Wildlife and Nature Observation"
    HEALTH_AND_MEDICAL_DATA = "Health and Medical Data"
    AGRICULTURAL_AND_FOOD_PRODUCTION = "Agricultural and Food Production"
    EDUCATIONAL_AND_PUBLIC_SERVICES = "Educational and Public Services"


class DescriptionRow(Row):
    """A row of ``TS_Dataset_MCQ/{train,val,test}.json``: pick the series' description among four.

    The authors' ``ts2desc_mcq.yaml`` labels a row by its ``description``, so the answer is the option
    equal to it. The published options list it first in every row, and the authors' loader shuffles
    them, unseeded, before it formats them.
    """

    options: Options
    # JSON gives the category as a string, which strict mode rejects for an enum. Lax mode still accepts
    # only the enum's values.
    scenario_category: ScenarioCategory = Field(strict=False)
    """The series' domain."""

    @model_validator(mode="after")
    def _description_is_the_first_option(self) -> Self:
        if self.options[0] != self.description:
            raise TimeFValidationError("the first option isn't the row's description")
        return self


class QuestionRow(Row):
    """A row of ``MCQ_1_TS.jsonl``: a question about one series, with four options.

    Some published questions and options are blank, and some options repeat. They are kept as
    published: :attr:`source_issues` names them, and ``release.check_blank_lines`` pins the blank rows.
    """

    question: StrictStr
    options: Annotated[list[StrictStr], Field(min_length=4, max_length=4)]
    answer_index: Annotated[StrictInt, Field(ge=0, le=3)]
    """The position of the right option, which the authors' loader formats as a letter."""

    @property
    def source_issues(self) -> tuple[SourceIssue, ...]:
        """Return the row's issues, in :class:`SourceIssue` order, or none."""
        answer = self.options[self.answer_index]
        found = {
            SourceIssue.BLANK_QUESTION: not self.question.strip(),
            SourceIssue.BLANK_OPTION: any(not option.strip() for option in self.options),
            SourceIssue.REPEATED_OPTION: len(set(self.options)) < len(self.options),
            SourceIssue.ANSWER_TEXT_REPEATED: self.options.count(answer) > 1,
        }
        return tuple(issue for issue in SourceIssue if found[issue])

    @property
    def has_blank_text(self) -> bool:
        """Return whether the question or an option is blank, which only the pinned lines may be."""
        return not {SourceIssue.BLANK_QUESTION, SourceIssue.BLANK_OPTION}.isdisjoint(self.source_issues)


@unique
class QuestionCategory(StrEnum):
    """A two-series question's published ``category``.

    The release's README: the "category of question that checks either the similarity or difference
    between the counterfactual and the original series".
    """

    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


class CounterfactualRow(QuestionRow):
    """A row of ``MCQ_2_TS.jsonl``: a question that compares a series with a changed version of it."""

    # JSON gives the category as a string, which strict mode rejects for an enum. Lax mode still accepts
    # only the enum's values.
    category: QuestionCategory = Field(strict=False)
    counterfactual: Text
    """The change, as a question: "What if ...?". The authors' prompt doesn't include it."""
    new_generator: Text
    new_series_shape: SeriesShape = Field(alias="new_series")
    """The changed series' shape. Its values stay in the file."""

    @field_validator("new_series_shape", mode="before")
    @classmethod
    def _new_series_shape(cls, values: Any) -> SeriesShape:
        return SeriesShape.of(values)


R = TypeVar("R", bound=Row)


def _context_ref(info: ValidationInfo) -> RowRef:
    ref = (info.context or {}).get("ref")
    if not isinstance(ref, RowRef):
        raise TimeFValidationError("a row is validated with its RowRef as context")
    return ref


def _distinct_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        # Reject a repeated field instead of keeping its last value.
        if key in result:
            raise TimeFValidationError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise TimeFValidationError(f"{name} is not a JSON number")


def _decode(raw: bytes, ref: RowRef) -> Any:
    """Decode one line, rejecting repeated fields and the NaN and Infinity tokens Python accepts.

    Returns:
        The decoded value.

    Raises:
        TimeFValidationError: If the line isn't one valid JSON value, with its file and line.
    """
    try:
        return json.loads(raw, object_pairs_hook=_distinct_fields, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc


def read_rows(path: Path, file: SourceFile, row_type: type[R]) -> Iterator[R]:
    """Validate every row of one file as ``row_type``, in file order, keeping no row's values.

    Args:
        path: The file's local path.
        file: Which source file it is, which each row's reference names.
        row_type: The row class the file's rows are, such as :class:`QuestionRow` for ``MCQ_1_TS.jsonl``.

    Yields:
        Each row.

    Raises:
        TimeFValidationError: If a line isn't a valid row of ``row_type``, with its file and line.
    """  # noqa: DOC502 - raised by _decode and Row.parse
    with path.open("rb") as handle:
        line = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                return
            line += 1
            ref = RowRef(path, file, line, offset)
            yield row_type.parse(_decode(raw, ref), ref)


def reread_values(ref: RowRef, field: Literal["series", "new_series"], expected: SeriesShape) -> list[int | float]:
    """Reread one series from its row's line, and check it is the series the row was read with.

    Returns:
        The series' values, as published.

    Raises:
        TimeFFormatError: If the line no longer holds that series, as when the file changed.
    """
    with ref.path.open("rb") as handle:
        handle.seek(ref.offset)
        raw = handle.readline()
    try:
        row = _decode(raw, ref)
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"{ref.context}: {exc}") from exc
    values = row.get(field) if isinstance(row, dict) else None
    if not isinstance(values, list):
        raise TimeFFormatError(f"{ref.context}: {field} is not a list; the file changed")
    try:
        shape = SeriesShape.of(values)
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"{ref.context}: {exc}") from exc
    if shape != expected:
        raise TimeFFormatError(f"{ref.context}: {field} differs from the converted row; the file changed")
    return values
