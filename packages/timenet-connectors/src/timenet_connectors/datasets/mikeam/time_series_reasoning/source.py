"""Read and validate reasoning rows, index source identities and detect changed files."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Generic, NamedTuple, TypeVar

import pyarrow as pa
from pydantic import Field

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.datasets.mikeam.time_series_reasoning.vocabulary import (
    DESCRIPTION_FILES,
    SERIES_NAME,
    SERIES_TEXTS,
    AnnotationKey,
    CategoryT,
    File,
    QuestionCategory,
    ScenarioCategory,
    SeriesField,
    SourceIssue,
    ValueType,
)
from timenet_connectors.sources.errors import error_context
from timenet_connectors.sources.jsonl import JsonlLocation, iter_jsonl_rows, read_jsonl_row
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.sources.rows import NonBlankText, SourceRow
from timenet_connectors.sources.signals import numeric_signal


class SeriesFingerprint(NamedTuple):
    """A series without its values: what a reread must find again."""

    length: int
    dtype: ValueType
    digest: str
    """The sha256 of the values as JSON, which tells copies of a series apart across files."""


def _fingerprint_of(values: list[int | float]) -> SeriesFingerprint:
    """Check a published series and describe it.

    Returns:
        Its length, value type and digest.

    Raises:
        TimeNetConnectorError: If a sample is nonfinite or cannot be represented exactly.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by numeric_signal
    dtype = ValueType.INT32 if all(type(value) is int for value in values) else ValueType.FLOAT64
    numeric_signal(values, "series", spec=dtype.spec)
    return SeriesFingerprint(len(values), dtype, hashlib.sha256(json.dumps(values).encode()).hexdigest())


@dataclass(frozen=True)
class RowLocation:
    """Where one row lives: its file, its line, and the line's first byte."""

    location: JsonlLocation
    file: File

    @property
    def context(self) -> str:
        """Where the row is, for error messages."""
        return self.location.context


class _SeriesFields(SourceRow):
    uuid: NonBlankText
    description: NonBlankText
    description_short: NonBlankText
    description_tiny: NonBlankText
    characteristics: NonBlankText
    generator: NonBlankText
    metadata: dict[NonBlankText, NonBlankText]
    series: list[int | float] = Field(min_length=1)


class DescriptionFields(_SeriesFields):
    """A description question with four published options."""

    scenario_category: NonBlankText
    options: list[NonBlankText] = Field(min_length=4, max_length=4)


class QuestionFields(_SeriesFields):
    """A question and its published answer index; blank text is retained."""

    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, lt=4)


class CounterfactualFields(QuestionFields):
    """A question comparing the original series with its counterfactual."""

    category: NonBlankText
    counterfactual: NonBlankText
    new_generator: NonBlankText
    new_series: list[int | float] = Field(min_length=1)


_ROW_MODELS = {
    File.SERIES: _SeriesFields,
    **dict.fromkeys(DESCRIPTION_FILES, DescriptionFields),
    File.QUESTIONS: QuestionFields,
    File.COUNTERFACTUALS: CounterfactualFields,
}


RowT = TypeVar("RowT", bound=_SeriesFields)


def _fingerprint(values: list[int | float], ref: RowLocation, key: str) -> SeriesFingerprint:
    with error_context(f"{ref.context}: {key}"):
        return _fingerprint_of(values)


def _category(value: str, kind: type[CategoryT], ref: RowLocation, key: str) -> CategoryT:
    if value not in set(kind):
        raise TimeNetConnectorError(
            f"{ref.context}: {key} must be one of {[item.value for item in kind]}, got {value!r}"
        )
    return kind(value)


def _series(ref: RowLocation, row: _SeriesFields) -> "SeriesDescriptor":
    texts = {key: getattr(row, key.value) for key in SERIES_TEXTS}
    # Source dictionary order is part of the authors' printed prompt.
    texts[AnnotationKey.METADATA] = str(row.metadata)
    return SeriesDescriptor(ref, row.uuid, MappingProxyType(texts), _fingerprint(row.series, ref, "series"))


@dataclass(frozen=True)
class SeriesDescriptor:
    """A series as one row publishes it, without its values."""

    ref: RowLocation
    uuid: str
    texts: Mapping[AnnotationKey, str]
    fingerprint: SeriesFingerprint

    def check_copy(self, main: "SeriesDescriptor", *, values: bool) -> None:
        """Require this row's copy of a series to equal the series' own row.

        Args:
            main: The series' row in ``TS_Dataset.jsonl``.
            values: Whether the values must match too. The description files hold rounded copies, so
                only their texts must.

        Raises:
            TimeNetConnectorError: If the texts, the metadata or, when asked, the values differ.
        """
        differs = [key.value for key in main.texts if self.texts[key] != main.texts[key]]
        if values and self.fingerprint != main.fingerprint:
            differs.append(SERIES_NAME)
        if differs:
            raise TimeNetConnectorError(
                f"{self.ref.context}: {differs} differ from the series' row, {File.SERIES} line {main.ref.location.line}"
            )


class FileReader(Generic[RowT]):
    """Reads one file's rows in order and hashes its bytes, so a reread can tell the file changed."""

    def __init__(self, path: Path, file: File, model: type[RowT]) -> None:
        self.path = path
        self.file = file
        self.model = model
        self._hash = hashlib.blake2b(digest_size=16)

    @property
    def digest(self) -> str:
        """The digest of the bytes read so far."""
        return self._hash.hexdigest()

    def rows(self) -> Iterator[tuple[RowLocation, RowT]]:
        """Decode every line, in file order.

        Yields:
            Each row.
        """
        for located in iter_jsonl_rows(self.path, self.model):
            self._hash.update(located.raw)
            location = located.location
            ref = RowLocation(location, self.file)
            yield ref, located.value


def _read_values(ref: RowLocation, key: SeriesField) -> list[int | float]:
    """Read the selected series from its recorded source location.

    Returns:
        The series' values, as published.

    Raises:
        TimeFValidationError: If the selector is unknown or incompatible with the source file.
    """
    if not isinstance(key, SeriesField):
        raise TimeFValidationError(f"expected a SeriesField, got {key!r}")
    if key is SeriesField.COUNTERFACTUAL and ref.file is not File.COUNTERFACTUALS:
        raise TimeFValidationError(f"{ref.context}: {key} requires a counterfactual row")
    row = read_jsonl_row(ref.location, _ROW_MODELS[ref.file]).value
    if key is SeriesField.COUNTERFACTUAL:
        if not isinstance(row, CounterfactualFields):
            raise TimeFValidationError(f"{ref.context}: {key} requires a counterfactual row")
        return row.new_series
    return row.series


def load_series(ref: RowLocation, key: SeriesField, fingerprint: SeriesFingerprint) -> pa.Array:
    """Return validated samples after checking the lazy reread against its scan."""
    values = checked_reread(
        partial(_read_values, ref, key),
        fingerprint=lambda values: _fingerprint(values, ref, key),
        expected=fingerprint,
        context=f"{ref.context}: {key}",
    )
    return numeric_signal(values, f"{ref.context}: {key}", spec=fingerprint.dtype.spec)


@dataclass(frozen=True)
class CounterfactualSeries:
    """The changed series of one counterfactual, as its first row publishes it."""

    ref: RowLocation
    new_generator: str
    fingerprint: SeriesFingerprint


class RoundedCopyKey(NamedTuple):
    """A source series and the digest of its rounded copy."""

    uuid: str
    digest: str


class CounterfactualKey(NamedTuple):
    """A source series and its published counterfactual description."""

    uuid: str
    counterfactual: str


@dataclass(frozen=True)
class Release:
    """A completed scan of the source release."""

    paths: Mapping[File, Path]
    digests: Mapping[File, str]
    series: Mapping[str, SeriesDescriptor]
    rounded_copies: Mapping[RoundedCopyKey, SeriesDescriptor]
    counterfactuals: Mapping[CounterfactualKey, CounterfactualSeries]

    @classmethod
    def read(cls, paths: Mapping[File, Path]) -> "Release":
        """Return the completed release after validating every source file."""
        scan = _ReleaseBuilder(dict(paths))
        scan.read()
        return cls(
            MappingProxyType(dict(scan.paths)),
            MappingProxyType(scan.digests),
            MappingProxyType(scan.series),
            MappingProxyType(scan.rounded_copies),
            MappingProxyType(scan.counterfactuals),
        )


@dataclass
class _ReleaseBuilder:
    """Mutable state confined to a single release scan."""

    paths: Mapping[File, Path]
    digests: dict[File, str] = field(default_factory=dict)
    """The digest of each file's bytes, which pass two checks."""
    series: dict[str, SeriesDescriptor] = field(default_factory=dict)
    """One entry per row of ``TS_Dataset.jsonl``, by uuid, in file order."""
    rounded_copies: dict[RoundedCopyKey, SeriesDescriptor] = field(default_factory=dict)
    """One descriptor per distinct series in the description files that differs from its series' row, by
    uuid and value digest."""
    counterfactuals: dict[CounterfactualKey, CounterfactualSeries] = field(default_factory=dict)
    """One entry per changed series, by uuid and counterfactual, in file order."""

    def read(self) -> None:
        """Read each file once, in :class:`File` order, and check that the files agree.

        Raises:
            TimeNetConnectorError: If a row is malformed, a uuid repeats or is unknown, a row's copy of
                its series differs from ``TS_Dataset.jsonl``, or a counterfactual holds two changed series.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
        self._read_series()
        for file in DESCRIPTION_FILES:
            self._read_descriptions(file)
        self._read_questions()
        self._read_counterfactuals()

    def _reader(self, file: File, model: type[RowT]) -> FileReader[RowT]:
        return FileReader(self.paths[file], file, model)

    def _read_series(self) -> None:
        reader = self._reader(File.SERIES, _SeriesFields)
        for ref, row in reader.rows():
            series = _series(ref, row)
            if series.uuid in self.series:
                first = self.series[series.uuid].ref.location.line
                raise TimeNetConnectorError(f"{ref.context}: uuid {series.uuid} repeats line {first}")
            self.series[series.uuid] = series
        self.digests[File.SERIES] = reader.digest

    def _main(self, copy: SeriesDescriptor) -> SeriesDescriptor:
        """Return the series' row that a row's copy repeats.

        Raises:
            TimeNetConnectorError: If the uuid is not in ``TS_Dataset.jsonl``.
        """
        if copy.uuid not in self.series:
            raise TimeNetConnectorError(f"{copy.ref.context}: uuid {copy.uuid} is not in {File.SERIES}")
        return self.series[copy.uuid]

    def _read_descriptions(self, file: File) -> None:
        reader = self._reader(file, DescriptionFields)
        for ref, row in reader.rows():
            copy = read_description(ref, row).series
            main = self._main(copy)
            copy.check_copy(main, values=False)
            if copy.fingerprint.length != main.fingerprint.length:
                raise TimeNetConnectorError(
                    f"{copy.ref.context}: a rounded copy has {copy.fingerprint.length} values, "
                    f"its series' row {main.fingerprint.length}"
                )
            key = RoundedCopyKey(copy.uuid, copy.fingerprint.digest)
            # The authors' description task reads this file's values, so a rounded copy is a Record of its own.
            if copy.fingerprint != main.fingerprint and key not in self.rounded_copies:
                self.rounded_copies[key] = copy
        self.digests[file] = reader.digest

    def _read_questions(self) -> None:
        reader = self._reader(File.QUESTIONS, QuestionFields)
        for ref, row in reader.rows():
            question = read_question(ref, row)
            question.series.check_copy(self._main(question.series), values=True)
        self.digests[File.QUESTIONS] = reader.digest

    def _read_counterfactuals(self) -> None:
        reader = self._reader(File.COUNTERFACTUALS, CounterfactualFields)
        for ref, row in reader.rows():
            counterfactual = read_counterfactual(ref, row)
            counterfactual.question.series.check_copy(self._main(counterfactual.question.series), values=True)
            self._add_counterfactual(counterfactual)
        self.digests[File.COUNTERFACTUALS] = reader.digest

    def _add_counterfactual(self, row: "_Counterfactual") -> None:
        """Keep the first row of a counterfactual, or check a later row's changed series against it.

        Raises:
            TimeNetConnectorError: If the rows of one counterfactual hold different changed series.
        """
        ref, uuid = row.question.series.ref, row.question.series.uuid
        key = CounterfactualKey(uuid, row.counterfactual)
        known = self.counterfactuals.get(key)
        if known is None:
            self.counterfactuals[key] = CounterfactualSeries(ref, row.new_generator, row.new_fingerprint)
        elif (known.new_generator, known.fingerprint) != (row.new_generator, row.new_fingerprint):
            raise TimeNetConnectorError(
                f"{ref.context}: the changed series differs from line {known.ref.location.line}'s for the same counterfactual"
            )


@dataclass(frozen=True)
class Description:
    """A row of ``TS_Dataset_MCQ/{train,val,test}.json``: pick the series' description among four."""

    series: SeriesDescriptor
    options: tuple[str, ...]
    """The published options, with the series' description first."""
    scenario_category: ScenarioCategory


def read_description(ref: RowLocation, row: DescriptionFields) -> Description:
    """Validate a description row.

    The authors' ``ts2desc_mcq.yaml`` labels a row by its ``description``, so the answer is the option
    equal to it, and the release lists it first in every row.

    Returns:
        The row.

    Raises:
        TimeNetConnectorError: If the row is malformed, or its first option is not its description.
    """
    series = _series(ref, row)
    options = tuple(row.options)
    if options[0] != series.texts[AnnotationKey.DESCRIPTION]:
        raise TimeNetConnectorError(f"{ref.context}: the first option is not the row's description")
    return Description(series, options, _category(row.scenario_category, ScenarioCategory, ref, "scenario_category"))


@dataclass(frozen=True)
class Question:
    """A row of ``MCQ_1_TS.jsonl``: a question about one series, with four options.

    Some published questions and options are blank, and some options repeat. They are kept as published,
    and :attr:`issues` names them.
    """

    series: SeriesDescriptor
    question: str
    options: tuple[str, ...]
    answer_index: int
    """The position of the right option, which the authors' loader formats as a letter."""

    @property
    def answer_text(self) -> str:
        """The right option, the text its target letter stands for."""
        return self.options[self.answer_index]

    @property
    def issues(self) -> list[SourceIssue]:
        """The row's issues, in :class:`SourceIssue` order, or none."""
        found = {
            SourceIssue.BLANK_QUESTION: not self.question.strip(),
            SourceIssue.BLANK_OPTION: any(not option.strip() for option in self.options),
            SourceIssue.REPEATED_OPTION: len(set(self.options)) < len(self.options),
            SourceIssue.ANSWER_TEXT_REPEATED: self.options.count(self.answer_text) > 1,
        }
        return [issue for issue in SourceIssue if found[issue]]


def read_question(ref: RowLocation, row: QuestionFields) -> Question:
    """Validate a question row. Its question and options may be blank, as published.

    Returns:
        The row.

    Raises:
        TimeNetConnectorError: If the row is malformed.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by source validation
    return Question(
        _series(ref, row),
        row.question,
        tuple(row.options),
        row.answer_index,
    )


@dataclass(frozen=True)
class _Counterfactual:
    """A row of ``MCQ_2_TS.jsonl``: a question that compares a series with a changed version of it."""

    question: Question
    category: QuestionCategory
    counterfactual: str
    """The change, as a question: "What if ...?". The authors' prompt does not include it."""
    new_generator: str
    new_fingerprint: SeriesFingerprint
    """The counterfactual series' fingerprint. Its values stay in the file."""


def read_counterfactual(ref: RowLocation, row: CounterfactualFields) -> _Counterfactual:
    """Validate a counterfactual row.

    Returns:
        The row.

    Raises:
        TimeNetConnectorError: If the row is malformed.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by source validation
    return _Counterfactual(
        read_question(ref, row),
        _category(row.category, QuestionCategory, ref, "category"),
        row.counterfactual,
        row.new_generator,
        _fingerprint(row.new_series, ref, "new_series"),
    )
