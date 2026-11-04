"""Convert the pinned Time Series Reasoning release into lazy Records and the authors' three tasks.

The six files become one Record per series, per rounded copy of a series and per changed series, and
one ``AnswerTask`` per row of the authors' task files. A task's prompt is the text the authors' loader
gives a model, and its target is the letter of the right option. README.md explains how this follows
the authors.

Pass one reads every file once, checks each row and keeps the small results, a digest of each file, and
a lazy Record for each series. Pass two rereads the five task files to stream the tasks, so they are
never all held in memory. A changed file fails the pass-two digest check.

Metadata is provenance and never model input. No annotation is a task's input: the prompt holds
everything the authors show a model.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, unique
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, ClassVar, NamedTuple, TypeVar

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef


_E = TypeVar("_E", bound=StrEnum)


@unique
class _File(StrEnum):
    """A file the connector reads, named by its path in the Hub repository.

    All six are JSON Lines, ``TS_Dataset_MCQ/*.json`` included. The release also holds
    ``TS_Dataset_MCQ/all.json``, which no config of the authors loads, and ``*_llava.json`` files for
    their image model. The connector reads neither.
    """

    SERIES = "TS_Dataset.jsonl"
    DESCRIPTION_TRAIN = "TS_Dataset_MCQ/train.json"
    DESCRIPTION_VAL = "TS_Dataset_MCQ/val.json"
    DESCRIPTION_TEST = "TS_Dataset_MCQ/test.json"
    QUESTIONS = "MCQ_1_TS.jsonl"
    COUNTERFACTUALS = "MCQ_2_TS.jsonl"

    @property
    def part(self) -> str:
        """The file's part name, the key the Hub base downloads it under."""
        return self.name.lower()


# The authors name the description splits by file. The question files hold no split: the authors' configs
# read train, val and test files for them that the release does not include.
_SPLITS = {
    _File.DESCRIPTION_TRAIN: Split.TRAIN,
    _File.DESCRIPTION_VAL: Split.VALIDATION,
    _File.DESCRIPTION_TEST: Split.TEST,
}
_DESCRIPTION_FILES = tuple(_SPLITS)


@unique
class _Annotation(StrEnum):
    """The key of an annotation the connector writes. A series' texts keep their published field names."""

    DESCRIPTION = "description"
    DESCRIPTION_SHORT = "description_short"
    DESCRIPTION_TINY = "description_tiny"
    CHARACTERISTICS = "characteristics"
    GENERATOR = "generator"
    """The authors' Python code for the Record's series: ``generator``, or ``new_generator`` for a changed
    series."""
    METADATA = "metadata"
    """The series' published metadata as the authors' prompt prints it: ``str()`` of the dict, in
    published key order."""
    COUNTERFACTUAL = "counterfactual"
    """A changed series' change, as a "What if ...?" question."""
    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"


_SERIES_TEXTS = (
    _Annotation.DESCRIPTION,
    _Annotation.DESCRIPTION_SHORT,
    _Annotation.DESCRIPTION_TINY,
    _Annotation.CHARACTERISTICS,
    _Annotation.GENERATOR,
)
"""The texts every row repeats for its series, each under its published field name."""


@unique
class _TaskMeta(StrEnum):
    """A key of the task metadata. Metadata is provenance, never model input."""

    FILE = "file"
    LINE = "line"
    """The task's line in its file, counted from 1."""
    SCENARIO_CATEGORY = "scenario_category"
    ANSWER_INDEX = "answer_index"
    """A question's published position of the right option."""
    ANSWER_TEXT = "answer_text"
    """A question's right option, the text its target letter stands for."""
    CATEGORY = "category"
    SOURCE_ISSUES = "source_issues"


@unique
class _SourceIssue(StrEnum):
    """A problem in a published question row, which the connector keeps as published."""

    BLANK_QUESTION = "blank_question"
    BLANK_OPTION = "blank_option"
    REPEATED_OPTION = "repeated_option"
    """Two options have the same text."""
    ANSWER_TEXT_REPEATED = "answer_text_repeated"
    """The answer's text is also a wrong option's, so only ``answer_index`` tells which one the authors
    count as right."""


@unique
class _ScenarioCategory(StrEnum):
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


@unique
class _QuestionCategory(StrEnum):
    """A two-series question's published ``category``: what the question checks."""

    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


@unique
class _ValueType(StrEnum):
    """The dtype a series is stored as, chosen so no published value changes.

    ``int32``, TimeF's widest integer dtype, for a series written in integers only; ``float64`` otherwise.
    """

    INT32 = "int32"
    FLOAT64 = "float64"

    @property
    def exact_ints(self) -> range:
        """The integers the dtype holds exactly."""
        return _EXACT_INTS[self]

    @property
    def spec(self) -> TimeSeriesSpec:
        """The spec of a series of this type. One ``spec_type`` has one dtype, so each has its own."""
        return _SPECS[self]


_EXACT_INTS = {
    _ValueType.INT32: range(-(2**31), 2**31),
    _ValueType.FLOAT64: range(-(2**53), 2**53 + 1),
}
_SPECS = {
    _ValueType.FLOAT64: TimeSeriesSpec(
        spec_type="tsr_series", name="Synthetic series", unit_value=None, dtype="float64"
    ),
    _ValueType.INT32: TimeSeriesSpec(
        spec_type="tsr_integer_series", name="Synthetic integer series", unit_value=None, dtype="int32"
    ),
}
_AXIS = OrdinalAxis()
_SERIES_NAME = "series"
"""The name of every Record's one Source and its one Signal."""

# The authors' loader, MultimodalMCQDataset.__getitem__ in behavioral-data/TSandLanguage at d4e748cb
# (src/models/tasks/multimodal.py), with format_abc_mcq: the config's context_prefix, then the row's
# context_columns joined by newlines, then one line per option, "A) ...". The label is the right letter.
_DESCRIPTION_PREFIX = "What description best relates to this time series?"
"""The ``context_prefix`` of ``ts2desc_mcq.yaml``, whose ``context_columns`` is empty."""
_QUESTION_PREFIX = "Answer the following question about the time series"
"""The ``context_prefix`` of ``qa_mcq.yaml`` and ``qa_counterfactual.yaml``, whose ``context_columns`` are
``description_tiny``, ``metadata`` and ``question``."""
_OPTION_COUNT = 4
"""Every question offers four options; the authors' configs set ``num_classes: 4``."""
_LETTERS = "ABCD"


class _Shape(NamedTuple):
    """A series without its values: what a reread must find again."""

    length: int
    dtype: _ValueType
    digest: str
    """The sha256 of the values as JSON, which tells copies of a series apart across files."""


def _shape_of(values: Any) -> _Shape:
    """Check a published series and describe it.

    Returns:
        Its length, value type and digest.

    Raises:
        TimeFValidationError: If it is not a nonempty list of finite numbers that its value type holds
            exactly.
    """
    if not isinstance(values, list) or not values:
        raise TimeFValidationError("a series must be a nonempty list of numbers")
    ints = 0
    for value in values:
        # type(), not isinstance(): a JSON true decodes to bool, an int subclass, and is not a number.
        if type(value) is int:
            ints += 1
        elif type(value) is not float or not math.isfinite(value):
            raise TimeFValidationError(f"a series holds {value!r}, which is not a finite number")
    dtype = _ValueType.INT32 if ints == len(values) else _ValueType.FLOAT64
    if any(type(value) is int and value not in dtype.exact_ints for value in values):
        raise TimeFValidationError(f"a series holds an integer that {dtype} cannot hold exactly")
    return _Shape(len(values), dtype, hashlib.sha256(json.dumps(values).encode()).hexdigest())


@dataclass(frozen=True)
class _Ref:
    """Where one row lives: its file, its line, and the line's first byte."""

    path: Path
    file: _File
    line: int
    """The row's line number, counted from 1."""
    offset: int

    @property
    def context(self) -> str:
        """Where the row is, for error messages."""
        return f"{self.path} (line {self.line})"


def _distinct_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        # Reject a repeated field instead of keeping its last value.
        if key in result:
            raise TimeFValidationError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _decode(raw: bytes, ref: _Ref) -> dict[str, Any]:
    """Decode one line as a JSON object with no repeated field.

    Returns:
        The row's fields.

    Raises:
        TimeFValidationError: If the line is not one JSON object, with its file and line.
    """
    try:
        row = json.loads(raw, object_pairs_hook=_distinct_fields)
    except (ValueError, RecursionError) as exc:
        raise TimeFValidationError(f"{ref.context}: {exc}") from exc
    if not isinstance(row, dict):
        raise TimeFValidationError(f"{ref.context}: expected a row object")
    return row


class _Row:
    """One decoded line, whose typed getters fail with the file and line."""

    def __init__(self, ref: _Ref, fields: dict[str, Any]) -> None:
        self.ref = ref
        self._fields = fields

    def _get(self, key: str) -> Any:
        if key not in self._fields:
            raise TimeFValidationError(f"{self.ref.context}: missing field {str(key)!r}")
        return self._fields[key]

    def text(self, key: str, *, allow_blank: bool = False) -> str:
        """Return a text field as published, edge whitespace included.

        Raises:
            TimeFValidationError: If the field is missing, is not text, or is blank when that is not allowed.
        """
        value = self._get(key)
        if not isinstance(value, str) or not (allow_blank or value.strip()):
            raise TimeFValidationError(f"{self.ref.context}: {key} must be nonblank text, got {value!r}")
        return value

    def options(self, *, allow_blank: bool) -> list[str]:
        """Return the row's four options.

        Raises:
            TimeFValidationError: If they are not four texts, or one is blank when that is not allowed.
        """
        value = self._get("options")
        if (
            not isinstance(value, list)
            or len(value) != _OPTION_COUNT
            or not all(isinstance(option, str) and (allow_blank or option.strip()) for option in value)
        ):
            raise TimeFValidationError(f"{self.ref.context}: options must be {_OPTION_COUNT} texts, got {value!r}")
        return value

    def answer_index(self) -> int:
        """Return the position of the right option.

        Raises:
            TimeFValidationError: If it is not an integer from 0 to 3.
        """
        value = self._get("answer_index")
        if type(value) is not int or not 0 <= value < _OPTION_COUNT:
            raise TimeFValidationError(f"{self.ref.context}: answer_index must be 0 to 3, got {value!r}")
        return value

    def category(self, key: str, kind: type[_E]) -> _E:
        """Return a field that must be one of an enum's values.

        Raises:
            TimeFValidationError: If the field is not one of them.
        """
        value = self._get(key)
        if not isinstance(value, str) or value not in set(kind):
            allowed = [member.value for member in kind]
            raise TimeFValidationError(f"{self.ref.context}: {key} must be one of {allowed}, got {value!r}")
        return kind(value)

    def shape(self, key: str) -> _Shape:
        """Check a series field and describe it, without keeping its values.

        Returns:
            The series' length, value type and digest.

        Raises:
            TimeFValidationError: If it is not a valid series, with the file and line.
        """
        try:
            return _shape_of(self._get(key))
        except TimeFValidationError as exc:
            raise TimeFValidationError(f"{self.ref.context}: {key}: {exc}") from exc

    def series(self) -> "_Series":
        """Return the series every row repeats: its texts, metadata and values' shape.

        Raises:
            TimeFValidationError: If a field is missing or malformed.
        """
        metadata = self._get("metadata")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
            for key, value in metadata.items()
        ):
            raise TimeFValidationError(f"{self.ref.context}: metadata must map nonblank texts to nonblank texts")
        texts = {key: self.text(key) for key in _SERIES_TEXTS}
        # The authors' loader prints str(metadata), so a copy's key order is part of its text.
        texts[_Annotation.METADATA] = str(metadata)
        return _Series(self.ref, self.text("uuid"), texts, self.shape("series"))


@dataclass(frozen=True)
class _Series:
    """A series as one row publishes it, without its values."""

    ref: _Ref
    uuid: str
    texts: Mapping[_Annotation, str]
    shape: _Shape

    def check_copy(self, main: "_Series", *, values: bool) -> None:
        """Require this row's copy of a series to equal the series' own row.

        Args:
            main: The series' row in ``TS_Dataset.jsonl``.
            values: Whether the values must match too. The description files hold rounded copies, so
                only their texts must.

        Raises:
            TimeFValidationError: If the texts, the metadata or, when asked, the values differ.
        """
        differs = [key.value for key in main.texts if self.texts[key] != main.texts[key]]
        if values and self.shape != main.shape:
            differs.append(_SERIES_NAME)
        if differs:
            raise TimeFValidationError(
                f"{self.ref.context}: {differs} differ from the series' row, {_File.SERIES} line {main.ref.line}"
            )


class _FileReader:
    """Reads one file's rows in order and hashes its bytes, so a reread can tell the file changed."""

    def __init__(self, path: Path, file: _File) -> None:
        self.path = path
        self.file = file
        self._hash = hashlib.blake2b(digest_size=16)

    @property
    def digest(self) -> str:
        """The digest of the bytes read so far."""
        return self._hash.hexdigest()

    def rows(self) -> Iterator[_Row]:
        """Decode every line, in file order.

        Yields:
            Each row.
        """
        with self.path.open("rb") as handle:
            line = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    return
                line += 1
                self._hash.update(raw)
                ref = _Ref(self.path, self.file, line, offset)
                yield _Row(ref, _decode(raw, ref))


def _reread_values(ref: _Ref, key: str, expected: _Shape) -> list[int | float]:
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
        values = _decode(raw, ref).get(key)
        shape = _shape_of(values)
    except TimeFValidationError as exc:
        raise TimeFFormatError(f"{ref.context}: {key}: {exc}; the file changed") from exc
    # _shape_of accepted only a list, but a type checker can't see that.
    if shape != expected or not isinstance(values, list):
        raise TimeFFormatError(f"{ref.context}: {key} differs from the converted row; the file changed")
    return values


def _load(ref: _Ref, key: str, shape: _Shape) -> pa.Array:
    # The Arrow type of the spec's dtype, as timenet's own Signal values derive it.
    arrow_type = pa.from_numpy_dtype(np.dtype(shape.dtype.spec.dtype))
    return pa.array(_reread_values(ref, key, shape), type=arrow_type)


def _record(uuid: str, ref: _Ref, key: str, shape: _Shape, texts: Mapping[_Annotation, str]) -> Record:
    """Build a Record whose one Signal rereads one series from one line.

    Returns:
        The Record, whose values stay in the source file until read.
    """
    signal = Signal.from_loader(
        name=_SERIES_NAME,
        spec=shape.dtype.spec,
        time_axis=_AXIS,
        n_values=shape.length,
        loader=partial(_load, ref, key, shape),
    )
    metadata: dict[str, JsonValue] = {"uuid": uuid, "file": ref.file.value, "line": ref.line}
    record = Record(sources=(Source(name=_SERIES_NAME, signals=(signal,)),), metadata=metadata)
    for annotation_key, text in texts.items():
        record.annotate(Annotation(key=annotation_key, value=text))
    return record


@dataclass(frozen=True)
class _Changed:
    """The changed series of one counterfactual, as its first row publishes it."""

    ref: _Ref
    new_generator: str
    shape: _Shape
    record: Record


@dataclass
class _Release:
    """The release after pass one: every Record, and what pass two needs to build the tasks."""

    paths: Mapping[_File, Path]
    digests: dict[_File, str] = field(default_factory=dict)
    """The digest of each file's bytes, which pass two checks."""
    series: dict[str, tuple[_Series, Record]] = field(default_factory=dict)
    """One entry per row of ``TS_Dataset.jsonl``, by uuid, in file order."""
    copies: dict[tuple[str, str], Record] = field(default_factory=dict)
    """One Record per distinct series in the description files that differs from its series' row, by
    uuid and value digest."""
    changed: dict[tuple[str, str], _Changed] = field(default_factory=dict)
    """One entry per changed series, by uuid and counterfactual, in file order."""

    def read(self) -> "_Release":
        """Read each file once, in :class:`_File` order, and check that the files agree.

        Returns:
            This release, with every Record built.

        Raises:
            TimeFValidationError: If a row is malformed, a uuid repeats or is unknown, a row's copy of
                its series differs from ``TS_Dataset.jsonl``, or a counterfactual holds two changed series.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
        self._read_series()
        for file in _DESCRIPTION_FILES:
            self._read_descriptions(file)
        self._read_questions()
        self._read_counterfactuals()
        return self

    def _reader(self, file: _File) -> _FileReader:
        return _FileReader(self.paths[file], file)

    def _read_series(self) -> None:
        reader = self._reader(_File.SERIES)
        for row in reader.rows():
            series = row.series()
            if series.uuid in self.series:
                first = self.series[series.uuid][0].ref.line
                raise TimeFValidationError(f"{row.ref.context}: uuid {series.uuid} repeats line {first}")
            record = _record(series.uuid, series.ref, _SERIES_NAME, series.shape, series.texts)
            self.series[series.uuid] = (series, record)
        self.digests[_File.SERIES] = reader.digest

    def _main(self, copy: _Series) -> _Series:
        """Return the series' row that a row's copy repeats.

        Raises:
            TimeFValidationError: If the uuid is not in ``TS_Dataset.jsonl``.
        """
        if copy.uuid not in self.series:
            raise TimeFValidationError(f"{copy.ref.context}: uuid {copy.uuid} is not in {_File.SERIES}")
        return self.series[copy.uuid][0]

    def _read_descriptions(self, file: _File) -> None:
        reader = self._reader(file)
        for row in reader.rows():
            copy = _read_description(row).series
            main = self._main(copy)
            copy.check_copy(main, values=False)
            if copy.shape.length != main.shape.length:
                raise TimeFValidationError(
                    f"{copy.ref.context}: a rounded copy has {copy.shape.length} values, "
                    f"its series' row {main.shape.length}"
                )
            key = (copy.uuid, copy.shape.digest)
            # The authors' description task reads this file's values, so a rounded copy is a Record of its own.
            if copy.shape != main.shape and key not in self.copies:
                self.copies[key] = _record(copy.uuid, copy.ref, _SERIES_NAME, copy.shape, copy.texts)
        self.digests[file] = reader.digest

    def _read_questions(self) -> None:
        reader = self._reader(_File.QUESTIONS)
        for row in reader.rows():
            question = _read_question(row)
            question.series.check_copy(self._main(question.series), values=True)
        self.digests[_File.QUESTIONS] = reader.digest

    def _read_counterfactuals(self) -> None:
        reader = self._reader(_File.COUNTERFACTUALS)
        for row in reader.rows():
            counterfactual = _read_counterfactual(row)
            counterfactual.question.series.check_copy(self._main(counterfactual.question.series), values=True)
            self._add_changed(counterfactual)
        self.digests[_File.COUNTERFACTUALS] = reader.digest

    def _add_changed(self, row: "_Counterfactual") -> None:
        """Keep the first row of a counterfactual, or check a later row's changed series against it.

        Raises:
            TimeFValidationError: If the rows of one counterfactual hold different changed series.
        """
        ref, uuid = row.question.series.ref, row.question.series.uuid
        key = (uuid, row.counterfactual)
        known = self.changed.get(key)
        if known is None:
            texts = {_Annotation.COUNTERFACTUAL: row.counterfactual, _Annotation.GENERATOR: row.new_generator}
            record = _record(uuid, ref, "new_series", row.new_shape, texts)
            self.changed[key] = _Changed(ref, row.new_generator, row.new_shape, record)
        elif (known.new_generator, known.shape) != (row.new_generator, row.new_shape):
            raise TimeFValidationError(
                f"{ref.context}: the changed series differs from line {known.ref.line}'s for the same counterfactual"
            )

    def records(self) -> Iterator[Record]:
        """Return the series Records, the rounded copies, then the changed series, each in source order.

        Yields:
            Each Record.
        """
        for _, record in self.series.values():
            yield record
        yield from self.copies.values()
        for changed in self.changed.values():
            yield changed.record

    def tasks(self) -> Iterator[AnswerTask]:
        """Return every task by rereading the five task files: description, question, counterfactual.

        Each call rereads them, so the tasks can be iterated more than once. The files must stay
        available until writing ends.

        Yields:
            The description tasks, then the question tasks, then the counterfactual tasks.
        """
        for file in _DESCRIPTION_FILES:
            yield from self._stream(file, partial(self._from_description_row, split=_SPLITS[file]))
        yield from self._stream(_File.QUESTIONS, self._from_question_row)
        yield from self._stream(_File.COUNTERFACTUALS, self._from_counterfactual_row)

    def _stream(self, file: _File, build: Callable[[_Row], AnswerTask]) -> Iterator[AnswerTask]:
        """Build a task from each row of a file read again, and check at its end that the file is unchanged.

        Yields:
            The task of each row.

        Raises:
            TimeFFormatError: If the file's bytes differ from pass one's, or a row no longer reads
                or names a series that pass one did not find.
        """
        reader = self._reader(file)
        try:
            for row in reader.rows():
                yield build(row)
        except (TimeFValidationError, KeyError) as exc:
            raise TimeFFormatError(f"{reader.path}: {exc}; the file changed since conversion") from exc
        if reader.digest != self.digests[file]:
            raise TimeFFormatError(f"{reader.path}: the file changed since conversion")

    def _from_description_row(self, row: _Row, split: Split) -> AnswerTask:
        description = _read_description(row)
        copy = self.copies.get((description.series.uuid, description.series.shape.digest))
        record = self.series[description.series.uuid][1] if copy is None else copy
        return _description_task(description, record, split)

    def _from_question_row(self, row: _Row) -> AnswerTask:
        question = _read_question(row)
        return _question_task(question, (self.series[question.series.uuid][1],))

    def _from_counterfactual_row(self, row: _Row) -> AnswerTask:
        counterfactual = _read_counterfactual(row)
        uuid = counterfactual.question.series.uuid
        changed = self.changed[uuid, counterfactual.counterfactual].record
        return _question_task(counterfactual.question, (self.series[uuid][1], changed), counterfactual.category)


@dataclass(frozen=True)
class _Description:
    """A row of ``TS_Dataset_MCQ/{train,val,test}.json``: pick the series' description among four."""

    series: _Series
    options: list[str]
    """The published options, with the series' description first."""
    scenario_category: _ScenarioCategory

    def shuffled_options(self) -> list[str]:
        """Shuffle the options once, with a seed of the row's own.

        The authors' loader shuffles them with ``random.shuffle`` and no seed, so each load shows its own
        order, while the published order puts the answer first. A stored prompt needs one order, so this
        seeds ``random.Random`` with the row's file and line: every build gives the row the same order.

        Returns:
            The options in the order the task's prompt shows them.
        """
        options = list(self.options)
        ref = self.series.ref
        random.Random(f"{ref.file.value}:{ref.line}").shuffle(options)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - a shuffle, not cryptography
        return options


def _read_description(row: _Row) -> _Description:
    """Validate a description row.

    The authors' ``ts2desc_mcq.yaml`` labels a row by its ``description``, so the answer is the option
    equal to it, and the release lists it first in every row.

    Returns:
        The row.

    Raises:
        TimeFValidationError: If the row is malformed, or its first option is not its description.
    """
    series = row.series()
    options = row.options(allow_blank=False)
    if options[0] != series.texts[_Annotation.DESCRIPTION]:
        raise TimeFValidationError(f"{row.ref.context}: the first option is not the row's description")
    return _Description(series, options, row.category("scenario_category", _ScenarioCategory))


@dataclass(frozen=True)
class _Question:
    """A row of ``MCQ_1_TS.jsonl``: a question about one series, with four options.

    Some published questions and options are blank, and some options repeat. They are kept as published,
    and :attr:`issues` names them.
    """

    series: _Series
    question: str
    options: list[str]
    answer_index: int
    """The position of the right option, which the authors' loader formats as a letter."""

    @property
    def answer_text(self) -> str:
        """The right option, the text its target letter stands for."""
        return self.options[self.answer_index]

    @property
    def issues(self) -> list[_SourceIssue]:
        """The row's issues, in :class:`_SourceIssue` order, or none."""
        found = {
            _SourceIssue.BLANK_QUESTION: not self.question.strip(),
            _SourceIssue.BLANK_OPTION: any(not option.strip() for option in self.options),
            _SourceIssue.REPEATED_OPTION: len(set(self.options)) < len(self.options),
            _SourceIssue.ANSWER_TEXT_REPEATED: self.options.count(self.answer_text) > 1,
        }
        return [issue for issue in _SourceIssue if found[issue]]


def _read_question(row: _Row) -> _Question:
    """Validate a question row. Its question and options may be blank, as published.

    Returns:
        The row.

    Raises:
        TimeFValidationError: If the row is malformed.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the row's getters
    return _Question(
        row.series(),
        row.text("question", allow_blank=True),
        row.options(allow_blank=True),
        row.answer_index(),
    )


@dataclass(frozen=True)
class _Counterfactual:
    """A row of ``MCQ_2_TS.jsonl``: a question that compares a series with a changed version of it."""

    question: _Question
    category: _QuestionCategory
    counterfactual: str
    """The change, as a question: "What if ...?". The authors' prompt does not include it."""
    new_generator: str
    new_shape: _Shape
    """The changed series' shape. Its values stay in the file."""


def _read_counterfactual(row: _Row) -> _Counterfactual:
    """Validate a counterfactual row.

    Returns:
        The row.

    Raises:
        TimeFValidationError: If the row is malformed.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the row's getters
    return _Counterfactual(
        _read_question(row),
        row.category("category", _QuestionCategory),
        row.text("counterfactual"),
        row.text("new_generator"),
        row.shape("new_series"),
    )


def _prompt(prefix: str, columns: tuple[str, ...], options: list[str]) -> str:
    # The prefix and the first column meet without a separator, as in the authors' loader.
    lettered = "\n".join(f"{letter}) {option}" for letter, option in zip(_LETTERS, options, strict=True))
    return prefix + "\n".join(columns) + "\n" + lettered


def _provenance(ref: _Ref) -> dict[str, JsonValue]:
    return {_TaskMeta.FILE: ref.file.value, _TaskMeta.LINE: ref.line}


def _description_task(row: _Description, record: Record, split: Split) -> AnswerTask:
    """Return the task that picks the series' description among four (the authors' ``ts2desc_mcq``).

    Returns:
        The task, its options in the row's seeded order and its target the description's letter.
    """
    options = row.shuffled_options()
    letter = _LETTERS[options.index(row.series.texts[_Annotation.DESCRIPTION])]
    metadata = _provenance(row.series.ref) | {_TaskMeta.SCENARIO_CATEGORY: row.scenario_category.value}
    return AnswerTask(
        inputs=(record,),
        prompt=_prompt(_DESCRIPTION_PREFIX, (), options),
        targets=(letter,),
        split=split,
        metadata=metadata,
    )


def _question_task(
    row: _Question, records: tuple[Record, ...], category: _QuestionCategory | None = None
) -> AnswerTask:
    """Return the task that answers a question (the authors' ``qa_mcq`` or ``qa_counterfactual``).

    The release gives these tasks no split. The options keep their published order, and the target is
    the letter at ``answer_index``.

    Args:
        row: The question row.
        records: The input Records: the series, then for a counterfactual its changed version, in the
            order the authors' loader joins them.
        category: A counterfactual question's published category, ``None`` for a one-series question.

    Returns:
        The task.
    """
    texts = row.series.texts
    prompt = _prompt(
        _QUESTION_PREFIX, (texts[_Annotation.DESCRIPTION_TINY], texts[_Annotation.METADATA], row.question), row.options
    )
    issues: list[JsonValue] = [issue.value for issue in row.issues]
    metadata = _provenance(row.series.ref) | {
        _TaskMeta.ANSWER_INDEX: row.answer_index,
        _TaskMeta.ANSWER_TEXT: row.answer_text,
        _TaskMeta.SOURCE_ISSUES: issues,
    }
    if category is not None:
        metadata[_TaskMeta.CATEGORY] = category.value
    return AnswerTask(inputs=records, prompt=prompt, targets=(_LETTERS[row.answer_index],), metadata=metadata)


class TimeSeriesReasoningConnector(BaseHuggingFaceConnector):
    """Build the dataset from the six files of its pinned Hugging Face release."""

    HF_REPO = "mikeam/time-series-reasoning"
    SOURCE_REVISION = "d6a1c7d48d8a454b9db0b2137383fc30847e6ec0"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {file.part: file.value for file in _File}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the six files into lazy Records and a stream of one task per task row.

        Pass one reads each file once and checks every row. The tasks reread the five task files, so
        the files must stay available until writing ends.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.

        Raises:
            TimeFValidationError: If a file is missing, a row is malformed, or a row's copy of its
                series differs from ``TS_Dataset.jsonl``.
        """
        by_part = {ref.part: ref.path for ref in raw_refs}
        missing = [file.value for file in _File if file.part not in by_part]
        if missing:
            raise TimeFValidationError(f"the release needs each of its files, missing {missing}")
        release = _Release({file: by_part[file.part] for file in _File}).read()
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (_Annotation.SOURCE_REPOSITORY, self.HF_REPO),
            (_Annotation.SOURCE_REVISION, self.SOURCE_REVISION),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        for record in release.records():
            dataset.add_record(record=record)
        dataset.set_task_stream((AnswerTask,), release.tasks)
        return dataset


CONNECTOR = TimeSeriesReasoningConnector
