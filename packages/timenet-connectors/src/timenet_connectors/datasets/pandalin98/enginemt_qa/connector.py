"""Convert the pinned EngineMT-QA files into lazy Records and one task per QA pair.

Each HDF5 row that a QA pair names becomes a Record whose 33 signals read the row on demand. Each QA
pair that has a label becomes one task of its stage. README.md explains how this follows the authors.

Metadata is provenance and never model input.
"""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum, unique
from functools import lru_cache, partial
import hashlib
import json
import logging
from pathlib import Path
import re
import string
from typing import Any, ClassVar, NamedTuple, TypedDict, cast

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    Label,
    LabelSet,
    Split,
    Task,
    TimeSeriesSpec,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef


_LOGGER = logging.getLogger(__name__)

_N_STEPS = 600
_N_CHANNELS = 33
"""The released channel count. The paper says 32. The authors' audit confirms 33 and names none of them."""
_MULTI_CYCLE_ROWS = 10
"""How many rows a multi-cycle QA names."""
_STEPS_PER_CYCLE = _N_STEPS // _MULTI_CYCLE_ROWS
"""The steps of each cycle that the authors' model reads in a multi-cycle task: the first 60 of 600."""
_SAMPLE_RATE_HZ = 1
"""The paper's "10-minute (600 points) time-series segments" (Section 5.2). The HDF5 file states no rate."""
_ROW_SHAPE = (_N_STEPS, _N_CHANNELS)
_SEQ_DATA = "seq_data"
"""The HDF5 dataset of every row's values, shaped ``(rows, steps, channels)``."""
_DATA_ID = "data_ID"
"""The HDF5 dataset of each row's id, which a QA's ``id`` names. Row ``i`` has id ``i + 1``."""
_SOURCE_NAME = "N-CMAPSS engine"
_CHANNELS = tuple(f"channel_{index:02d}" for index in range(_N_CHANNELS))
"""The release names no channel. The zero-padded names sort in the source's order."""
# In 24 sampled rows, every channel has a mean near 0 and a standard deviation near 1, so the values
# look z-scored. The release states neither the units nor the statistics.
_SPEC = TimeSeriesSpec(
    spec_type="enginemt_qa_channel", name="EngineMT-QA channel, z-scored", unit_value=None, dtype="float64"
)
_AXIS = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ)
# A published row id: a decimal number with no sign and no leading zero, such as "278".
_ROW_ID = re.compile(r"[1-9][0-9]*")
# One printed option of a closed question: its letter, then its text, such as "a: Health\n".
_OPTION = re.compile(r"([a-z]):\s+([^\n]+)\n")
_DIGEST_BYTES = 16
"""The size of a line or pair digest. 16 bytes make an accidental match between two of the release's
144,042 pairs practically impossible, and keep the digests of a file small."""


@unique
class _Key(StrEnum):
    """The keys of the annotations and the metadata that the connector writes."""

    ROW_ID = "row_id"
    SOURCE_FILE = "source_file"
    SOURCE_LINE = "source_line"
    QA_OFFSET = "qa_offset"
    STAGE = "stage"
    STEPS_PER_CYCLE = "steps_per_cycle"
    CLOSED_QUESTION = "closed_question"
    SOURCE_ISSUES = "source_issues"
    UNLABELLED_PAIRS = "unlabelled_pairs"


@unique
class SourceIssue(StrEnum):
    """A problem in a published pair, which the connector keeps as published and names in the task's metadata.

    The names match the TSAQA connector's, so one filter can serve both datasets.
    """

    REPEATED_QUESTION = "repeated_question"
    """Another pair of the same file has the same rows, question and answer."""
    REPEATED_ACROSS_SPLITS = "repeated_across_splits"
    """The other file has a pair with the same rows, question and answer."""


@unique
class _Part(StrEnum):
    """The release's three files: the signals, and the train and test QA."""

    SIGNALS = "signals"
    TRAIN = "train"
    TEST = "test"

    @property
    def source_file(self) -> str:
        """The file's name in the Hub repository, such as ``train_qa.jsonl``."""
        return "time_series_data.h5" if self is _Part.SIGNALS else f"{self.value}_qa.jsonl"

    @property
    def task_split(self) -> Split:
        """The split of a QA file: the authors train on ``train_qa`` and test on ``test_qa``.

        Raises:
            TimeFValidationError: If the part is not a QA file.
        """
        if self not in _SPLITS:
            raise TimeFValidationError(f"{self} is not a QA file")
        return _SPLITS[self]


_SPLITS = {_Part.TRAIN: Split.TRAIN, _Part.TEST: Split.TEST}
_QA_PARTS = tuple(_SPLITS)


@unique
class _Stage(IntEnum):
    """The release's ``stage``: one of the paper's four tasks."""

    UNDERSTANDING = 1
    PERCEPTION = 2
    REASONING = 3
    DECISION_MAKING = 4

    @property
    def is_closed(self) -> bool:
        """Whether the stage asks closed questions, which the release marks ``close``."""
        return self in {_Stage.PERCEPTION, _Stage.REASONING}

    @property
    def is_multi_cycle(self) -> bool:
        """Whether the stage's questions name 10 cycles rather than one."""
        return self in {_Stage.REASONING, _Stage.DECISION_MAKING}


@unique
class _Attribute(StrEnum):
    """The release's ``attribute`` of a question."""

    OPEN = "open"
    CLOSE = "close"


@unique
class _Role(StrEnum):
    """The ``from`` of a turn: a question comes from ``human`` and its answer from ``gpt``."""

    HUMAN = "human"
    GPT = "gpt"


def _option_letters(name: str, options: Sequence[str], *, multi_label: bool = False) -> LabelSet:
    """Build the label set of a closed question: one label per option letter, from ``a``.

    Returns:
        The label set. Each label's name is a letter, and its description is the option the letter
        stands for.
    """
    labels = tuple(
        Label(name=letter, description=option)
        for letter, option in zip(string.ascii_lowercase[: len(options)], options, strict=True)
    )
    return LabelSet(name=name, labels=labels, multi_label=multi_label)


_COMPONENT_HEALTH = _option_letters("component_health", ("Health", "Fault"))
_FAULT_CAUSE = _option_letters(
    "fault_cause",
    (
        "HPT efficiency modifier",
        "LPT efficiency modifier",
        "Fan efficiency modifier",
        "HPT flow modifier",
        "LPT flow modifier",
        "Fan flow modifier",
    ),
    # The authors' answers list every cause that applies, such as "a, b, e".
    multi_label=True,
)
_HEALTH_STATE = _option_letters("health_state", ("Operating Normally", "Abnormal Degradation", "Imminent Failure"))
_ENGINE_CONDITION = _option_letters(
    "engine_condition",
    ("Good Condition", "Moderate Condition", "Poor Condition", "Bad Condition", "Extremely Poor Condition"),
)
_FAILURE_PROBABILITY = _option_letters("failure_probability", ("1%-10%", "10%-30%", "30%-50%", "50%-70%", "70%-100%"))


class _ClosedSpec(NamedTuple):
    stage: _Stage
    label_set: LabelSet
    stem: str
    """The question's text up to its ``?``. The release publishes each closed question with one stem."""


_REASONING_STEM = (
    "Given the time series signal <ts>, by perceiving the engine signal across 10 cycles to reflect different "
    "health states and performing temporal inference to predict the health decline trend, "
)
_ONE_CYCLE = "of the given engine signal in one cycle"


def _prints_options(options: str, label_set: LabelSet) -> bool:
    """Tell whether the printed options of a question are the label set's, in letter order.

    Returns:
        ``True`` if each printed letter and text is a label's name and description, in label order.
    """
    printed = _OPTION.findall(options)
    return printed == [(label.name, label.description) for label in label_set.labels]


@unique
class _ClosedQuestion(StrEnum):
    """A closed question, which the release always publishes with the same stem and options."""

    FAN_HEALTH = "fan_health"
    LPC_HEALTH = "lpc_health"
    HPC_HEALTH = "hpc_health"
    HPT_HEALTH = "hpt_health"
    LPT_HEALTH = "lpt_health"
    FAULT_CAUSE = "fault_cause"
    HEALTH_STATE = "health_state"
    ENGINE_CONDITION = "engine_condition"
    FAILURE_PROBABILITY = "failure_probability"

    @property
    def stage(self) -> _Stage:
        """The question's stage."""
        return _CLOSED[self].stage

    @property
    def label_set(self) -> LabelSet:
        """The question's option letters."""
        return _CLOSED[self].label_set

    @classmethod
    def of(cls, stage: _Stage, text: str) -> "_ClosedQuestion":
        """Identify a published closed question by its stem, and check that it prints its label set's options.

        Args:
            stage: The stage that the release gives the question.
            text: The published question.

        Returns:
            The question.

        Raises:
            ValueError: If no closed question of the stage has this stem, or its printed options differ from
                the label set's, which would change what each letter means.
        """
        stem, _, options = text.partition("?")
        question = _BY_STEM.get(stem)
        if question is None or question.stage is not stage:
            raise ValueError(f"unknown stage {stage} closed question {text[:80]!r}")
        if not _prints_options(options, question.label_set):
            raise ValueError(f"{question} prints options other than its label set's: {options!r}")
        return question

    def parse_answer(self, answer: str) -> tuple[str, ...]:
        """Parse a published answer, such as ``a`` or ``a, b, e``, into its letters.

        Returns:
            The letters, in published order.

        Raises:
            ValueError: If the answer isn't distinct letters of the question's options, joined by ``", "``,
                or holds several letters where one is expected.
        """
        letters = tuple(answer.split(", "))
        try:
            self.label_set.check_targets(letters)
        except TimeFValidationError as exc:
            raise ValueError(f"{self}: answer {answer!r}: {exc}") from exc
        return letters


def _health_spec(component: str) -> _ClosedSpec:
    return _ClosedSpec(_Stage.PERCEPTION, _COMPONENT_HEALTH, f"What is the health status of {component} {_ONE_CYCLE}")


_CLOSED = {
    _ClosedQuestion.FAN_HEALTH: _health_spec("fan"),
    _ClosedQuestion.LPC_HEALTH: _health_spec("LPC"),
    _ClosedQuestion.HPC_HEALTH: _health_spec("HPC"),
    _ClosedQuestion.HPT_HEALTH: _health_spec("HPT"),
    _ClosedQuestion.LPT_HEALTH: _health_spec("LPT"),
    _ClosedQuestion.FAULT_CAUSE: _ClosedSpec(
        _Stage.PERCEPTION,
        _FAULT_CAUSE,
        f"Was the timing issue in the operation {_ONE_CYCLE} caused by efficiency or flow problems",
    ),
    _ClosedQuestion.HEALTH_STATE: _ClosedSpec(
        _Stage.PERCEPTION, _HEALTH_STATE, f"What is the potential health state {_ONE_CYCLE}"
    ),
    _ClosedQuestion.ENGINE_CONDITION: _ClosedSpec(
        _Stage.REASONING, _ENGINE_CONDITION, _REASONING_STEM + "what is the qualitative condition of the engine"
    ),
    _ClosedQuestion.FAILURE_PROBABILITY: _ClosedSpec(
        _Stage.REASONING, _FAILURE_PROBABILITY, _REASONING_STEM + "what is the probability range of engine failure"
    ),
}
_BY_STEM = {spec.stem: question for question, spec in _CLOSED.items()}


class _FileCounts(NamedTuple):
    """The pinned size of a QA file, which conversion checks so that a short or changed file fails."""

    lines: int
    pairs: Mapping[_Stage, int]
    """The pairs of each stage, empty answers included."""
    unlabelled: Mapping[_ClosedQuestion, int]
    """The closed pairs with an empty answer, for each question that has any."""


class _Place(NamedTuple):
    """Where a pair was published."""

    part: _Part
    line: int
    """The 0-based line in the file, which the authors call ``source_line``."""
    offset: int
    """The 0-based place in the line, which the authors call ``qa_offset``."""

    @property
    def metadata(self) -> dict[str, JsonValue]:
        """The place as the authors' evaluation names it."""
        return {_Key.SOURCE_FILE: self.part.source_file, _Key.SOURCE_LINE: self.line, _Key.QA_OFFSET: self.offset}


@dataclass(frozen=True)
class _Pair:
    """One published question with its answer, and the rows it is about."""

    place: _Place
    rows: tuple[int, ...]
    """The row ids in published order: one for a single-cycle question and ten for a multi-cycle one."""
    stage: _Stage
    question: str
    """The question as published, with its ``<ts>`` marker."""
    answer: str
    """The answer as published: free text, or letters for a closed question."""
    closed: _ClosedQuestion | None
    """The closed question, or ``None`` for an open one."""

    @property
    def missing_label(self) -> _ClosedQuestion | None:
        """The closed question, if its published answer is empty. Such a pair is a missing label and has no task."""
        return None if self.answer else self.closed

    @property
    def has_label(self) -> bool:
        """Whether the pair has a task."""
        return self.missing_label is None

    @property
    def digest(self) -> bytes:
        """A digest of the rows, the question and the answer, which match for a repeated pair."""
        content = json.dumps([self.rows, self.question, self.answer]).encode()
        return hashlib.blake2b(content, digest_size=_DIGEST_BYTES).digest()

    def inputs(self, records: Mapping[int, Record]) -> tuple[Record, ...]:
        """Return the Records of the pair's rows, in published order.

        Raises:
            TimeFFormatError: If a row has no Record, as when a QA file changed since conversion.
        """
        missing = [row for row in self.rows if row not in records]
        if missing:
            raise TimeFFormatError(
                f"{self.place.part.source_file} line {self.place.line + 1}: no Record for rows {missing}"
            )
        return tuple(records[row] for row in self.rows)

    def task(self, inputs: tuple[Record, ...], issues: Sequence[SourceIssue]) -> AnswerTask | ClassificationTask:
        """Build the task of a pair that has a label.

        Args:
            inputs: The Records of the pair's rows, in published order.
            issues: The pair's source issues, in ``SourceIssue`` order.

        Returns:
            An answer task for an open question, with the published answer as target. A classification task
            for a closed question, with the published letters as targets over its label set.
        """
        issue_names: list[JsonValue] = [issue.value for issue in issues]
        metadata = self.place.metadata | {_Key.STAGE: self.stage.value, _Key.SOURCE_ISSUES: issue_names}
        if self.stage.is_multi_cycle:
            metadata[_Key.STEPS_PER_CYCLE] = _STEPS_PER_CYCLE
        split = self.place.part.task_split
        if self.closed is None:
            return AnswerTask(
                inputs=inputs, prompt=self.question, targets=(self.answer,), split=split, metadata=metadata
            )
        return ClassificationTask(
            inputs=inputs,
            prompt=self.question,
            targets=self.closed.parse_answer(self.answer),
            label_set=self.closed.label_set,
            split=split,
            metadata=metadata | {_Key.CLOSED_QUESTION: self.closed.value},
        )


# The published JSON shapes. A value is ``object`` until the parser checks it. The turns use the
# functional form because their key ``from`` is a Python keyword.
class _Line(TypedDict):
    id: object
    name: object
    conversations: object


_Question = TypedDict("_Question", {"from": object, "stage": object, "attribute": object, "value": object})
_Answer = TypedDict("_Answer", {"from": object, "value": object})


def _require_keys(value: object, keys: frozenset[str]) -> dict[str, object]:
    """Check that a JSON value is an object with exactly these keys.

    Returns:
        The object.

    Raises:
        ValueError: If the value is not an object, or lacks a key or has another.
    """
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"expected an object with the keys {sorted(keys)}, got {value!r}")
    return value


def _parse_line(raw: bytes) -> _Line:
    """Parse one JSON line and check its keys.

    Returns:
        The object, typed by its keys. ``_require_keys`` raises ``ValueError`` if they differ.
    """
    return cast(_Line, _require_keys(json.loads(raw), _Line.__required_keys__))


def _parse_question(value: object) -> _Question:
    """Check the keys of a question turn.

    Returns:
        The object, typed by its keys. ``_require_keys`` raises ``ValueError`` if they differ.
    """
    return cast(_Question, _require_keys(value, _Question.__required_keys__))


def _parse_answer(value: object) -> _Answer:
    """Check the keys of an answer turn.

    Returns:
        The object, typed by its keys. ``_require_keys`` raises ``ValueError`` if they differ.
    """
    return cast(_Answer, _require_keys(value, _Answer.__required_keys__))


def _require_text(value: object, *, allow_blank: bool = False) -> str:
    """Check that a JSON value is text.

    Returns:
        The text.

    Raises:
        ValueError: If the value is not text, or is blank and ``allow_blank`` is false.
    """
    if not isinstance(value, str) or not (allow_blank or value.strip()):
        raise ValueError(f"expected {'text' if allow_blank else 'nonblank text'}, got {value!r}")
    return value


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def _row_ids(line: _Line, n_rows: int) -> tuple[int, ...]:
    """Read the row ids of a line, and check that its ``name`` repeats them.

    Returns:
        One id for a single-cycle line, ten for a multi-cycle line, in published order.

    Raises:
        ValueError: If an id is not a row of the file, or the ids repeat or are not one or ten.
    """
    raw_ids, raw_names = line["id"], line["name"]
    is_list = isinstance(raw_ids, list)
    ids = _as_list(raw_ids)
    names_match = isinstance(raw_names, list) == is_list and _as_list(raw_names) == [f"{i}.csv" for i in ids]
    if not names_match:
        raise ValueError(f"name {raw_names!r} doesn't repeat id {raw_ids!r}")
    rows: list[int] = []
    for i in ids:
        if not isinstance(i, str) or not _ROW_ID.fullmatch(i):
            raise ValueError(f"id {raw_ids!r} is not decimal row ids")
        rows.append(int(i))
    if is_list and len(rows) != _MULTI_CYCLE_ROWS:
        raise ValueError(f"a multi-cycle id names {len(rows)} rows, not {_MULTI_CYCLE_ROWS}")
    if len(set(rows)) != len(rows):
        raise ValueError(f"id {raw_ids!r} repeats a row")
    if max(rows) > n_rows:
        raise ValueError(f"id {raw_ids!r} names a row past the file's {n_rows} rows")
    return tuple(rows)


def _parse_stage(value: object) -> _Stage:
    """Read a published stage, which is the text ``"1"`` to ``"4"``.

    Returns:
        The stage.

    Raises:
        ValueError: If the value is not one of the four stages.
    """
    if not isinstance(value, str) or value not in {str(stage.value) for stage in _Stage}:
        raise ValueError(f"bad stage {value!r}")
    return _Stage(int(value))


def _parse_attribute(value: object) -> _Attribute:
    """Read a published ``attribute``.

    Returns:
        The attribute.

    Raises:
        ValueError: If the value is not ``open`` or ``close``.
    """
    if not isinstance(value, str) or value not in set(_Attribute):
        raise ValueError(f"bad attribute {value!r}")
    return _Attribute(value)


def _parse_pair(place: _Place, rows: tuple[int, ...], question_turn: object, answer_turn: object) -> _Pair:
    """Check one question and its answer against its stage, and build the pair.

    Returns:
        The pair.

    Raises:
        ValueError: If the stage, the format and the number of rows disagree, a closed question is unknown, or
            an answer is malformed.
    """
    question, answer = _parse_question(question_turn), _parse_answer(answer_turn)
    if question["from"] != _Role.HUMAN or answer["from"] != _Role.GPT:
        raise ValueError("conversations must alternate a human question and a gpt answer")
    stage, attribute = _parse_stage(question["stage"]), _parse_attribute(question["attribute"])
    if (attribute is _Attribute.CLOSE) != stage.is_closed:
        raise ValueError(f"stage {stage} questions are not {attribute.value!r}")
    if (len(rows) > 1) != stage.is_multi_cycle:
        raise ValueError(f"a stage {stage} question names {len(rows)} rows")
    text, reply = _require_text(question["value"]), _require_text(answer["value"], allow_blank=True)
    if text.count("<ts>") != 1:
        raise ValueError("a question must hold one <ts> marker")
    closed = _ClosedQuestion.of(stage, text) if stage.is_closed else None
    if closed is None and not reply.strip():
        raise ValueError("an open question has an empty answer")
    if closed is not None and reply:
        # Parse here, so a malformed answer fails the read and not the task stream.
        closed.parse_answer(reply)
    return _Pair(place, rows, stage, text, reply, closed)


@dataclass
class _QAFile:
    """One QA file. It is read twice, and the second read must see the file the first one did."""

    path: Path
    part: _Part
    n_rows: int
    line_digests: list[bytes] | None = field(default=None, init=False)
    """A digest of each line, kept by the first complete read."""

    def pairs(self) -> Iterator[_Pair]:
        """Read every line of the file and flatten it into pairs, in file order.

        The first complete read keeps a digest of each line. A later read checks each line against its
        digest before it parses the line.

        Yields:
            Each published pair, empty answers included.

        Raises:
            TimeFValidationError: If a line or a pair is malformed, with its file, line and pair.
            TimeFFormatError: If the file cannot be read, or changed since the first complete read.
        """  # ruff: ignore[docstring-extraneous-exception] - TimeFValidationError is raised by _line_pairs
        expected = self.line_digests
        digests: list[bytes] = []
        try:
            with self.path.open("rb") as lines:
                for number, raw in enumerate(lines):
                    digest = hashlib.blake2b(raw, digest_size=_DIGEST_BYTES).digest()
                    if expected is not None and (number >= len(expected) or digest != expected[number]):
                        raise TimeFFormatError(f"{self.path} line {number + 1}: the line changed since conversion")
                    digests.append(digest)
                    yield from self._line_pairs(number, raw)
        except OSError as exc:
            raise TimeFFormatError(f"{self.path}: cannot read the file: {exc}") from exc
        if expected is None:
            self.line_digests = digests
        elif len(digests) != len(expected):
            raise TimeFFormatError(f"{self.path}: lines were removed since conversion")

    def _line_pairs(self, number: int, raw: bytes) -> Iterator[_Pair]:
        where = f"{self.part.source_file} line {number + 1}"
        try:
            line = _parse_line(raw)
            rows = _row_ids(line, self.n_rows)
            turns = line["conversations"]
            if not isinstance(turns, list) or not turns or len(turns) % 2:
                raise ValueError("conversations must alternate a human question and a gpt answer")
            for offset in range(len(turns) // 2):
                where = f"{self.part.source_file} line {number + 1}, pair {offset}"
                place = _Place(self.part, number, offset)
                yield _parse_pair(place, rows, turns[2 * offset], turns[2 * offset + 1])
        except ValueError as exc:
            raise TimeFValidationError(f"{where}: {exc}") from exc


def _import_h5py() -> Any:
    """Import ``h5py`` when a file is read, so the connector can load without it.

    Returns:
        The ``h5py`` module.

    Raises:
        ImportError: If ``h5py``, declared in this connector's requirements, is not installed.
    """
    try:
        import h5py  # ruff: ignore[import-outside-top-level]
    except ImportError as exc:
        raise ImportError("reading EngineMT-QA needs h5py, declared in this connector's requirements.txt") from exc
    return h5py


def _open_signals(path: Path) -> Any:
    """Open the signals file for reading.

    Returns:
        The open ``h5py.File``, to use as a context manager.

    Raises:
        TimeFFormatError: If the file cannot be opened as HDF5, as with a truncated download.
    """
    h5py = _import_h5py()
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise TimeFFormatError(f"{path}: cannot open the signals file: {exc}") from exc


def _check_layout(path: Path) -> int:
    """Check that the signals file holds exactly ``seq_data`` and ``data_ID``, with the ids ``1..n``.

    Returns:
        The number of rows ``n``.

    Raises:
        TimeFValidationError: If the layout, the shape, the dtype or the ids differ from the published file's.
        TimeFFormatError: If the file cannot be opened or read as HDF5.
    """
    h5py = _import_h5py()
    with _open_signals(path) as file:
        if set(file) != {_SEQ_DATA, _DATA_ID}:
            raise TimeFValidationError(f"{path}: expected {_SEQ_DATA} and {_DATA_ID}, got {sorted(file)}")
        values, ids = file[_SEQ_DATA], file[_DATA_ID]
        for name, node in ((_SEQ_DATA, values), (_DATA_ID, ids)):
            if not isinstance(node, h5py.Dataset):
                raise TimeFValidationError(f"{path}: {name} is a {type(node).__name__}, not an HDF5 dataset")
        if values.shape[1:] != _ROW_SHAPE or values.dtype != np.float64:
            raise TimeFValidationError(f"{path}: {_SEQ_DATA} is {values.dtype} {values.shape}")
        n_rows = values.shape[0]
        if ids.shape != (n_rows,) or not np.issubdtype(ids.dtype, np.integer):
            raise TimeFValidationError(f"{path}: {_DATA_ID} is {ids.dtype} {ids.shape}, not {n_rows} integer ids")
        try:
            published_ids = ids[:]
        except OSError as exc:
            raise TimeFFormatError(f"{path}: cannot read {_DATA_ID}: {exc}") from exc
        if not np.array_equal(published_ids, np.arange(1, n_rows + 1)):
            raise TimeFValidationError(f"{path}: {_DATA_ID} is not the row ids 1 to {n_rows}")
    return n_rows


def _read_row(file: Any, path: Path, row_id: int) -> np.ndarray:
    """Read one row's values from an open signals file.

    Returns:
        The values, shaped ``(600, 33)``: steps, then channels.

    Raises:
        TimeFFormatError: If the file no longer holds the row, as when it changed since conversion, or a
            block of the file cannot be read.
        TimeFValidationError: If the row holds a NaN or an infinite value. z-scored values are finite.
    """
    try:
        values = np.asarray(file[_SEQ_DATA][row_id - 1], dtype=np.float64)
    except (KeyError, IndexError, ValueError, OSError) as exc:
        raise TimeFFormatError(f"{path}: cannot read row id {row_id}: {exc}") from exc
    if values.shape != _ROW_SHAPE:
        raise TimeFFormatError(f"{path}: row id {row_id} is shaped {values.shape}; the file changed")
    if not np.isfinite(values).all():
        raise TimeFValidationError(f"{path}: row id {row_id} holds a NaN or an infinite value")
    return values


class _RowReader:
    """Read one channel of a row when a signal asks for it.

    The writer loads a Record's signals one after another, so caching the last row reads each row once
    for all 33 channels.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._row = lru_cache(maxsize=1)(self._read)

    def _read(self, row_id: int) -> np.ndarray:
        with _open_signals(self._path) as file:
            return _read_row(file, self._path, row_id)

    def __call__(self, row_id: int, channel: int) -> pa.Array:
        """Return one channel of a row.

        Returns:
            The channel's 600 float64 values.
        """
        return pa.array(self._row(row_id)[:, channel], type=pa.float64())


def _record(row_id: int, reader: _RowReader) -> Record:
    """Build a row's Record: one Source of 33 lazy signals, and the row id as metadata.

    Returns:
        The Record, whose values stay in the HDF5 file until read.
    """
    signals = tuple(
        Signal.from_loader(
            name=name, spec=_SPEC, time_axis=_AXIS, n_values=_N_STEPS, loader=partial(reader, row_id, channel)
        )
        for channel, name in enumerate(_CHANNELS)
    )
    return Record(sources=(Source(name=_SOURCE_NAME, signals=signals),), metadata={_Key.ROW_ID: row_id})


@dataclass
class _Scan:
    """What one read of every QA pair finds."""

    named_rows: set[int] = field(default_factory=set)
    """The rows that the pairs with a label name."""
    unlabelled: list[_Place] = field(default_factory=list)
    """The places of the closed pairs with an empty answer."""
    digests: dict[_Part, Counter[bytes]] = field(default_factory=dict)
    """How often each labelled pair's digest occurs in each file."""

    def issues(self, pair: _Pair) -> list[SourceIssue]:
        """Name the source issues of a labelled pair.

        Returns:
            The issues, in ``SourceIssue`` order.
        """
        digest = pair.digest
        found = []
        if self.digests[pair.place.part][digest] > 1:
            found.append(SourceIssue.REPEATED_QUESTION)
        if any(self.digests[part][digest] for part in _QA_PARTS if part is not pair.place.part):
            found.append(SourceIssue.REPEATED_ACROSS_SPLITS)
        return found


def _scan(qa_files: Sequence[_QAFile], counts: Mapping[_Part, _FileCounts]) -> _Scan:
    """Check every pair once, and collect what the dataset needs from all of them.

    Returns:
        The rows that labelled pairs name, the unlabelled pairs, and the digests of the labelled pairs.

    Raises:
        TimeFValidationError: If a file's counts differ from the pinned ones.
    """
    scan = _Scan()
    for qa in qa_files:
        digests: Counter[bytes] = Counter()
        pairs: Counter[_Stage] = Counter()
        unlabelled: Counter[_ClosedQuestion] = Counter()
        for pair in qa.pairs():
            pairs[pair.stage] += 1
            question = pair.missing_label
            if question is None:
                scan.named_rows.update(pair.rows)
                digests[pair.digest] += 1
            else:
                scan.unlabelled.append(pair.place)
                unlabelled[question] += 1
        found = _FileCounts(len(qa.line_digests or ()), dict(pairs), dict(unlabelled))
        if found != counts[qa.part]:
            raise TimeFValidationError(f"{qa.path}: expected {counts[qa.part]}, found {found}")
        scan.digests[qa.part] = digests
    if scan.unlabelled:
        _LOGGER.warning("%d closed pairs have an empty answer and get no task", len(scan.unlabelled))
    return scan


def _unlabelled_annotation(places: Sequence[_Place]) -> Annotation:
    pairs: list[JsonValue] = [place.metadata for place in places]
    return Annotation(
        key=_Key.UNLABELLED_PAIRS,
        value=len(places),
        description="How many closed pairs have an empty published answer. They are missing labels and have no task.",
        metadata={"pairs": pairs},
    )


class EngineMTQAConnector(BaseHuggingFaceConnector):
    """Build EngineMT-QA from the pinned files of its Hugging Face release."""

    HF_REPO = "pandalin98/EngineMT-QA"
    SOURCE_REVISION = "ea2051a15512212976c22c1a075ec549c7d48eef"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {part.value: part.source_file for part in _Part}
    # Counted on the pinned files. They match the authors' README: 101,565 train and 42,477 test pairs.
    SOURCE_COUNTS: ClassVar[Mapping[_Part, _FileCounts]] = {
        _Part.TRAIN: _FileCounts(
            lines=13_291,
            pairs={
                _Stage.UNDERSTANDING: 41_128,
                _Stage.PERCEPTION: 35_987,
                _Stage.REASONING: 16_300,
                _Stage.DECISION_MAKING: 8_150,
            },
            unlabelled={_ClosedQuestion.FAULT_CAUSE: 1_054, _ClosedQuestion.HEALTH_STATE: 215},
        ),
        _Part.TEST: _FileCounts(
            lines=5_539,
            pairs={
                _Stage.UNDERSTANDING: 17_240,
                _Stage.PERCEPTION: 15_085,
                _Stage.REASONING: 6_768,
                _Stage.DECISION_MAKING: 3_384,
            },
            unlabelled={_ClosedQuestion.FAULT_CAUSE: 422, _ClosedQuestion.HEALTH_STATE: 78},
        ),
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Convert the files into one lazy Record per named row, and a stream of one task per labelled pair.

        Conversion checks the signals file's layout and every QA line once, and compares each QA file's
        counts with the pinned ones. The values stay in the HDF5 file until written, so the files must stay
        available until then. A closed pair with an empty answer is a missing label and gets no task.

        Args:
            raw_refs: The downloaded files, one per part.

        Returns:
            A dataset with the Records, the unlabelled pairs as an annotation, and a stream of every task.

        Raises:
            TimeFValidationError: If a part is missing, or a file is malformed or has other counts.
            TimeFFormatError: If a file cannot be read.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers it calls
        paths = {_Part(ref.part): ref.path for ref in raw_refs}
        if paths.keys() != set(_Part):
            raise TimeFValidationError(f"EngineMT-QA needs the parts {[part.value for part in _Part]}")
        signals_path = paths[_Part.SIGNALS]
        n_rows = _check_layout(signals_path)
        qa_files = [_QAFile(paths[part], part, n_rows) for part in _QA_PARTS]
        scan = _scan(qa_files, self.SOURCE_COUNTS)
        reader = _RowReader(signals_path)
        records = {row_id: _record(row_id, reader) for row_id in sorted(scan.named_rows)}

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(_unlabelled_annotation(scan.unlabelled))
        for record in records.values():
            dataset.add_record(record=record)

        def tasks() -> Iterator[Task]:
            for qa in qa_files:
                for pair in qa.pairs():
                    if pair.has_label:
                        yield pair.task(pair.inputs(records), scan.issues(pair))

        dataset.set_task_stream((AnswerTask, ClassificationTask), tasks)
        return dataset


CONNECTOR = EngineMTQAConnector
