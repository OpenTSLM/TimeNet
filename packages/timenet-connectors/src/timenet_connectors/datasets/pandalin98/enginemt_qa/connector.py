"""Convert the pinned EngineMT-QA files into lazy Records and one task per labelled QA pair.

A single-row question gets the Record of its row. A multi-row question gets one Record that joins the
first 60 steps of its 10 rows, as the authors' model reads them. README.md explains the choices.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum, unique
from functools import lru_cache, partial
import hashlib
import json
import logging
from pathlib import Path
import re
import string
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from timenet.dataset import OrdinalAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonValue
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    Label,
    LabelSet,
    MissingReason,
    MissingValues,
    Split,
    Task,
    TimeSeriesSpec,
)
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.sources.readers import DataRow, JsonlDataReader


_LOGGER = logging.getLogger(__name__)

_N_STEPS = 600
_N_CHANNELS = 33
"""The released channel count. The paper says 32, but the published file holds 33 and names none of them.

See docs/ISSUE_23_CHANNEL_AUDIT.md in the ITFormer repository.
"""
_MULTI_ROWS = 10
"""How many rows a multi-row question names."""
_STEPS_PER_ROW = _N_STEPS // _MULTI_ROWS
"""The steps of each row that the authors' code joins for a multi-row question: the first 60 of 600."""
_SAMPLE_RATE_HZ = 1
"""Consecutive rows continue each other, and the paper calls a row a "10-minute (600 points)" segment."""
_ROW_SHAPE = (_N_STEPS, _N_CHANNELS)
_SEQ_DATA = "seq_data"
"""The HDF5 dataset of every row's values, shaped ``(rows, steps, channels)``."""
_DATA_ID = "data_ID"
"""The HDF5 dataset of each row's id, which a QA's ``id`` names. Row ``i`` has id ``i + 1``."""
_SOURCE_NAME = "N-CMAPSS engine"
_CHANNELS = tuple(f"channel_{index:02d}" for index in range(_N_CHANNELS))
"""The release names no channel. The zero-padded names sort in the source's order."""
_ROW_SPEC = TimeSeriesSpec(
    spec_type="enginemt_qa_channel", name="EngineMT-QA channel, as published", unit_value=None, dtype="float64"
)
_JOINED_SPEC = TimeSeriesSpec(
    spec_type="enginemt_qa_joined_channel",
    name="EngineMT-QA channel, first 60 steps of 10 rows",
    unit_value=None,
    dtype="float64",
    missing=MissingValues(
        reason=MissingReason.ALIGNMENT_PADDING,
        description="The flight ended within the first 60 steps of a row, and the release filled the rest with zeros.",
    ),
)
_ROW_AXIS = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ)
_JOINED_AXIS = OrdinalAxis()
"""The 10 blocks come from different times, so the joined steps have an order but no clock."""
# A published row id: a decimal number with no sign and no leading zero, such as "278".
_ROW_ID = re.compile(r"[1-9][0-9]*")
# One printed option of a closed question: its letter, then its text, such as "a: Health\n".
_OPTION = re.compile(r"([a-z]):\s+([^\n]+)\n")
_DIGEST_BYTES = 16
"""The size of a pair digest, which makes an accidental match between two of the 144,042 pairs practically
impossible."""


@unique
class _Key(StrEnum):
    """The keys of the annotations and the metadata that the connector writes."""

    ROW_ID = "row_id"
    ROW_IDS = "row_ids"
    PADDED_STEPS = "padded_steps"
    STEPS_PER_ROW = "steps_per_row"
    SOURCE_FILE = "source_file"
    SOURCE_LINE = "source_line"
    QA_OFFSET = "qa_offset"
    STAGE = "stage"
    CLOSED_QUESTION = "closed_question"
    SOURCE_ISSUES = "source_issues"
    UNLABELLED_PAIRS = "unlabelled_pairs"


@unique
class SourceIssue(StrEnum):
    """A problem in a published pair, which the connector keeps and names in the task's metadata.

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
    def is_multi_row(self) -> bool:
        """Whether the stage's questions name 10 rows rather than one."""
        return self in {_Stage.REASONING, _Stage.DECISION_MAKING}


def _option_letters(name: str, options: Sequence[str], *, multi_label: bool = False) -> LabelSet:
    """Build the label set of a closed question: one label per option letter from ``a``, described by its option.

    Returns:
        The label set.
    """
    labels = [
        Label(name=letter, description=option)
        for letter, option in zip(string.ascii_lowercase[: len(options)], options, strict=True)
    ]
    return LabelSet(labels, multi_label=multi_label, annotations=[Annotation("name", value=name)])


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
    """Tell whether a question prints exactly the label set's letters and descriptions, in order.

    Returns:
        ``True`` if the printed options are the label set's.
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

        Returns:
            The question.

        Raises:
            ValueError: If no closed question of the stage has this stem, or it prints other options.
        """
        stem, _, options = text.partition("?")
        question = _BY_STEM.get(stem)
        if question is None or question.stage is not stage:
            raise ValueError(f"unknown stage {stage} closed question {text[:80]!r}")
        if not _prints_options(options, question.label_set):
            raise ValueError(f"{question} prints options other than its label set's: {options!r}")
        return question

    def parse_answer(self, answer: str) -> tuple[str, ...]:
        """Parse a published answer, such as ``a`` or ``a, b, e``, into its letters in published order.

        Returns:
            The letters.

        Raises:
            ValueError: If the answer is not distinct option letters joined by ``", "``, or has too many.
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
    """The row ids in published order: one for a single-row question and ten for a multi-row one."""
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

    def task(self, record: Record, issues: Sequence[SourceIssue]) -> AnswerTask | ClassificationTask:
        """Build the task of a labelled pair over the Record of its rows.

        Returns:
            An answer task with the published answer for an open question, or a classification task with the
            published letters for a closed one.
        """
        issue_names: list[JsonValue] = [issue.value for issue in issues]
        metadata = self.place.metadata | {_Key.STAGE: self.stage.value, _Key.SOURCE_ISSUES: issue_names}
        split = self.place.part.task_split
        if self.closed is None:
            return AnswerTask(
                inputs=(record,), prompt=self.question, targets=(self.answer,), split=split, metadata=metadata
            )
        return ClassificationTask(
            inputs=(record,),
            prompt=self.question,
            targets=self.closed.parse_answer(self.answer),
            label_set=self.closed.label_set,
            split=split,
            metadata=metadata | {_Key.CLOSED_QUESTION: self.closed.value},
        )


class _Turn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _QuestionTurn(_Turn):
    role: Literal["human"] = Field(alias="from")
    stage: Literal["1", "2", "3", "4"]
    attribute: Literal["open", "close"]
    value: str


class _AnswerTurn(_Turn):
    role: Literal["gpt"] = Field(alias="from")
    value: str


class _Line(DataRow):
    """One published QA line: the rows, their file names, and the alternating question and answer turns."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | list[str]
    name: str | list[str]
    conversations: list[_QuestionTurn | _AnswerTurn]


def _row_ids(line: _Line, n_rows: int) -> tuple[int, ...]:
    """Read the row ids of a line, and check that its ``name`` repeats them.

    Returns:
        One id for a single-row line, ten for a multi-row line, in published order.

    Raises:
        ValueError: If an id is not a row of the file, or the ids repeat or are not one or ten.
    """
    is_list = isinstance(line.id, list)
    ids = line.id if isinstance(line.id, list) else [line.id]
    names = line.name if isinstance(line.name, list) else [line.name]
    if isinstance(line.name, list) != is_list or names != [f"{i}.csv" for i in ids]:
        raise ValueError(f"name {line.name!r} doesn't repeat id {line.id!r}")
    if not all(_ROW_ID.fullmatch(i) for i in ids):
        raise ValueError(f"id {line.id!r} is not decimal row ids")
    rows = tuple(int(i) for i in ids)
    if is_list and len(rows) != _MULTI_ROWS:
        raise ValueError(f"a multi-row id names {len(rows)} rows, not {_MULTI_ROWS}")
    if len(set(rows)) != len(rows):
        raise ValueError(f"id {line.id!r} repeats a row")
    if max(rows) > n_rows:
        raise ValueError(f"id {line.id!r} names a row past the file's {n_rows} rows")
    return rows


def _parse_pair(place: _Place, rows: tuple[int, ...], question: object, answer: object) -> _Pair:
    """Check one question and its answer against its stage, and build the pair.

    Returns:
        The pair.

    Raises:
        ValueError: If the turns don't alternate, the stage, format and rows disagree, or an answer is malformed.
    """
    if not isinstance(question, _QuestionTurn) or not isinstance(answer, _AnswerTurn):
        raise ValueError("conversations must alternate a human question and a gpt answer")
    stage = _Stage(int(question.stage))
    if (question.attribute == "close") != stage.is_closed:
        raise ValueError(f"stage {stage} questions are not {question.attribute!r}")
    if (len(rows) > 1) != stage.is_multi_row:
        raise ValueError(f"a stage {stage} question names {len(rows)} rows")
    text, reply = question.value, answer.value
    if not text.strip() or text.count("<ts>") != 1:
        raise ValueError("a question must hold one <ts> marker")
    closed = _ClosedQuestion.of(stage, text) if stage.is_closed else None
    if closed is None and not reply.strip():
        raise ValueError("an open question has an empty answer")
    if closed is not None and reply:
        # Parse here, so a malformed answer fails the read and not the task stream.
        closed.parse_answer(reply)
    return _Pair(place, rows, stage, text, reply, closed)


@dataclass(frozen=True)
class _QAFile:
    """One QA file and the number of rows its ids may name."""

    path: Path
    part: _Part
    n_rows: int

    def lines(self) -> Iterator[tuple[int, list[_Pair]]]:
        """Read every line of the file and flatten it into pairs, in file order.

        Yields:
            The 0-based line number and the line's pairs, empty answers included.

        Raises:
            TimeFValidationError: If a line breaks a rule of the release, with its file, line and pair.
        """
        for number, line in enumerate(JsonlDataReader(self.path, _Line).iter_rows()):
            try:
                yield number, self._line_pairs(number, line)
            except ValueError as exc:
                raise TimeFValidationError(f"{self.part.source_file} line {number + 1}: {exc}") from exc

    def _line_pairs(self, number: int, line: _Line) -> list[_Pair]:
        """Split a line into its pairs.

        Returns:
            The pairs, in published order.

        Raises:
            ValueError: If the line breaks a rule of the release. The message names the pair, if any.
        """
        rows = _row_ids(line, self.n_rows)
        turns = line.conversations
        if not turns or len(turns) % 2:
            raise ValueError("conversations must alternate a human question and a gpt answer")
        pairs = []
        for offset in range(len(turns) // 2):
            try:
                pairs.append(_parse_pair(_Place(self.part, number, offset), rows, *turns[2 * offset : 2 * offset + 2]))
            except ValueError as exc:
                raise ValueError(f"pair {offset}: {exc}") from exc
        return pairs

    def pairs(self) -> Iterator[_Pair]:
        """Yield every pair of the file, in file order."""
        for _, pairs in self.lines():
            yield from pairs


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
    """Open the signals file for reading, as a context manager.

    Returns:
        The open ``h5py.File``.

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


def _read(file: Any, path: Path, row_id: int, steps: slice) -> np.ndarray:
    """Read steps of one row from an open signals file, and check that they are finite.

    Returns:
        The values, shaped ``(steps, 33)``.

    Raises:
        TimeFFormatError: If the file no longer holds the row, or a block of it cannot be read.
        TimeFValidationError: If a value is NaN or infinite.
    """
    try:
        values = np.asarray(file[_SEQ_DATA][row_id - 1, steps], dtype=np.float64)
    except (KeyError, IndexError, ValueError, OSError) as exc:
        raise TimeFFormatError(f"{path}: cannot read row id {row_id}: {exc}") from exc
    if values.ndim != len(_ROW_SHAPE) or values.shape[1:] != _ROW_SHAPE[1:]:
        raise TimeFFormatError(f"{path}: row id {row_id} is shaped {values.shape}; the file changed")
    if not np.isfinite(values).all():
        raise TimeFValidationError(f"{path}: row id {row_id} holds a NaN or an infinite value")
    return values


def _real_steps(values: np.ndarray, where: str) -> int:
    """Count the steps before a row's padding: the release fills the end of a flight's last row with zeros.

    Returns:
        The number of steps before the trailing steps whose 33 values are all exactly zero.

    Raises:
        TimeFValidationError: If the row holds only zeros, or an all-zero step before its last real step.
    """
    real = ~(values == 0).all(axis=1)
    if not real.any():
        raise TimeFValidationError(f"{where} holds only zeros")
    n_real = int(np.flatnonzero(real)[-1]) + 1
    if not real[:n_real].all():
        raise TimeFValidationError(f"{where} has a step of 33 zeros before its last real step")
    return n_real


def _row_lengths(path: Path, rows: Iterable[int]) -> dict[int, int]:
    """Find how many real steps each row has, reading a whole row only when its last step is all zeros.

    Returns:
        The number of real steps of each row.
    """
    lengths = {}
    with _open_signals(path) as file:
        for row in sorted(rows):
            last = _read(file, path, row, slice(_N_STEPS - 1, _N_STEPS))
            if (last == 0).all():
                lengths[row] = _real_steps(_read(file, path, row, slice(None)), f"{path}: row id {row}")
            else:
                lengths[row] = _N_STEPS
    return lengths


class _RowReader:
    """Read the real steps of one row when a signal of its Record asks for them.

    The writer loads a Record's signals one after another, so caching the last row reads each row once.
    """

    def __init__(self, path: Path, lengths: Mapping[int, int]) -> None:
        self._path = path
        self._lengths = lengths
        self._row = lru_cache(maxsize=1)(self._read)

    def _read(self, row_id: int) -> np.ndarray:
        with _open_signals(self._path) as file:
            values = _read(file, self._path, row_id, slice(None))
        if values.shape != _ROW_SHAPE:
            raise TimeFFormatError(f"{self._path}: row id {row_id} is shaped {values.shape}; the file changed")
        n_real = _real_steps(values, f"{self._path}: row id {row_id}")
        if n_real != self._lengths[row_id]:
            raise TimeFFormatError(f"{self._path}: row id {row_id} has {n_real} real steps; the file changed")
        return values[:n_real]

    def __call__(self, row_id: int, channel: int) -> pa.Array:
        """Return one channel of a row's real steps.

        Returns:
            The channel's float64 values.
        """
        return pa.array(self._row(row_id)[:, channel], type=pa.float64())


class _JoinedReader:
    """Join the first 60 steps of 10 rows when a signal of their Record asks for them.

    A step that is padding in its row becomes null. The cache reads each group of rows once for all 33 channels.
    """

    def __init__(self, path: Path, lengths: Mapping[int, int]) -> None:
        self._path = path
        self._lengths = lengths
        self._joined = lru_cache(maxsize=1)(self._read)

    def _read(self, rows: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        blocks = []
        with _open_signals(self._path) as file:
            for row in rows:
                values = _read(file, self._path, row, slice(0, _STEPS_PER_ROW))
                n_real = min(self._lengths[row], _STEPS_PER_ROW)
                where = f"{self._path}: row id {row}"
                if n_real < _STEPS_PER_ROW and not (values[n_real:] == 0).all():
                    raise TimeFFormatError(f"{where} has values after its padding started; the file changed")
                if n_real != _real_steps(values[:n_real], where):
                    raise TimeFFormatError(f"{where} has padding before step {n_real}; the file changed")
                blocks.append(values)
        missing = np.concatenate([_block_missing(self._lengths[row]) for row in rows])
        return np.concatenate(blocks), missing

    def __call__(self, rows: tuple[int, ...], channel: int) -> pa.Array:
        """Return one channel of the joined rows.

        Returns:
            The channel's 600 float64 values, with a null for each padded step.
        """
        values, missing = self._joined(rows)
        return pa.array(values[:, channel], mask=missing, type=pa.float64())


def _block_missing(length: int) -> np.ndarray:
    """Mark the padded steps among the first 60 steps of a row with ``length`` real steps.

    Returns:
        One flag per step, ``True`` for padding.
    """
    return np.arange(_STEPS_PER_ROW) >= length


def _row_record(row_id: int, length: int, reader: _RowReader) -> Record:
    """Build the Record of one row: 33 lazy signals over its real steps, at 1 Hz.

    Returns:
        The Record, with the row id and the number of padded steps the connector left out as metadata.
    """
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=_ROW_SPEC,
            time_axis=_ROW_AXIS,
            n_values=length,
            null_count=0,
            loader=partial(reader, row_id, channel),
        )
        for channel, name in enumerate(_CHANNELS)
    )
    metadata: dict[str, JsonValue] = {_Key.ROW_ID: row_id, _Key.PADDED_STEPS: _N_STEPS - length}
    return Record(sources=[Source(_SOURCE_NAME, signals=signals)], metadata=metadata)


def _joined_record(rows: tuple[int, ...], lengths: Mapping[int, int], reader: _JoinedReader) -> Record:
    """Build the Record of a multi-row question: the first 60 steps of each of its 10 rows, joined in order.

    Returns:
        The Record, with the row ids and the steps taken from each row as metadata.
    """
    null_count = int(sum(_block_missing(lengths[row]).sum() for row in rows))
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=_JOINED_SPEC,
            time_axis=_JOINED_AXIS,
            n_values=_N_STEPS,
            null_count=null_count,
            loader=partial(reader, rows, channel),
        )
        for channel, name in enumerate(_CHANNELS)
    )
    metadata: dict[str, JsonValue] = {_Key.ROW_IDS: list(rows), _Key.STEPS_PER_ROW: _STEPS_PER_ROW}
    return Record(sources=[Source(_SOURCE_NAME, signals=signals)], metadata=metadata)


@dataclass
class _Scan:
    """What one read of every QA pair finds."""

    single_rows: set[int] = field(default_factory=set)
    """The rows that labelled single-row pairs name."""
    row_groups: dict[tuple[int, ...], None] = field(default_factory=dict)
    """The row lists that labelled multi-row pairs name, in first-seen order."""
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
        n_lines = 0
        for _, line_pairs in qa.lines():
            n_lines += 1
            for pair in line_pairs:
                pairs[pair.stage] += 1
                question = pair.missing_label
                if question is not None:
                    scan.unlabelled.append(pair.place)
                    unlabelled[question] += 1
                    continue
                digests[pair.digest] += 1
                if pair.stage.is_multi_row:
                    scan.row_groups[pair.rows] = None
                else:
                    scan.single_rows.update(pair.rows)
        found = _FileCounts(n_lines, dict(pairs), dict(unlabelled))
        if found != counts[qa.part]:
            raise TimeFValidationError(f"{qa.path}: expected {counts[qa.part]}, found {found}")
        scan.digests[qa.part] = digests
    if scan.unlabelled:
        _LOGGER.warning("%d closed pairs have an empty answer and get no task", len(scan.unlabelled))
    return scan


def _unlabelled_annotation(places: Sequence[_Place]) -> Annotation:
    pairs: list[JsonValue] = [place.metadata for place in places]
    return Annotation(
        _Key.UNLABELLED_PAIRS,
        value=len(places),
        description="How many closed pairs have an empty published answer. They are missing labels and have no task.",
        metadata={"pairs": pairs},
    )


class EngineMTQAConnector(
    BaseHuggingFaceConnector,
    hf_repo="pandalin98/EngineMT-QA",
    source_revision="ea2051a15512212976c22c1a075ec549c7d48eef",
    source_files={part.value: part.source_file for part in _Part},
):
    """Build EngineMT-QA from the pinned files of its Hugging Face release."""

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

    def convert(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Convert the files into lazy Records and a stream of one task per labelled pair.

        Conversion checks the signals file's layout, every QA line, and each QA file's pinned counts. It reads
        the last step of each named row to find padded rows. The other values stay in the HDF5 file until
        written, so the files must stay available until then.

        Args:
            files: The downloaded file of each part.

        Returns:
            A dataset with the Records, the unlabelled pairs as an annotation, and a stream of every task.

        Raises:
            TimeFValidationError: If a file is malformed or has other counts.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _check_layout and _scan
        signals_path = files[_Part.SIGNALS]
        n_rows = _check_layout(signals_path)
        qa_files = [_QAFile(files[part], part, n_rows) for part in _QA_PARTS]
        scan = _scan(qa_files, self.SOURCE_COUNTS)
        grouped_rows = {row for rows in scan.row_groups for row in rows}
        lengths = _row_lengths(signals_path, scan.single_rows | grouped_rows)
        row_reader, joined_reader = _RowReader(signals_path, lengths), _JoinedReader(signals_path, lengths)
        records: dict[tuple[int, ...], Record] = {
            (row,): _row_record(row, lengths[row], row_reader) for row in sorted(scan.single_rows)
        }
        records |= {rows: _joined_record(rows, lengths, joined_reader) for rows in scan.row_groups}

        dataset = TimeFDataset(metadata=self.metadata(), annotations=[_unlabelled_annotation(scan.unlabelled)])
        for record in records.values():
            dataset.add_record(record)

        def tasks() -> Iterator[Task]:
            for qa in qa_files:
                for pair in qa.pairs():
                    if not pair.has_label:
                        continue
                    if pair.rows not in records:
                        raise TimeFFormatError(f"{qa.path}: {pair.place} names rows {pair.rows} without a Record")
                    yield pair.task(records[pair.rows], scan.issues(pair))

        dataset.set_task_stream((AnswerTask, ClassificationTask), tasks)
        return dataset


CONNECTOR = EngineMTQAConnector
