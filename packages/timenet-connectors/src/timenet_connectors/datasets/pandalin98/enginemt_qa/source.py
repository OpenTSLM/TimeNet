"""Read and validate EngineMT-QA files and retain checked source facts."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pyarrow as pa
from pydantic import Field, field_validator
from pydantic_core import PydanticCustomError

from timenet.errors import TimeNetConnectorError
from timenet.json import JsonValue
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.datasets.pandalin98.enginemt_qa.vocabulary import (
    CHANNELS,
    DATA_ID,
    DIGEST_BYTES,
    MULTI_CYCLE_ROWS,
    N_STEPS,
    QA_PARTS,
    ROW_ID_PATTERN,
    ROW_SHAPE,
    SEQ_DATA,
    SPEC,
    Attribute,
    ClosedQuestion,
    MetadataKey,
    Part,
    SourceIssue,
    Stage,
)
from timenet_connectors.sources.jsonl import iter_jsonl_rows
from timenet_connectors.sources.rows import NonBlankText, SourceRow, validate_row
from timenet_connectors.sources.signals import numeric_signal


_LOGGER = logging.getLogger(__name__)


class FileCounts(NamedTuple):
    """The pinned size of a QA file, which conversion checks so that a short or changed file fails."""

    lines: int
    pairs: Mapping[Stage, int]
    """The pairs of each stage, empty answers included."""
    unlabelled: Mapping[ClosedQuestion, int]
    """The closed pairs with an empty answer, for each question that has any."""


class PairLocation(NamedTuple):
    """Where a pair was published."""

    part: Part
    line_index: int
    """The 0-based line in the file, which the authors call ``source_line``."""
    pair_index: int
    """The 0-based place in the line, which the authors call ``qa_offset``."""

    @property
    def metadata(self) -> dict[str, JsonValue]:
        """The place as the authors' evaluation names it."""
        return {
            MetadataKey.SOURCE_FILE: self.part.source_file,
            MetadataKey.SOURCE_LINE: self.line_index,
            MetadataKey.QA_OFFSET: self.pair_index,
        }


@dataclass(frozen=True)
class Pair:
    """One published question with its answer, and the rows it is about."""

    place: PairLocation
    rows: tuple[int, ...]
    """The row ids in published order: one for a single-cycle question and ten for a multi-cycle one."""
    stage: Stage
    question: str
    """The question as published, with its ``<ts>`` marker."""
    answer: str
    """The answer as published: free text, or letters for a closed question."""
    closed: ClosedQuestion | None
    """The closed question, or ``None`` for an open one."""

    @property
    def missing_label(self) -> ClosedQuestion | None:
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
        return hashlib.blake2b(content, digest_size=DIGEST_BYTES).digest()


class _Line(SourceRow):
    """One published JSON line, before source-specific ID and conversation checks."""

    id: str | list[str]
    name: str | list[str]
    conversations: list[dict[str, object]]


class _Question(SourceRow):
    """A published question turn with its verbatim text."""

    role: Literal["human"] = Field(alias="from")
    stage: Stage
    attribute: Attribute = Field(strict=False)
    value: NonBlankText

    @field_validator("stage", mode="before")
    @classmethod
    def _published_stage(cls, value: object) -> Stage:
        if not isinstance(value, str) or value not in {str(stage.value) for stage in Stage}:
            raise PydanticCustomError("stage", "expected a stage from '1' to '4'")
        return Stage(int(value))


class _Answer(SourceRow):
    """An answer turn; blank text represents a published missing label."""

    role: Literal["gpt"] = Field(alias="from")
    value: str


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def _row_ids(line: _Line, n_rows: int) -> tuple[int, ...]:
    """Read the row ids of a line, and check that its ``name`` repeats them.

    Returns:
        One id for a single-cycle line, ten for a multi-cycle line, in published order.

    Raises:
        TimeNetConnectorError: If an id is not a row of the file, or the ids repeat or are not one or ten.
    """
    raw_ids, raw_names = line.id, line.name
    is_list = isinstance(raw_ids, list)
    ids = _as_list(raw_ids)
    names_match = isinstance(raw_names, list) == is_list and _as_list(raw_names) == [f"{i}.csv" for i in ids]
    if not names_match:
        raise TimeNetConnectorError(f"name {raw_names!r} doesn't repeat id {raw_ids!r}")
    rows: list[int] = []
    for i in ids:
        if not isinstance(i, str) or not ROW_ID_PATTERN.fullmatch(i):
            raise TimeNetConnectorError(f"id {raw_ids!r} is not decimal row ids")
        rows.append(int(i))
    if is_list and len(rows) != MULTI_CYCLE_ROWS:
        raise TimeNetConnectorError(f"a multi-cycle id names {len(rows)} rows, not {MULTI_CYCLE_ROWS}")
    if len(set(rows)) != len(rows):
        raise TimeNetConnectorError(f"id {raw_ids!r} repeats a row")
    if max(rows) > n_rows:
        raise TimeNetConnectorError(f"id {raw_ids!r} names a row past the file's {n_rows} rows")
    return tuple(rows)


def _parse_pair(place: PairLocation, rows: tuple[int, ...], question_turn: object, answer_turn: object) -> Pair:
    """Check one question and its answer against its stage, and build the pair.

    Returns:
        The pair.

    Raises:
        TimeNetConnectorError: If the stage, the format and the number of rows disagree, a closed question is unknown, or
            an answer is malformed.
    """
    question, answer = (
        validate_row(question_turn, _Question, location="question"),
        validate_row(answer_turn, _Answer, location="answer"),
    )
    stage, attribute = question.stage, question.attribute
    if (attribute is Attribute.CLOSE) != stage.is_closed:
        raise TimeNetConnectorError(f"stage {stage} questions are not {attribute.value!r}")
    if (len(rows) > 1) != stage.is_multi_cycle:
        raise TimeNetConnectorError(f"a stage {stage} question names {len(rows)} rows")
    text, reply = question.value, answer.value
    if text.count("<ts>") != 1:
        raise TimeNetConnectorError("a question must hold one <ts> marker")
    closed = ClosedQuestion.of(stage, text) if stage.is_closed else None
    if closed is None and not reply.strip():
        raise TimeNetConnectorError("an open question has an empty answer")
    if closed is not None and reply:
        # Parse here, so a malformed answer fails the read and not the task stream.
        closed.parse_answer(reply)
    return Pair(place, rows, stage, text, reply, closed)


@dataclass
class QAFile:
    """One QA file. It is read twice, and the second read must see the file the first one did."""

    path: Path
    part: Part
    n_rows: int
    line_digests: list[bytes] | None = field(default=None, init=False)
    """A digest of each line, kept by the first complete read."""

    def pairs(self) -> Iterator[Pair]:
        """Read every line of the file and flatten it into pairs, in file order.

        The first complete read keeps a digest of each line. A later read checks each line against its
        digest before it builds pairs.

        Yields:
            Each published pair, empty answers included.

        Raises:
            TimeNetConnectorError: If a line or a pair is malformed, with its file, line and pair.
            TimeNetConnectorError: If the file cannot be read, or changed since the first complete read.
        """
        expected = self.line_digests
        digests: list[bytes] = []
        try:
            for located in iter_jsonl_rows(self.path, _Line):
                number = located.location.line - 1
                digest = hashlib.blake2b(located.raw, digest_size=DIGEST_BYTES).digest()
                if expected is not None and (number >= len(expected) or digest != expected[number]):
                    raise TimeNetConnectorError(f"{located.location.context}: the line changed since conversion")
                digests.append(digest)
                yield from self._line_pairs(number, located.value)
        except TimeNetConnectorError as exc:
            if expected is not None:
                raise TimeNetConnectorError(f"{exc}; the file changed since conversion") from exc
            raise
        if expected is None:
            self.line_digests = digests
        elif len(digests) != len(expected):
            raise TimeNetConnectorError(f"{self.path}: lines were removed since conversion")

    def _line_pairs(self, number: int, line: _Line) -> Iterator[Pair]:
        where = f"{self.path} (line {number + 1})"
        try:
            rows = _row_ids(line, self.n_rows)
            turns = line.conversations
            if not turns or len(turns) % 2:
                raise TimeNetConnectorError("conversations must alternate a human question and a gpt answer")
            for offset in range(len(turns) // 2):
                where = f"{self.path} (line {number + 1}), pair {offset}"
                place = PairLocation(self.part, number, offset)
                yield _parse_pair(place, rows, turns[2 * offset], turns[2 * offset + 1])
        except (ValueError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{where}: {exc}") from exc


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
        TimeNetConnectorError: If the file cannot be opened as HDF5, as with a truncated download.
    """
    h5py = _import_h5py()
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise TimeNetConnectorError(f"{path}: cannot open the signals file: {exc}") from exc


def read_recording_count(path: Path) -> int:
    """Check that the signals file holds exactly ``seq_data`` and ``data_ID``, with the ids ``1..n``.

    Returns:
        The number of rows ``n``.

    Raises:
        TimeNetConnectorError: If the layout, the shape, the dtype or the ids differ from the published file's.
        TimeNetConnectorError: If the file cannot be opened or read as HDF5.
    """
    h5py = _import_h5py()
    with _open_signals(path) as file:
        if set(file) != {SEQ_DATA, DATA_ID}:
            raise TimeNetConnectorError(f"{path}: expected {SEQ_DATA} and {DATA_ID}, got {sorted(file)}")
        values, ids = file[SEQ_DATA], file[DATA_ID]
        for name, node in ((SEQ_DATA, values), (DATA_ID, ids)):
            if not isinstance(node, h5py.Dataset):
                raise TimeNetConnectorError(f"{path}: {name} is a {type(node).__name__}, not an HDF5 dataset")
        if values.shape[1:] != ROW_SHAPE or values.dtype != np.float64:
            raise TimeNetConnectorError(f"{path}: {SEQ_DATA} is {values.dtype} {values.shape}")
        n_rows = values.shape[0]
        if ids.shape != (n_rows,) or not np.issubdtype(ids.dtype, np.integer):
            raise TimeNetConnectorError(f"{path}: {DATA_ID} is {ids.dtype} {ids.shape}, not {n_rows} integer ids")
        try:
            published_ids = ids[:]
        except OSError as exc:
            raise TimeNetConnectorError(f"{path}: cannot read {DATA_ID}: {exc}") from exc
        if not np.array_equal(published_ids, np.arange(1, n_rows + 1)):
            raise TimeNetConnectorError(f"{path}: {DATA_ID} is not the row ids 1 to {n_rows}")
    return n_rows


def _read_row(file: Any, path: Path, row_id: int) -> np.ndarray:
    """Read one row's values from an open signals file.

    Returns:
        The values, shaped ``(600, 33)``: steps, then channels.

    Raises:
        TimeNetConnectorError: If the file no longer holds the row, as when it changed since conversion, or a
            block of the file cannot be read.
        TimeNetConnectorError: If the row holds a NaN or an infinite value. z-scored values are finite.
    """
    try:
        values = np.asarray(file[SEQ_DATA][row_id - 1], dtype=np.float64)
    except (KeyError, IndexError, ValueError, OSError) as exc:
        raise TimeNetConnectorError(f"{path}: cannot read row id {row_id}: {exc}") from exc
    if values.shape != ROW_SHAPE:
        raise TimeNetConnectorError(f"{path}: row id {row_id} is shaped {values.shape}; the file changed")
    if not np.isfinite(values).all():
        raise TimeNetConnectorError(f"{path}: row id {row_id} holds a NaN or an infinite value")
    return values


def make_channel_loader(path: Path) -> SharedChannelLoader[int]:
    """Return a conversion-wide bounded cache of decoded HDF5 rows."""

    def read_channels(row_id: int) -> tuple[pa.Array, ...]:
        with _open_signals(path) as file:
            values = _read_row(file, path, row_id)
        return tuple(
            numeric_signal(values[:, channel], f"{path}: row {row_id}, {name}", spec=SPEC, n_values=N_STEPS)
            for channel, name in enumerate(CHANNELS)
        )

    return SharedChannelLoader(read_channels)


@dataclass
class QAInventory:
    """What one read of every QA pair finds."""

    named_rows: set[int] = field(default_factory=set)
    """The rows that the pairs with a label name."""
    unlabelled: list[PairLocation] = field(default_factory=list)
    """The places of the closed pairs with an empty answer."""
    digests: dict[Part, Counter[bytes]] = field(default_factory=dict)
    """How often each labelled pair's digest occurs in each file."""

    def issues(self, pair: Pair) -> list[SourceIssue]:
        """Name the source issues of a labelled pair.

        Returns:
            The issues, in ``SourceIssue`` order.
        """
        digest = pair.digest
        found = []
        if self.digests[pair.place.part][digest] > 1:
            found.append(SourceIssue.REPEATED_QUESTION)
        if any(self.digests[part][digest] for part in QA_PARTS if part is not pair.place.part):
            found.append(SourceIssue.REPEATED_ACROSS_SPLITS)
        return found


def scan_questions(qa_files: Sequence[QAFile], counts: Mapping[Part, FileCounts]) -> QAInventory:
    """Check every pair once, and collect what the dataset needs from all of them.

    Returns:
        The rows that labelled pairs name, the unlabelled pairs, and the digests of the labelled pairs.

    Raises:
        TimeNetConnectorError: If a file's counts differ from the pinned ones.
    """
    scan = QAInventory()
    for qa in qa_files:
        digests: Counter[bytes] = Counter()
        pairs: Counter[Stage] = Counter()
        unlabelled: Counter[ClosedQuestion] = Counter()
        for pair in qa.pairs():
            pairs[pair.stage] += 1
            question = pair.missing_label
            if question is None:
                scan.named_rows.update(pair.rows)
                digests[pair.digest] += 1
            else:
                scan.unlabelled.append(pair.place)
                unlabelled[question] += 1
        found = FileCounts(len(qa.line_digests or ()), dict(pairs), dict(unlabelled))
        if found != counts[qa.part]:
            raise TimeNetConnectorError(f"{qa.path}: expected {counts[qa.part]}, found {found}")
        scan.digests[qa.part] = digests
    if scan.unlabelled:
        _LOGGER.warning("%d closed pairs have an empty answer and get no task", len(scan.unlabelled))
    return scan
