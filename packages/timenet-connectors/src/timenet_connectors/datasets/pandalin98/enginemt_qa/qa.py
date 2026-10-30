"""Read the published QA files: one JSON line per row or 10-row group, flattened into QA pairs.

A line is ``{"id": ..., "name": ..., "conversations": [question, answer, ...]}``. ``id`` names one HDF5
row (a string such as ``"7323"``) or ten (a list), and ``name`` repeats each as ``"<id>.csv"``. The
conversation alternates a ``human`` question, carrying its ``stage`` and ``attribute``, and a ``gpt``
answer. The authors' ``TsQaDataset._build_index`` flattens every line into one sample per pair, in file
order, and :func:`read_pairs` does the same.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints, ValidationError, model_validator

from timenet.errors import TimeFValidationError
from timenet.types import Split
from timenet_connectors.datasets.pandalin98.enginemt_qa.questions import ClosedQuestion, Stage
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import MULTI_CYCLE_ROWS, N_ROWS, Part, split_of


# A row id is a decimal string, as published; the HDF5 row is the id minus one.
RowId = Annotated[StrictStr, StringConstraints(pattern=r"^[1-9][0-9]*$")]
Text = Annotated[StrictStr, StringConstraints(pattern=r"\S")]


class _Question(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", populate_by_name=True)

    speaker: Literal["human"] = Field(alias="from")
    stage: Literal["1", "2", "3", "4"]
    attribute: Literal["open", "close"]
    value: Text


class _Answer(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", populate_by_name=True)

    speaker: Literal["gpt"] = Field(alias="from")
    value: StrictStr
    """Empty for some closed answers, as published."""


class QALine(BaseModel):
    """One published line, validated: its rows, and its questions with their answers."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    id: RowId | list[RowId]
    name: StrictStr | list[StrictStr]
    conversations: list[_Question | _Answer] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> "QALine":
        ids = self.id if isinstance(self.id, list) else [self.id]
        names = self.name if isinstance(self.name, list) else [self.name]
        if isinstance(self.id, list) != isinstance(self.name, list) or names != [f"{i}.csv" for i in ids]:
            raise TimeFValidationError(f"name {self.name!r} doesn't repeat id {self.id!r}")
        if isinstance(self.id, list) and len(ids) != MULTI_CYCLE_ROWS:
            raise TimeFValidationError(f"a multi-cycle id names {len(ids)} rows, not {MULTI_CYCLE_ROWS}")
        if len(set(ids)) != len(ids) or not all(int(i) <= N_ROWS for i in ids):
            raise TimeFValidationError(f"id {self.id!r} repeats a row or names one past {N_ROWS}")
        self.pairs()
        return self

    @property
    def rows(self) -> tuple[int, ...]:
        """Return the row ids the line names, in published order."""
        return tuple(int(i) for i in (self.id if isinstance(self.id, list) else [self.id]))

    def pairs(self) -> list[tuple[_Question, _Answer]]:
        """Return the conversation's questions, each with its answer, in published order.

        Raises:
            TimeFValidationError: If the conversation doesn't alternate a question and an answer.
        """
        turns = self.conversations
        if len(turns) % 2:
            raise TimeFValidationError("conversations must alternate a human question and a gpt answer")
        pairs = []
        for question, answer in zip(turns[::2], turns[1::2], strict=True):
            if not isinstance(question, _Question) or not isinstance(answer, _Answer):
                raise TimeFValidationError("conversations must alternate a human question and a gpt answer")
            pairs.append((question, answer))
        return pairs


@dataclass(frozen=True)
class QARef:
    """Where one QA pair was published: its file, its 0-based line, and its 0-based pair in the line.

    ``line`` and ``qa_offset`` are the authors' ``source_line`` and ``qa_offset``.
    """

    part: Part
    line: int
    qa_offset: int

    @property
    def split(self) -> Split:
        """Return the pair's split: its file's."""
        return split_of(self.part)

    @property
    def context(self) -> str:
        """Return where the pair is, for error messages."""
        return f"{self.part.source_file} line {self.line + 1}, pair {self.qa_offset}"


@dataclass(frozen=True)
class QAPair:
    """One published question and its answer, with the rows it is about."""

    ref: QARef
    rows: tuple[int, ...]
    """The row ids, one for a single-cycle question and ten for a multi-cycle one, in published order."""
    stage: Stage
    question: str
    """The question exactly as published, with its ``<ts>`` marker."""
    answer: str
    """The answer exactly as published: free text, or letters for a closed question."""
    closed: ClosedQuestion | None
    """The closed question, or None for an open one."""

    @property
    def letters(self) -> tuple[str, ...]:
        """Return a closed answer's letters, in published order; empty for a missing answer.

        Raises:
            TimeFValidationError: If the pair is an open question.
        """
        if self.closed is None:
            raise TimeFValidationError(f"{self.ref.context}: an open question has no letters")
        return self.closed.parse_answer(self.answer) if self.answer else ()


def _pair(ref: QARef, rows: tuple[int, ...], question: _Question, answer: _Answer) -> QAPair:
    """Check a question against its stage's format and number of rows, and pair it with its answer.

    Returns:
        The pair.

    Raises:
        TimeFValidationError: If the stage, attribute and rows disagree, a closed question is unknown, or an
            answer is malformed.
    """
    stage = Stage(int(question.stage))
    if (question.attribute == "close") != stage.is_closed:
        raise TimeFValidationError(f"stage {stage} questions are not {question.attribute!r}")
    if (len(rows) > 1) != stage.is_multi_cycle:
        raise TimeFValidationError(f"a stage {stage} question names {len(rows)} rows")
    if question.value.count("<ts>") != 1:
        raise TimeFValidationError("a question must hold one <ts> marker")
    closed = ClosedQuestion.of(stage, question.value) if stage.is_closed else None
    if closed is None and not answer.value.strip():
        raise TimeFValidationError("an open question has an empty answer")
    if closed is not None and answer.value:
        # Parse once here, so a malformed answer fails the read, not the task stream.
        closed.parse_answer(answer.value)
    return QAPair(ref, rows, stage, question.value, answer.value, closed)


def read_pairs(path: Path, part: Part) -> Iterator[QAPair]:
    """Validate every line of one QA file and flatten it into pairs, in file order.

    Yields:
        Each published QA pair, empty answers included.

    Raises:
        TimeFValidationError: If a line or pair is malformed, with file, line and pair.
    """
    with path.open(encoding="utf-8") as lines:
        for number, text in enumerate(lines):
            ref = QARef(part, number, 0)
            try:
                line = QALine.model_validate(json.loads(text))
                for offset, (question, answer) in enumerate(line.pairs()):
                    ref = QARef(part, number, offset)
                    yield _pair(ref, line.rows, question, answer)
            except (ValidationError, ValueError) as exc:
                raise TimeFValidationError(f"{ref.context}: {exc}") from exc
