"""EngineMT-QA's four stages and its nine closed questions, each with its option letters.

The paper's four tasks are the release's ``stage`` values. Stages 1 and 4 are open questions, scored with
BLEU and ROUGE; stages 2 and 3 are closed, scored with accuracy and F1 (paper Section 5.1).

Each closed question is published with one text and one option list in every row, so a letter always
means the same option. The authors' model answers with the letters, such as ``a, b, e``, and their
``closed_question_metrics`` scores the set of letters. So the letters are the classes, and each closed
question's :class:`Vocabulary` names the option each letter stands for.
"""

from enum import IntEnum, StrEnum, unique
import re
from typing import NamedTuple

from timenet.errors import TimeFValidationError


@unique
class Stage(IntEnum):
    """The release's ``stage``: one of the paper's four tasks."""

    UNDERSTANDING = 1
    PERCEPTION = 2
    REASONING = 3
    DECISION_MAKING = 4

    @property
    def is_closed(self) -> bool:
        """Return whether the stage asks closed questions, which the release marks ``close``."""
        return self in {Stage.PERCEPTION, Stage.REASONING}

    @property
    def is_multi_cycle(self) -> bool:
        """Return whether the stage's questions name 10 cycles rather than one."""
        return self in {Stage.REASONING, Stage.DECISION_MAKING}


class _Options(NamedTuple):
    options: tuple[str, ...]
    """The option names, in letter order from ``a``."""
    multiple: bool
    """Whether an answer may hold several letters."""


@unique
class Vocabulary(StrEnum):
    """The option list of one or more closed questions: its letters are the classes."""

    COMPONENT_HEALTH = "component_health"
    FAULT_CAUSE = "fault_cause"
    HEALTH_STATE = "health_state"
    ENGINE_CONDITION = "engine_condition"
    FAILURE_PROBABILITY = "failure_probability"

    @property
    def options(self) -> tuple[str, ...]:
        """Return the option names, in letter order from ``a``, as the question text spells them."""
        return _VOCABULARIES[self].options

    @property
    def letters(self) -> tuple[str, ...]:
        """Return the option letters, ``a`` first."""
        return tuple("abcdef"[: len(self.options)])

    @property
    def multiple(self) -> bool:
        """Return whether an answer may hold several letters. Only the fault cause does."""
        return _VOCABULARIES[self].multiple


_VOCABULARIES = {
    Vocabulary.COMPONENT_HEALTH: _Options(("Health", "Fault"), multiple=False),
    Vocabulary.FAULT_CAUSE: _Options(
        (
            "HPT efficiency modifier",
            "LPT efficiency modifier",
            "Fan efficiency modifier",
            "HPT flow modifier",
            "LPT flow modifier",
            "Fan flow modifier",
        ),
        multiple=True,
    ),
    Vocabulary.HEALTH_STATE: _Options(
        ("Operating Normally", "Abnormal Degradation", "Imminent Failure"), multiple=False
    ),
    Vocabulary.ENGINE_CONDITION: _Options(
        ("Good Condition", "Moderate Condition", "Poor Condition", "Bad Condition", "Extremely Poor Condition"),
        multiple=False,
    ),
    Vocabulary.FAILURE_PROBABILITY: _Options(("1%-10%", "10%-30%", "30%-50%", "50%-70%", "70%-100%"), multiple=False),
}


class _Closed(NamedTuple):
    stage: Stage
    vocabulary: Vocabulary
    text: str


def _health_text(component: str) -> str:
    return f"What is the health status of {component} of the given engine signal in one cycle?\n <ts> a: Health\n b: Fault\n"


_REASONING_PREFIX = (
    "Given the time series signal <ts>, by perceiving the engine signal across 10 cycles to reflect different "
    "health states and performing temporal inference to predict the health decline trend, "
)


@unique
class ClosedQuestion(StrEnum):
    """A closed question, which the release always publishes with the same text."""

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
    def stage(self) -> Stage:
        """Return the question's stage."""
        return _CLOSED[self].stage

    @property
    def vocabulary(self) -> Vocabulary:
        """Return the question's option list."""
        return _CLOSED[self].vocabulary

    @property
    def text(self) -> str:
        """Return the question exactly as published, ``<ts>`` and options included."""
        return _CLOSED[self].text

    @classmethod
    def of(cls, stage: Stage, text: str) -> "ClosedQuestion":
        """Identify a published closed question by its exact text.

        Returns:
            The question.

        Raises:
            TimeFValidationError: If no closed question of the stage has this text.
        """
        question = _BY_TEXT.get(text)
        if question is None or question.stage is not stage:
            raise TimeFValidationError(f"unknown stage {stage} closed question {text[:80]!r}")
        return question

    def parse_answer(self, answer: str) -> tuple[str, ...]:
        """Parse a published answer, such as ``a`` or ``a, b, e``, into its letters.

        Returns:
            The letters, in published order.

        Raises:
            TimeFValidationError: If the answer isn't distinct letters of the question's options, joined by
                ``", "``, or holds several letters where one is expected.
        """
        letters = tuple(answer.split(", "))
        vocabulary = self.vocabulary
        if not set(letters) <= set(vocabulary.letters) or len(set(letters)) != len(letters):
            raise TimeFValidationError(f"{self}: answer {answer!r} is not distinct letters of {vocabulary.letters}")
        if len(letters) > 1 and not vocabulary.multiple:
            raise TimeFValidationError(f"{self}: answer {answer!r} holds several letters, where one is expected")
        return letters


_CLOSED = {
    ClosedQuestion.FAN_HEALTH: _Closed(Stage.PERCEPTION, Vocabulary.COMPONENT_HEALTH, _health_text("fan")),
    ClosedQuestion.LPC_HEALTH: _Closed(Stage.PERCEPTION, Vocabulary.COMPONENT_HEALTH, _health_text("LPC")),
    ClosedQuestion.HPC_HEALTH: _Closed(Stage.PERCEPTION, Vocabulary.COMPONENT_HEALTH, _health_text("HPC")),
    ClosedQuestion.HPT_HEALTH: _Closed(Stage.PERCEPTION, Vocabulary.COMPONENT_HEALTH, _health_text("HPT")),
    ClosedQuestion.LPT_HEALTH: _Closed(Stage.PERCEPTION, Vocabulary.COMPONENT_HEALTH, _health_text("LPT")),
    ClosedQuestion.FAULT_CAUSE: _Closed(
        Stage.PERCEPTION,
        Vocabulary.FAULT_CAUSE,
        "Was the timing issue in the operation of the given engine signal in one cycle caused by efficiency or "
        "flow problems?\n <ts> a: HPT efficiency modifier\n b: LPT efficiency modifier\n c: Fan efficiency "
        "modifier\n d: HPT flow modifier\n e: LPT flow modifier\n f: Fan flow modifier\n",
    ),
    ClosedQuestion.HEALTH_STATE: _Closed(
        Stage.PERCEPTION,
        Vocabulary.HEALTH_STATE,
        "What is the potential health state of the given engine signal in one cycle?\n <ts> a: Operating "
        "Normally\n b: Abnormal Degradation\n c: Imminent Failure\n",
    ),
    # The published text has two spaces after "a:".
    ClosedQuestion.ENGINE_CONDITION: _Closed(
        Stage.REASONING,
        Vocabulary.ENGINE_CONDITION,
        _REASONING_PREFIX + "what is the qualitative condition of the engine? a:  Good Condition\n b: Moderate "
        "Condition\n c: Poor Condition\n d: Bad Condition\n e: Extremely Poor Condition\n",
    ),
    ClosedQuestion.FAILURE_PROBABILITY: _Closed(
        Stage.REASONING,
        Vocabulary.FAILURE_PROBABILITY,
        _REASONING_PREFIX + "what is the probability range of engine failure? a: 1%-10%\n b: 10%-30%\n "
        "c: 30%-50%\n d: 50%-70%\n e: 70%-100%\n",
    ),
}
_BY_TEXT = {closed.text: question for question, closed in _CLOSED.items()}

_OPTION = re.compile(r"([a-f]):\s+([^\n]+)\n")


def _check_texts() -> None:
    """Check at import that each question's text offers exactly its vocabulary's options.

    Raises:
        TimeFValidationError: If a text and its vocabulary disagree.
    """
    for question, closed in _CLOSED.items():
        offered = tuple(_OPTION.findall(closed.text.split("<ts>", 1)[1].split("?")[-1]))
        if offered != tuple(zip(closed.vocabulary.letters, closed.vocabulary.options, strict=True)):
            raise TimeFValidationError(f"{question}: its text offers {offered}, not its vocabulary")


_check_texts()
