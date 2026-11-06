"""EngineMT-QA release constants, channel contracts and closed-question vocabulary."""

from collections.abc import Sequence
from enum import IntEnum, StrEnum, unique
import re
import string
from typing import NamedTuple

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Label,
    LabelSet,
    Split,
    TimeSeriesSpec,
)


N_STEPS = 600


_N_CHANNELS = 33


"""The released channel count. The paper says 32. The authors' audit confirms 33 and names none of them."""


MULTI_CYCLE_ROWS = 10


"""How many rows a multi-cycle QA names."""


STEPS_PER_CYCLE = N_STEPS // MULTI_CYCLE_ROWS


"""The steps of each cycle that the authors' model reads in a multi-cycle task: the first 60 of 600."""


_SAMPLE_RATE_HZ = 1


"""The paper's "10-minute (600 points) time-series segments" (Section 5.2). The HDF5 file states no rate."""


ROW_SHAPE = (N_STEPS, _N_CHANNELS)


SEQ_DATA = "seq_data"


"""The HDF5 dataset of every row's values, shaped ``(rows, steps, channels)``."""


DATA_ID = "data_ID"


"""The HDF5 dataset of each row's id, which a QA's ``id`` names. Row ``i`` has id ``i + 1``."""


SOURCE_NAME = "N-CMAPSS engine"


CHANNELS = tuple(f"channel_{index:02d}" for index in range(_N_CHANNELS))


"""The release names no channel. The zero-padded names sort in the source's order."""


# In 24 sampled rows, every channel has a mean near 0 and a standard deviation near 1, so the values
# look z-scored. The release states neither the units nor the statistics.
SPEC = TimeSeriesSpec(
    spec_type="enginemt_qa_channel", name="EngineMT-QA channel, z-scored", unit_value=None, dtype="float64"
)


AXIS = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ)


# A published row id: a decimal number with no sign and no leading zero, such as "278".
ROW_ID_PATTERN = re.compile(r"[1-9][0-9]*")


# One printed option of a closed question: its letter, then its text, such as "a: Health\n".
_OPTION = re.compile(r"([a-z]):\s+([^\n]+)\n")


DIGEST_BYTES = 16


"""The size of a line or pair digest. 16 bytes make an accidental match between two of the release's
144,042 pairs practically impossible, and keep the digests of a file small."""


@unique
class MetadataKey(StrEnum):
    """Keys stored in record, source or task metadata."""

    ROW_ID = "row_id"
    SOURCE_FILE = "source_file"
    SOURCE_LINE = "source_line"
    QA_OFFSET = "qa_offset"
    STAGE = "stage"
    STEPS_PER_CYCLE = "steps_per_cycle"
    CLOSED_QUESTION = "closed_question"
    SOURCE_ISSUES = "source_issues"


@unique
class AnnotationKey(StrEnum):
    """Keys of attached annotations."""

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
class Part(StrEnum):
    """The release's three files: the signals, and the train and test QA."""

    SIGNALS = "signals"
    TRAIN = "train"
    TEST = "test"

    @property
    def source_file(self) -> str:
        """The file's name in the Hub repository, such as ``train_qa.jsonl``."""
        return "time_series_data.h5" if self is Part.SIGNALS else f"{self.value}_qa.jsonl"

    @property
    def timef_split(self) -> Split:
        """The split of a QA file: the authors train on ``train_qa`` and test on ``test_qa``.

        Raises:
            TimeNetConnectorError: If the part is not a QA file.
        """
        if self not in _SPLITS:
            raise TimeNetConnectorError(f"{self} is not a QA file")
        return _SPLITS[self]


_SPLITS = {Part.TRAIN: Split.TRAIN, Part.TEST: Split.TEST}


QA_PARTS = tuple(_SPLITS)


@unique
class Stage(IntEnum):
    """The release's ``stage``: one of the paper's four tasks."""

    UNDERSTANDING = 1
    PERCEPTION = 2
    REASONING = 3
    DECISION_MAKING = 4

    @property
    def is_closed(self) -> bool:
        """Whether the stage asks closed questions, which the release marks ``close``."""
        return self in {Stage.PERCEPTION, Stage.REASONING}

    @property
    def is_multi_cycle(self) -> bool:
        """Whether the stage's questions name 10 cycles rather than one."""
        return self in {Stage.REASONING, Stage.DECISION_MAKING}


@unique
class Attribute(StrEnum):
    """The release's ``attribute`` of a question."""

    OPEN = "open"
    CLOSE = "close"


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
    stage: Stage
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
class ClosedQuestion(StrEnum):
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
    def stage(self) -> Stage:
        """The question's stage."""
        return _CLOSED[self].stage

    @property
    def label_set(self) -> LabelSet:
        """The question's option letters."""
        return _CLOSED[self].label_set

    @classmethod
    def of(cls, stage: Stage, text: str) -> "ClosedQuestion":
        """Identify a published closed question by its stem, and check that it prints its label set's options.

        Args:
            stage: The stage that the release gives the question.
            text: The published question.

        Returns:
            The question.

        Raises:
            TimeNetConnectorError: If no closed question of the stage has this stem, or its printed options differ from
                the label set's, which would change what each letter means.
        """
        stem, _, options = text.partition("?")
        question = _BY_STEM.get(stem)
        if question is None or question.stage is not stage:
            raise TimeNetConnectorError(f"unknown stage {stage} closed question {text[:80]!r}")
        if not _prints_options(options, question.label_set):
            raise TimeNetConnectorError(f"{question} prints options other than its label set's: {options!r}")
        return question

    def parse_answer(self, answer: str) -> tuple[str, ...]:
        """Parse a published answer, such as ``a`` or ``a, b, e``, into its letters.

        Returns:
            The letters, in published order.

        Raises:
            TimeNetConnectorError: If the answer isn't distinct letters of the question's options, joined by ``", "``,
                or holds several letters where one is expected.
        """
        letters = tuple(answer.split(", "))
        try:
            self.label_set.check_targets(letters)
        except TimeNetConnectorError as exc:
            raise TimeNetConnectorError(f"{self}: answer {answer!r}: {exc}") from exc
        return letters


def _health_spec(component: str) -> _ClosedSpec:
    return _ClosedSpec(Stage.PERCEPTION, _COMPONENT_HEALTH, f"What is the health status of {component} {_ONE_CYCLE}")


_CLOSED = {
    ClosedQuestion.FAN_HEALTH: _health_spec("fan"),
    ClosedQuestion.LPC_HEALTH: _health_spec("LPC"),
    ClosedQuestion.HPC_HEALTH: _health_spec("HPC"),
    ClosedQuestion.HPT_HEALTH: _health_spec("HPT"),
    ClosedQuestion.LPT_HEALTH: _health_spec("LPT"),
    ClosedQuestion.FAULT_CAUSE: _ClosedSpec(
        Stage.PERCEPTION,
        _FAULT_CAUSE,
        f"Was the timing issue in the operation {_ONE_CYCLE} caused by efficiency or flow problems",
    ),
    ClosedQuestion.HEALTH_STATE: _ClosedSpec(
        Stage.PERCEPTION, _HEALTH_STATE, f"What is the potential health state {_ONE_CYCLE}"
    ),
    ClosedQuestion.ENGINE_CONDITION: _ClosedSpec(
        Stage.REASONING, _ENGINE_CONDITION, _REASONING_STEM + "what is the qualitative condition of the engine"
    ),
    ClosedQuestion.FAILURE_PROBABILITY: _ClosedSpec(
        Stage.REASONING, _FAILURE_PROBABILITY, _REASONING_STEM + "what is the probability range of engine failure"
    ),
}


_BY_STEM = {spec.stem: question for question, spec in _CLOSED.items()}
