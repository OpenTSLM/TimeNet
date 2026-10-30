"""EngineMT-QA's four tasks, one per stage, each QA pair built as the TimeNet task it becomes.

Every task keeps the published question as its prompt, ``<ts>`` marker included, and its file as its
split. Its metadata names the pair as the authors' evaluation does: the file, the 0-based line
(``source_line``) and the 0-based pair in the line (``qa_offset``).
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, ClassificationTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.pandalin98.enginemt_qa.keys import vocabulary_key
from timenet_connectors.datasets.pandalin98.enginemt_qa.qa import QAPair
from timenet_connectors.datasets.pandalin98.enginemt_qa.questions import Stage
from timenet_connectors.datasets.pandalin98.enginemt_qa.release import MULTI_CYCLE_ROWS, N_STEPS


STEPS_PER_CYCLE = N_STEPS // MULTI_CYCLE_ROWS
"""The steps of each cycle the authors' model reads in a multi-cycle task: the first 60 of 600.

``TsQaDataset.__getitem__`` concatenates ``seq_data[id - 1][: 600 // 10]`` over the 10 ids, so its input
keeps the single-cycle shape (600, 33). The task's inputs are the 10 whole Records; ``model_input``
rebuilds the authors' input from them.
"""


@dataclass(frozen=True)
class EngineMTQATask(DatasetTask):
    """One published QA pair, on the Records of the rows it names, in published order."""

    STAGE: ClassVar[Stage]

    pair: QAPair
    records: tuple[Record, ...]

    @property
    def split(self) -> Split:
        """Return the task's split: its QA file's."""
        return self.pair.ref.split

    def _metadata(self) -> dict[str, Any]:
        ref = self.pair.ref
        metadata: dict[str, Any] = {
            "source_file": ref.part.source_file,
            "source_line": ref.line,
            "qa_offset": ref.qa_offset,
        }
        if self.STAGE.is_multi_cycle:
            metadata["steps_per_cycle"] = STEPS_PER_CYCLE
        return metadata


@dataclass(frozen=True)
class _OpenTask(EngineMTQATask):
    """An open question, which the authors score with BLEU and ROUGE against the published answer."""

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    def to_timenet_task(self) -> AnswerTask:
        """Return the answer task, with the published answer as its target."""
        return AnswerTask(
            inputs=self.records,
            prompt=self.pair.question,
            targets=(self.pair.answer,),
            split=self.split,
            metadata=self._metadata(),
        )


@dataclass(frozen=True)
class _ClosedTask(EngineMTQATask):
    """A closed question, whose classes are its option letters, which the authors score as a set."""

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the published letters as targets.

        Raises:
            TimeFValidationError: If the pair is open, or its answer is empty.
        """
        closed, letters = self.pair.closed, self.pair.letters
        if closed is None or not letters:
            raise TimeFValidationError(f"{self.pair.ref.context}: a closed task needs a published answer")
        return ClassificationTask(
            inputs=self.records,
            prompt=self.pair.question,
            targets=letters,
            target_schema=vocabulary_key(closed.vocabulary),
            split=self.split,
            metadata=self._metadata() | {"closed_question": closed.value},
        )


@dataclass(frozen=True)
class UnderstandingTask(_OpenTask):
    """Stage 1: describe one cycle's change in a named signal and what it implies."""

    STAGE: ClassVar[Stage] = Stage.UNDERSTANDING


@dataclass(frozen=True)
class PerceptionTask(_ClosedTask):
    """Stage 2: from one cycle, name a component's health, the fault's causes, or the engine's health state."""

    STAGE: ClassVar[Stage] = Stage.PERCEPTION


@dataclass(frozen=True)
class ReasoningTask(_ClosedTask):
    """Stage 3: from 10 cycles, name the engine's condition or its failure probability range."""

    STAGE: ClassVar[Stage] = Stage.REASONING


@dataclass(frozen=True)
class DecisionTask(_OpenTask):
    """Stage 4: from 10 cycles, recommend maintenance actions."""

    STAGE: ClassVar[Stage] = Stage.DECISION_MAKING


TASK_KINDS: tuple[type[EngineMTQATask], ...] = (UnderstandingTask, PerceptionTask, ReasoningTask, DecisionTask)
"""The task kinds, in stage order."""

KIND_OF_STAGE = {kind.STAGE: kind for kind in TASK_KINDS}
