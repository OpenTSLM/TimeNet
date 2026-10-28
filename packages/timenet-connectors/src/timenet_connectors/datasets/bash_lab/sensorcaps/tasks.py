"""The SensorCaps source task: one row's caption of its IMU Record."""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import AnswerTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask


@dataclass(frozen=True)
class SensorCaptionTask(DatasetTask):
    """One IMU-caption pair, as GPT-4o-mini generated it: the instruction, its inputs, and the caption."""

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    record: Record
    instruction: str
    """The system message, verbatim: the instruction the caption answers."""
    caption: str
    """The assistant message, verbatim: its ``Features:`` summary, then its ``Narration:``."""

    @property
    def split(self) -> Split | None:
        """Return ``None``: the source publishes no split of its rows."""
        return None

    def to_timenet_task(self) -> AnswerTask:
        """Return the caption task: the instruction as prompt, and the generator's inputs as task inputs.

        Returns:
            The task as TimeNet stores it.
        """
        return AnswerTask(
            inputs=(self.record,),
            prompt=self.instruction,
            targets=(self.caption,),
            # The rest of the user message: the summary, the derived features, and the location.
            input_annotations=self.record.annotations,
            split=self.split,
        )
