"""The SensorCaps source task: one row's caption of its IMU Record."""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import AnswerTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask


INPUT_ANNOTATION_KEYS = frozenset({"summary", "location"})
"""The record annotations that a task gives a model: the activity label and the sensor location.

The derived features stay on the record only. They are about 36,000 characters of NumPy text per
row, and the source computed them from a 120-reading window, not from the 60 published readings.
"""


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
        """Return the caption task: the instruction as prompt, and the readings and their context as inputs.

        Returns:
            The task as TimeNet stores it.
        """
        return AnswerTask(
            inputs=(self.record,),
            prompt=self.instruction,
            targets=(self.caption,),
            # The summary and the location of the user message, in Record order.
            input_annotations=tuple(
                annotation for annotation in self.record.annotations if annotation.key in INPUT_ANNOTATION_KEYS
            ),
            split=self.split,
        )
