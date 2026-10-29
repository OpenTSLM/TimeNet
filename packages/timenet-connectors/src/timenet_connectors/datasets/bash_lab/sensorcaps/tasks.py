"""The SensorCaps source task: one row's caption of its IMU Record."""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import AnswerTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import ParsedRow


# Record annotations the captioning model receives with the sensor values.
_CONTEXT_KEYS = frozenset({"activity_summary", "sensor_location"})


@dataclass(frozen=True)
class SensorCaptionTask(DatasetTask):
    """One source row's caption, prompted by its system message and derived-feature text."""

    TIMENET_TYPE: ClassVar[type[Task]] = AnswerTask

    record: Record
    row: ParsedRow

    @property
    def split(self) -> Split:
        """Return the training split, which every SensorCaps task belongs to."""
        return Split.TRAIN

    def to_timenet_task(self) -> AnswerTask:
        """Return the caption task with its instruction prompt and context annotations.

        Returns:
            The task as TimeNet stores it.
        """
        row = self.row
        context = tuple(annotation for annotation in self.record.annotations if annotation.key in _CONTEXT_KEYS)
        return AnswerTask(
            inputs=(self.record,),
            targets=(row.caption,),
            prompt=f"{row.system}\n\n{row.features}" if row.features.strip() else row.system,
            input_annotations=context,
            split=self.split,
        )
