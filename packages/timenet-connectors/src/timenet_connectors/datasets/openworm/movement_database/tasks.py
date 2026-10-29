"""The movement database's one benchmark task: name a video's strain from the worm's posture over time."""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import ClassificationTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.openworm.movement_database.benchmark import split_of
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey


@dataclass(frozen=True)
class StrainTask(DatasetTask):
    """One video's class, which the authors' classifier predicts from the whole video's skeletons.

    The authors classify a whole video, joining all its trajectory pieces, so the task has no scope. Their
    model reads 48 tangent angles, or 6 eigenworms, computed from the skeletons after resampling them to
    25 fps; README.md says how. The benchmark asks no question in words, so the task has no prompt.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    record: Record
    fold: int
    target: str
    """The class: the strain, then the sex chromosomes, such as ``N2_XO``."""

    @property
    def split(self) -> Split:
        """Return the task's split: test for the authors' test fold, else train."""
        return split_of(self.fold)

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the video's class as target."""
        return ClassificationTask(
            inputs=(self.record,),
            targets=(self.target,),
            target_schema=AnnotationKey.STRAIN_VOCABULARY,
            split=self.split,
        )
