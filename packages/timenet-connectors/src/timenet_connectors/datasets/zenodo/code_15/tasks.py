"""CODE-15%'s two tasks per exam: its patient's age, and its abnormalities among the six.

Lima et al. 2021 predict the age, and Ribeiro et al. 2020 classify the abnormalities. The release defines no split. Each task takes the split its paper uses CODE-15% as: Lima et al. hold all of it
out to test their age model, and Ribeiro et al. train their abnormality model on the CODE cohort it is part of.
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, ScalarPredictionTask, Split, Task, ureg
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.zenodo.code_15.exams import ExamRow
from timenet_connectors.datasets.zenodo.code_15.keys import AnnotationKey


AGE_SPLIT = Split.TEST
"""Lima et al. 2021: "The CODE cohort was divided into 85-15% splits, with the 85% split being used for
developing the model and 15% hold-out being the one used in subsequent analyses and referred to as CODE-15%"."""
ABNORMALITY_SPLIT = Split.TRAIN
"""Ribeiro et al. 2020 train on 98% of the CODE cohort and validate on 2%. The ``automatic-ecg-diagnosis``
README: "Part of the training data (the CODE-15% dataset) is openly available". Which of its exams fell in the
2% isn't published."""

_NN_PREDICTED_AGE = (
    "exams.csv's nn_predicted_age: the age the authors' network predicts from the exam's trace (Lima et al. 2021), "
    "for the gap between ECG-age and age."
)
_NORMAL_ECG = (
    "exams.csv's normal_ecg: whether the exam is normal by its clinical report and by the Glasgow program's "
    "measurements (Lima et al. 2021)."
)


@dataclass(frozen=True)
class Code15AgeTask(DatasetTask):
    """An exam's patient's age, which the authors' network predicts from the trace.

    ``train.py`` of ``ecg-age-prediction`` regresses ``exams.csv``'s ``age`` on every exam. The published
    prediction is a task annotation, never model input.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ScalarPredictionTask

    record: Record
    row: ExamRow

    @property
    def split(self) -> Split:
        """Return the task's split: test, as Lima et al. 2021 hold CODE-15% out."""
        return AGE_SPLIT

    def to_timenet_task(self) -> ScalarPredictionTask:
        """Return the scalar task, with the age in years as target."""
        task = ScalarPredictionTask(
            inputs=(self.record,),
            # A float with a unit keeps the type a regression metric needs.
            targets=(float(self.row.age),),
            unit=ureg.year,
            target_name="age",
            split=self.split,
        )
        task.annotate(
            Annotation(
                key=AnnotationKey.NN_PREDICTED_AGE,
                value=self.row.nn_predicted_age,
                unit=ureg.year,
                description=_NN_PREDICTED_AGE,
            )
        )
        return task


@dataclass(frozen=True)
class Code15AbnormalityTask(DatasetTask):
    """An exam's abnormalities among the six, which the authors' network scores with one sigmoid each.

    The target holds every abnormality the exam is labelled with, in the authors' output order, and is empty
    when it has none. ``normal_ecg`` is a task annotation, never model input.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    record: Record
    row: ExamRow

    @property
    def split(self) -> Split:
        """Return the task's split: train, as Ribeiro et al. 2020 train on the CODE cohort."""
        return ABNORMALITY_SPLIT

    def to_timenet_task(self) -> ClassificationTask:
        """Return the multi-label classification task."""
        task = ClassificationTask(
            inputs=(self.record,),
            targets=tuple(abnormality.value for abnormality in self.row.abnormalities),
            target_schema=AnnotationKey.ABNORMALITY_VOCABULARY,
            split=self.split,
        )
        task.annotate(Annotation(key=AnnotationKey.NORMAL_ECG, value=self.row.normal_ecg, description=_NORMAL_ECG))
        return task
