"""Build repeatable CODE-15 age and abnormality tasks from checked exams."""

from collections.abc import Iterator

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    ScalarPredictionTask,
    Task,
    ureg,
)
from timenet_connectors.datasets.zenodo.code_15.source import Exam, ExamRow
from timenet_connectors.datasets.zenodo.code_15.vocabulary import (
    ABNORMAL_UNSPECIFIED_LABEL,
    ABNORMALITIES,
    ABNORMALITY_SPLIT,
    AGE_SPLIT,
    NN_PREDICTED_AGE_NOTE,
    NORMAL_ECG_LABEL,
    AnnotationKey,
)


def _age_task(record: Record, row: ExamRow) -> ScalarPredictionTask:
    """Return an exam's age task: its patient's age, as Lima et al. 2021 predict it from the trace.

    The authors' published prediction is a task annotation, never model input.

    Returns:
        The scalar task, with the age in years as target.
    """
    task = ScalarPredictionTask(
        inputs=(record,),
        # A float with a unit keeps the type a regression metric needs.
        targets=(float(row.age),),
        unit=ureg.year,
        target_name="age",
        split=AGE_SPLIT,
    )
    task.annotate(
        Annotation(
            key=AnnotationKey.NN_PREDICTED_AGE,
            value=row.nn_predicted_age,
            unit=ureg.year,
            description=NN_PREDICTED_AGE_NOTE,
        )
    )
    return task


def _abnormality_targets(row: ExamRow) -> tuple[str, ...]:
    """Return the labels of an exam's abnormality task.

    Returns:
        Every abnormality the exam is labelled with, in the authors' output order. An exam marked normal has only
        ``normal_ecg``; an exam with all seven columns false has only ``abnormal_unspecified``.
    """
    if row.normal_ecg:
        return (NORMAL_ECG_LABEL.name,)
    if not row.abnormalities:
        return (ABNORMAL_UNSPECIFIED_LABEL.name,)
    return tuple(abnormality.value for abnormality in row.abnormalities)


def _abnormality_task(record: Record, row: ExamRow) -> ClassificationTask:
    """Return the connector's eight-label task extending Ribeiro et al. 2020's six diagnosis classes.

    Returns:
        The multi-label classification task, whose targets ``_abnormality_targets`` gives.
    """
    return ClassificationTask(
        inputs=(record,),
        targets=_abnormality_targets(row),
        label_set=ABNORMALITIES,
        split=ABNORMALITY_SPLIT,
    )


def iter_tasks(exams: list[Exam], records: list[Record]) -> Iterator[Task]:
    """Yield age tasks, then abnormality tasks, in the checked exam order."""
    for exam, record in zip(exams, records, strict=True):
        yield _age_task(record, exam.row)
    for exam, record in zip(exams, records, strict=True):
        yield _abnormality_task(record, exam.row)
