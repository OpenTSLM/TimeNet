"""The names VeriTime writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique

from timenet_connectors.datasets.hayleyzhou1113.veritime.release import SourceTask


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"
    STEP1_LABEL = "step1_label"
    """A task's process label for step 1, the task intent."""
    STEP2_LABEL = "step2_label"
    """A task's process label for step 2, the key patterns, separated by semicolons."""
    STEP4_LABEL = "step4_label"
    """A task's process label for step 4, the preliminary answer."""
    STEP6_LABEL = "step6_label"
    """A task's process label for step 6, the final answer as the CoT states it."""
    SOURCE_ANSWER = "source_answer"
    """A scenario task's ChatTS answer, from which the authors extracted its label."""


def class_vocabulary(task: SourceTask) -> str:
    """Give the key of the dataset annotation that holds a classification task's classes.

    Returns:
        The key, such as ``ecg_vocabulary``, which each task of the kind states as its ``target_schema``.
    """
    return f"{task.slug}_vocabulary"
