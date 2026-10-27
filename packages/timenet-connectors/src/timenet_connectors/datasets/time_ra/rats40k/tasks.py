"""RATS40K's one task per row: name the series' anomaly category, built as the TimeNet task it becomes."""

from dataclasses import dataclass
from typing import Any, ClassVar

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, Split, Task
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.time_ra.rats40k._source import Row
from timenet_connectors.datasets.time_ra.rats40k.keys import AnnotationKey, category_vocabulary
from timenet_connectors.datasets.time_ra.rats40k.request import CATEGORY_NAMES, TASK_PROMPTS


_LABEL = "The release's Label, Normal or Anomaly, as published; it disagrees with the target in 1,184 rows."
_ACTION_ID = "The release's ActionID, as published; it doesn't always follow the prompt's numbering."


@dataclass(frozen=True)
class RATS40KCategoryTask(DatasetTask):
    """One row's task: answer its category name, as the authors' evaluation prompt asks.

    The authors ask one question per row and score detection, category and reasoning from one answer,
    so one task carries all three: the name as target, the ``Thought`` as rationale, and ``Label`` and
    ``ActionID`` as task annotations, which are never model input.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    row: Row
    record: Record

    @property
    def split(self) -> Split:
        """Return the task's split: the authors' split of the row."""
        return self.row.ref.split

    @property
    def target_in_vocabulary(self) -> bool:
        """Return whether the target is a name the variant's prompt offers.

        False for the univariate test rows labelled with multivariate names.
        """
        return self.row.action in CATEGORY_NAMES[self.row.ref.variant]

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the published name as target and the Thought as rationale."""
        metadata: dict[str, Any] = {
            "target_in_vocabulary": self.target_in_vocabulary,
            # In univariate test, a .pdf path marks the rows where gpt-4o ranks first, so it stays off the
            # input Record.
            "figure_path": self.row.figure_path,
        }
        if self.row.rank is not None:
            metadata["Rank"] = self.row.rank
        task = ClassificationTask(
            inputs=(self.record,),
            prompt=TASK_PROMPTS[self.row.ref.variant],
            targets=(self.row.action,),
            rationale=self.row.thought,
            split=self.split,
            input_annotations=tuple(a for a in self.record.annotations if a.key == AnnotationKey.SOURCE_DOMAIN),
            target_schema=category_vocabulary(self.row.ref.variant),
            metadata=metadata,
        )
        # The authors score detection and category codes against these. Task annotations, never input annotations.
        task.annotate(Annotation(key=AnnotationKey.LABEL, value=self.row.label, description=_LABEL))
        task.annotate(Annotation(key=AnnotationKey.ACTION_ID, value=self.row.action_id, description=_ACTION_ID))
        return task
