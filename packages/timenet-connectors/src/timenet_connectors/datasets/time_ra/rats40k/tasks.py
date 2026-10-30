"""RATS40K's one task per row: name the series' anomaly category, built as a TimeNet task."""

from typing import Any

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask
from timenet_connectors.datasets.time_ra.rats40k.keys import AnnotationKey, category_vocabulary
from timenet_connectors.datasets.time_ra.rats40k.request import CATEGORY_NAMES, TASK_PROMPTS
from timenet_connectors.datasets.time_ra.rats40k.rows import Row


_LABEL = "The release's Label, Normal or Anomaly, as published; it disagrees with the target in 1,184 rows."
_ACTION_ID = "The release's ActionID, as published; it doesn't always follow the prompt's numbering."


def category_task(row: Row, record: Record) -> ClassificationTask:
    """Return one row's task: answer its category name, as the authors' evaluation prompt asks.

    The authors ask one question per row and score detection, category and reasoning from one answer,
    so one task carries all three: the name as target, the ``Thought`` as rationale, and ``Label`` and
    ``ActionID`` as task annotations, which are never model input.

    Args:
        row: The row, whose split is the authors' split.
        record: The Record that holds the row's series.

    Returns:
        The classification task, with the published name as target and the Thought as rationale.
    """
    metadata: dict[str, Any] = {
        # False for the univariate test rows labelled with multivariate names.
        "target_in_vocabulary": row.action in CATEGORY_NAMES[row.ref.variant],
        # In univariate test, a .pdf path marks the rows where gpt-4o ranks first, so it stays off the
        # input Record.
        "figure_path": row.figure_path,
    }
    if row.rank is not None:
        metadata["Rank"] = row.rank
    task = ClassificationTask(
        inputs=(record,),
        prompt=TASK_PROMPTS[row.ref.variant],
        targets=(row.action,),
        rationale=row.thought,
        split=row.ref.split,
        input_annotations=tuple(a for a in record.annotations if a.key == AnnotationKey.SOURCE_DOMAIN),
        target_schema=category_vocabulary(row.ref.variant),
        metadata=metadata,
    )
    # The authors score detection and category codes against these. Task annotations, never input annotations.
    task.annotate(Annotation(key=AnnotationKey.LABEL, value=row.label, description=_LABEL))
    task.annotate(Annotation(key=AnnotationKey.ACTION_ID, value=row.action_id, description=_ACTION_ID))
    return task
