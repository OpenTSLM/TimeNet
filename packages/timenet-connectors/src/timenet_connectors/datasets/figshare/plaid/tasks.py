"""PLAID's two tasks, the two uses the paper names, each built as a TimeNet task.

Neither has a split. The authors define none: their GlobalSIP notebook tests each household in turn,
training on the rest, and a Record's ``subject_id`` names its household. Neither asks a question in words,
so neither has a prompt.
"""

from timenet.dataset import Record
from timenet.types import Annotation, ClassificationTask, LocalizationMode, TemporalLocalizationTask
from timenet_connectors.datasets.figshare.plaid.keys import AnnotationKey


def appliance_task(record: Record, appliance_type: str) -> ClassificationTask:
    """Return a submetered file's task: name the appliance type from its current and voltage.

    The authors' GlobalSIP notebook classifies the type from features of each file's last 10,000 samples,
    in steady state. The task covers the whole file, so the start-up transient the paper describes is in
    the input too.

    Args:
        record: The Record of the submetered file.
        appliance_type: The file's appliance type, as the metadata writes it.

    Returns:
        The classification task, with the type as target.
    """
    return ClassificationTask(
        inputs=(record,),
        targets=(appliance_type,),
        target_schema=AnnotationKey.APPLIANCE_TYPE_VOCABULARY,
    )


def switching_events_task(record: Record, events: tuple[Annotation, ...]) -> TemporalLocalizationTask:
    """Return an aggregated file's task: find when each appliance switched on and off, and which it was.

    These are the two steps of load monitoring the paper names, event detection and load identification, so
    one task asks both. Its targets are the Record's labelled events: each a point, keyed on or off, whose
    value is the appliance type.

    The mode is sparse: the labels need not mark every event. In 126 files the measurer's status string has
    one more ``on`` than the labelled events.

    Args:
        record: The Record of the aggregated file.
        events: The Record's event annotations, in event order.

    Returns:
        The localization task, with the Record's event annotations as targets.
    """
    return TemporalLocalizationTask(
        inputs=(record,),
        target_annotations=events,
        mode=LocalizationMode.SPARSE,
    )
