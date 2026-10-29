"""PLAID's two tasks, the two uses the paper names, each built as the TimeNet task it becomes.

Neither has a split. The authors define none: their GlobalSIP notebook tests each household in turn,
training on the rest, and a Record's ``subject_id`` names its household. Neither asks a question in words,
so neither has a prompt.
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    LocalizationMode,
    Split,
    Task,
    TemporalLocalizationTask,
)
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.figshare.plaid.keys import AnnotationKey


@dataclass(frozen=True)
class PlaidTask(DatasetTask):
    """One task on the Record of one CSV file."""

    record: Record

    @property
    def split(self) -> Split | None:
        """Return no split: the authors leave one household out at a time instead."""
        return None


@dataclass(frozen=True)
class PlaidApplianceTask(PlaidTask):
    """A submetered file's task: name the appliance type from its current and voltage.

    The authors' GlobalSIP notebook classifies the type from features of each file's last 10,000 samples,
    in steady state. The task covers the whole file, so the start-up transient the paper describes is in
    the input too.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    appliance_type: str

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the type as the metadata writes it as target."""
        return ClassificationTask(
            inputs=(self.record,),
            targets=(self.appliance_type,),
            target_schema=AnnotationKey.APPLIANCE_TYPE_VOCABULARY,
        )


@dataclass(frozen=True)
class PlaidSwitchingEventsTask(PlaidTask):
    """An aggregated file's task: find when each appliance switched on and off, and which appliance it was.

    These are the two steps of load monitoring the paper names, event detection and load identification, so
    one task asks both. Its targets are the Record's labelled events: each a point, keyed on or off, whose
    value is the appliance type.

    The mode is sparse: the labels need not mark every event. In 126 files the measurer's status string has
    one more ``on`` than the labelled events.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = TemporalLocalizationTask

    events: tuple[Annotation, ...]
    """The Record's event annotations, in event order."""

    def to_timenet_task(self) -> TemporalLocalizationTask:
        """Return the localization task, with the Record's event annotations as targets."""
        return TemporalLocalizationTask(
            inputs=(self.record,),
            target_annotations=self.events,
            mode=LocalizationMode.SPARSE,
        )
