"""The MIT-BIH tasks: what the authors' comparators score, and the classification of each reference beat.

The authors evaluate an algorithm on a whole record: it writes an annotation file, and the WFDB comparators
match it against the reference. ``bxb`` matches beats, with their labels, and ``epicmp`` matches episodes of
atrial fibrillation, atrial flutter and ventricular flutter, which the rhythm labels state. Each record has
one task for each. Neither comparator scores a beat given its reference location; the beat classification
tasks are TimeNet's, for the beat classification PhysioNet's usage notes list.

The release has no split, so no task has one, and asks no question in words, so no task has a prompt.
"""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    LocalizationMode,
    Task,
    TemporalLocalizationTask,
    TimePoint,
)
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.physionet.mitdb.keys import AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.vocabulary import BeatCode


class MitdbTask(DatasetTask):
    """A task of the MIT-BIH Arrhythmia Database, which has no split."""

    @property
    def split(self) -> None:
        """Return ``None``: the release has no split."""


@dataclass(frozen=True)
class BeatAnnotationTask(MitdbTask):
    """Find and label every beat of a record, the annotation file the authors' ``bxb`` compares.

    The target is the Record's beat annotations: each a point at the fiducial, valued by its code. ``bxb``
    scores both: a beat matches within 150 ms, and the matched labels are compared in the AAMI classes.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = TemporalLocalizationTask

    record: Record
    beats: tuple[Annotation, ...]

    def to_timenet_task(self) -> TemporalLocalizationTask:
        """Return the localization task, whose target is every beat annotation of the Record."""
        # Exhaustive: the cardiologists added every beat the detector missed.
        return TemporalLocalizationTask(
            inputs=(self.record,), target_annotations=self.beats, mode=LocalizationMode.EXHAUSTIVE
        )


@dataclass(frozen=True)
class RhythmAnnotationTask(MitdbTask):
    """Divide a record into rhythm episodes, from which the authors' ``epicmp`` takes the episodes it compares.

    The target is the Record's rhythm annotations: each an interval from a rhythm change to the next, valued
    by its label. ``epicmp`` compares the episodes of atrial fibrillation and atrial flutter.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = TemporalLocalizationTask

    record: Record
    rhythms: tuple[Annotation, ...]

    def to_timenet_task(self) -> TemporalLocalizationTask:
        """Return the localization task, whose target is every rhythm annotation of the Record."""
        # Exhaustive: the episodes cover the record from its first rhythm change to its end.
        return TemporalLocalizationTask(
            inputs=(self.record,), target_annotations=self.rhythms, mode=LocalizationMode.EXHAUSTIVE
        )


@dataclass(frozen=True)
class BeatClassificationTask(MitdbTask):
    """Name the code of one reference beat, given its fiducial point.

    The scope is the beat's point and names no signal, since the cardiologists read both leads. The whole
    Record stays available, as the beat's neighbours and the rhythm are what tell many codes apart.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    record: Record
    at_us: int
    """The fiducial point's time from the record's first sample."""
    code: BeatCode

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the beat's code as target."""
        return ClassificationTask(
            inputs=(self.record,),
            scope=TimePoint.micros(self.at_us),
            targets=(self.code.value,),
            target_schema=AnnotationKey.BEAT_VOCABULARY,
        )
