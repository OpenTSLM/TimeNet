"""The MIT-BIH tasks: what the authors' comparators score, and the classification of each reference beat.

The authors evaluate an algorithm on a whole record: it writes an annotation file, and the WFDB comparators
match it against the reference. ``bxb`` matches beats, with their labels, and ``epicmp`` matches episodes of
atrial fibrillation, atrial flutter and ventricular flutter, which the rhythm labels state. Each record has
one task for each. Neither comparator scores a beat given its reference location; the beat classification
tasks are TimeNet's, for the beat classification PhysioNet's usage notes list.

The release has no split, so no task has one, and asks no question in words, so no task has a prompt.
"""

from timenet.dataset import Record
from timenet.types import (
    Annotation,
    ClassificationTask,
    LocalizationMode,
    TemporalLocalizationTask,
    TimePoint,
)
from timenet_connectors.datasets.physionet.mitdb.keys import AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.vocabulary import BeatCode


def beat_annotation_task(record: Record, beats: tuple[Annotation, ...]) -> TemporalLocalizationTask:
    """Return the task that finds and labels every beat of a record, the annotation file ``bxb`` compares.

    ``bxb`` scores both: a beat matches within 150 ms, and the matched labels are compared in the AAMI classes.

    Args:
        record: The record's Record.
        beats: The Record's beat annotations: each a point at the fiducial, valued by its code.

    Returns:
        The localization task, whose target is every beat annotation of the Record.
    """
    # Exhaustive: the cardiologists added every beat the detector missed.
    return TemporalLocalizationTask(inputs=(record,), target_annotations=beats, mode=LocalizationMode.EXHAUSTIVE)


def rhythm_annotation_task(record: Record, rhythms: tuple[Annotation, ...]) -> TemporalLocalizationTask:
    """Return the task that divides a record into rhythm episodes, from which ``epicmp`` takes its episodes.

    ``epicmp`` compares the episodes of atrial fibrillation and atrial flutter.

    Args:
        record: The record's Record.
        rhythms: The Record's rhythm annotations: each an interval from a rhythm change to the next, valued
            by its label.

    Returns:
        The localization task, whose target is every rhythm annotation of the Record.
    """
    # Exhaustive: the episodes cover the record from its first rhythm change to its end.
    return TemporalLocalizationTask(inputs=(record,), target_annotations=rhythms, mode=LocalizationMode.EXHAUSTIVE)


def beat_classification_task(record: Record, at_us: int, code: BeatCode) -> ClassificationTask:
    """Return the task that names the code of one reference beat, given its fiducial point.

    The scope is the beat's point and names no signal, since the cardiologists read both leads. The whole
    Record stays available, as the beat's neighbours and the rhythm are what tell many codes apart.

    Args:
        record: The beat's Record.
        at_us: The fiducial point's time from the record's first sample.
        code: The beat's reference code.

    Returns:
        The classification task, with the beat's code as target.
    """
    return ClassificationTask(
        inputs=(record,),
        scope=TimePoint.micros(at_us),
        targets=(code.value,),
        target_schema=AnnotationKey.BEAT_VOCABULARY,
    )
