"""Build CODE-15 records and patient annotations from checked exams."""

from timenet.dataset import Record, Signal, SignalChannel, Source
from timenet.types import (
    Annotation,
    ureg,
)
from timenet_connectors.datasets.zenodo.code_15.source import Exam, TraceReader
from timenet_connectors.datasets.zenodo.code_15.vocabulary import (
    DEATH_NOTE,
    FOLLOW_UP_NOTE,
    SEX_NOTE,
    SOURCE_NAME,
    SPEC,
    TIME_AXIS,
    AnnotationKey,
    Lead,
    MetadataKey,
)
from timenet_connectors.demographics import sex_annotation, subject_ids_annotation


def build_record(exam: Exam, reader: TraceReader) -> Record:
    """Build an exam's Record: one Source of 12 lazy leads, and the patient's sex and mortality.

    The age and the labels are the tasks' targets, so they stay off the Record.

    Returns:
        The Record, whose values stay in the part until read.
    """
    row, padding = exam.row, exam.padding
    signals = Signal.from_shared_loader(
        loader=reader.shared,
        recording_key=exam,
        channels=tuple(
            SignalChannel(name=lead.value, spec=SPEC, time_axis=TIME_AXIS, n_values=padding.n_values) for lead in Lead
        ),
    )
    record = Record(
        sources=(Source(name=SOURCE_NAME, signals=signals),),
        metadata={
            MetadataKey.EXAM_ID: row.exam_id,
            MetadataKey.TRACE_FILE: row.trace_file,
            MetadataKey.N_PAD_BEFORE: padding.n_before,
            MetadataKey.N_PAD_AFTER: padding.n_after,
        },
    )
    record.add_annotations(
        (subject_ids_annotation([str(row.patient_id)]), sex_annotation(row.sex, description=SEX_NOTE))
    )
    if row.mortality is not None:
        record.annotate(Annotation(key=AnnotationKey.DEATH, value=row.mortality.died, description=DEATH_NOTE))
        record.annotate(
            Annotation(
                key=AnnotationKey.FOLLOW_UP, value=row.mortality.years, unit=ureg.year, description=FOLLOW_UP_NOTE
            )
        )
    return record
