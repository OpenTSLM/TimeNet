"""One Record per exam: its 12 leads as lazy signals in millivolts, without padding, and the patient's facts."""

from functools import lru_cache, partial

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet_connectors.datasets.zenodo.code_15.index import Exam
from timenet_connectors.datasets.zenodo.code_15.keys import AnnotationKey
from timenet_connectors.datasets.zenodo.code_15.preprocessing import Padding, to_millivolts
from timenet_connectors.datasets.zenodo.code_15.release import SAMPLE_RATE_HZ
from timenet_connectors.datasets.zenodo.code_15.tracings import TraceReader, TraceScan, zero_runs
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Lead
from timenet_connectors.metadata import RecordMetadata


SOURCE_NAME = "12-lead ECG"

# The release stores float32, and halving it to millivolts is exact.
_SPEC = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt, dtype="float32")
# One axis for every Record: RegularAxis is frozen.
_TIME_AXIS = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)

_SEX = "The patient's sex, decoded from exams.csv's is_male."
_DEATH = "Whether the patient died in the follow-up. exams.csv states it on at most one exam per patient."
_FOLLOW_UP = (
    "exams.csv's timey: the years to the patient's death, or to the end of follow-up if they survived. Stated on "
    "the same exam as death."
)


class Code15RecordMetadata(RecordMetadata):
    """A CODE-15% Record's metadata: the exam, and where its trace sits in the published one.

    The patient id is the shared ``subject_id``. ``n_pad_before`` and ``n_pad_after`` count the zeros the
    Record's signals leave out, so ``preprocessing.authors_input`` can rebuild the published trace.
    """

    exam_id: int
    trace_file: str
    n_pad_before: int
    n_pad_after: int

    @property
    def padding(self) -> Padding:
        """Return the zeros the published trace has around the Record's samples."""
        return Padding(n_before=self.n_pad_before, n_after=self.n_pad_after)


def record_metadata(record: Record) -> Code15RecordMetadata:
    """Read a CODE-15% Record's typed metadata back.

    Returns:
        The metadata: the patient as ``subject_id``, the exam, and where its trace sits.

    Raises:
        TimeFValidationError: If the Record's metadata is not CODE-15% metadata.
    """  # noqa: DOC502 - raised by from_json
    return Code15RecordMetadata.from_json(record.metadata)


class LeadLoader:
    """Load one lead of one exam lazily, rereading each trace once for all its leads.

    A loader serves the Records of one load. A Record's signals load one after another, so one cached trace
    suffices.
    """

    def __init__(self) -> None:
        self._reader = TraceReader()
        self._millivolts = lru_cache(maxsize=1)(self._read_millivolts)

    def _read_millivolts(self, scan: TraceScan) -> np.ndarray:
        """Reread a trace, check it is the one scanned, and return its samples in millivolts.

        Returns:
            The ``(n, 12)`` float32 samples, without padding.

        Raises:
            TimeFFormatError: If the trace's runs of zeros differ from the scan's, or it holds a value that isn't
                finite: the file changed since it was scanned.
        """
        location = scan.location
        trace = self._reader.read(location)
        leading, trailing = zero_runs(trace[np.newaxis])
        same_zeros = (int(leading[0]), int(trailing[0])) == (scan.leading_zeros, scan.trailing_zeros)
        if not same_zeros or not np.isfinite(trace).all():
            raise TimeFFormatError(f"{location.path}: row {location.row} changed since it was scanned")
        return to_millivolts(trace, scan.padding)

    def __call__(self, exam: Exam, lead: Lead) -> pa.Array:
        """Return one lead of an exam.

        Returns:
            The lead's float32 values in millivolts.
        """
        millivolts = self._millivolts(exam.scan)
        return pa.array(np.ascontiguousarray(millivolts[:, lead.column]), type=pa.float32())


def build_record(exam: Exam, leads: LeadLoader) -> Record:
    """Build an exam's Record: one Source of 12 lazy leads, and the patient's sex and mortality.

    The age and the labels are the tasks' targets, so they stay off the Record.

    Args:
        exam: The exam and its trace.
        leads: The loader the signals read their values through.

    Returns:
        The Record, whose values stay in the part until read.
    """
    padding = exam.scan.padding
    signals = tuple(
        Signal.from_loader(
            name=lead.value,
            spec=_SPEC,
            time_axis=_TIME_AXIS,
            n_values=padding.n_values,
            loader=partial(leads, exam, lead),
        )
        for lead in Lead
    )
    row = exam.row
    metadata = Code15RecordMetadata(
        subject_id=str(row.patient_id),
        exam_id=row.exam_id,
        trace_file=row.trace_file,
        n_pad_before=padding.n_before,
        n_pad_after=padding.n_after,
    )
    record = Record(sources=(Source(name=SOURCE_NAME, signals=signals),), metadata=metadata.to_json())
    record.annotate(Annotation(key=AnnotationKey.SEX, value=row.sex.value, description=_SEX))
    if row.death is not None and row.timey is not None:
        record.annotate(Annotation(key=AnnotationKey.DEATH, value=row.death, description=_DEATH))
        record.annotate(
            Annotation(key=AnnotationKey.FOLLOW_UP, value=row.timey, unit=ureg.year, description=_FOLLOW_UP)
        )
    return record
