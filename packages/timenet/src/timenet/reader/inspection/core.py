"""Public bounded inspection facade for TimeF readers."""

from __future__ import annotations

from timenet.reader.inspection.annotations import browse_annotations, window_annotations
from timenet.reader.inspection.details import overview, owner_record, record_detail
from timenet.reader.inspection.queries import ReaderQueries
from timenet.reader.inspection.requests import (
    AnnotationQuery,
    AnnotationWindowQuery,
    OwnerRecordQuery,
    RecordDetailQuery,
    TaskDetailQuery,
    TaskQuery,
)
from timenet.reader.inspection.results import AnnotationPage, Overview, OwnerRecord, RecordDetail, TaskDetail, TaskPage
from timenet.reader.inspection.tasks import task_detail, tasks


class ReaderInspection(ReaderQueries):
    """Expose bounded facts without hydrating the whole dataset or importing HTTP code."""

    def overview(self) -> Overview:
        """Return manifest-backed overview facts without reading signal values."""
        return overview(self._reader)

    def record_detail(self, body: RecordDetailQuery) -> RecordDetail:
        """Return persisted clock, span provenance and bounded metadata."""
        return record_detail(self._reader, body)

    def owner_record(self, body: OwnerRecordQuery) -> OwnerRecord | None:
        """Return the containing Record for a Source or Signal owner."""
        return owner_record(self._reader, body)

    def tasks(self, body: TaskQuery) -> TaskPage:
        """Return a bounded, lightweight task page."""
        return tasks(self._reader, body)

    def task_detail(self, body: TaskDetailQuery) -> TaskDetail | None:
        """Return bounded task relationships for the selected task."""
        return task_detail(self._reader, body)

    def browse_annotations(self, body: AnnotationQuery) -> AnnotationPage:
        """Return a bounded annotation projection."""
        return browse_annotations(self._reader, body)

    def window_annotations(self, body: AnnotationWindowQuery) -> AnnotationPage:
        """Return a bounded annotation projection."""
        return window_annotations(self._reader, body)
