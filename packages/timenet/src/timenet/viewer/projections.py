"""HTTP-friendly encoding of lightweight inspection pages."""

from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.schemas import RecordQuery, SignalQuery, SourceQuery


def session(inspection: ViewerInspection) -> dict[str, object]:
    """Return the public session capabilities."""
    return {"api_version": 1, "dataset_id": inspection.dataset_id, "limits": {"page": 200}}


def records(inspection: ViewerInspection, body: RecordQuery) -> dict[str, object]:
    """Return a bounded record-ID page."""
    page = inspection.records_page(body.query, body.after, body.limit)
    return {
        "api_version": 1,
        "items": [
            {
                "record_id": item.record_id,
                "start_us": None if item.start_us is None else str(item.start_us),
                "end_us": None if item.end_us is None else str(item.end_us),
            }
            for item in page.items
        ],
        "next_cursor": page.next_after,
    }


def sources(inspection: ViewerInspection, body: SourceQuery) -> dict[str, object]:
    """Return immediate sources for a selected record."""
    page = inspection.sources_page(body.record_id, body.parent_id, body.after, body.limit)
    return {"api_version": 1, "items": [item.__dict__ for item in page.items], "next_cursor": page.next_after}


def signals(inspection: ViewerInspection, body: SignalQuery) -> dict[str, object]:
    """Return direct signals for a selected source."""
    page = inspection.signals_page(body.record_id, body.source_id, body.after, body.limit)
    return {"api_version": 1, "items": [item.__dict__ for item in page.items], "next_cursor": page.next_after}
