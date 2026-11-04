"""Read the retrieval files: each row is one window's generated description and, for an event, its report.

``RetrievalIndex.read`` checks every row against the indexed windows (pass one) and keeps only which
window each row names. ``RetrievalIndex.rows`` rereads the files for the task stream (pass two), one
batch at a time, and checks the rows did not change.
"""

from collections.abc import Iterator, Sequence
import hashlib
import json
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet_connectors.datasets.trace.weather_events.archive import SplitName, TraceFile
from timenet_connectors.datasets.trace.weather_events.fields import (
    as_object,
    event_label,
    exact_int,
    field,
    text,
    text_field,
)
from timenet_connectors.datasets.trace.weather_events.measurements import Window, WindowIndex
from timenet_connectors.datasets.trace.weather_events.vocabulary import EVENT_ID, Channel, EventLabel


_COLUMNS = ["id", "description", "events"]
"""The retrieval columns read. The ``timeseries`` column copies a window's series, which a Record holds."""
_BATCH_ROWS = 1024
# generate_dsp joins the channel sentences with this spacing.
_SENTENCE_JOINER = " \n "


class Report(NamedTuple):
    """One event window's report, as the retrieval files publish it."""

    event_id: int
    event_type: EventLabel
    narrative: str

    @property
    def text(self) -> str:
        """The report as the authors' ``generate_er`` writes it for their text encoder."""
        return f"The weather event is {self.event_type.text}. {self.narrative}"


class RetrievalRow(NamedTuple):
    """One row of a retrieval file: a window's generated description and, for an event window, its report."""

    window_id: int
    prompt: str
    """The description as the authors' ``generate_dsp`` composes it for their text encoder."""
    sentences: tuple[str, ...]
    """The generated sentence about each channel, in channel order."""
    report: Report | None
    """The NOAA report of the event that followed the window; ``None`` for a window without one."""
    digest: bytes
    """A digest of the published row, to tell a reread of the same row from another."""

    @classmethod
    def parse(cls, published: object) -> "RetrievalRow":
        """Check one published retrieval row.

        The row publishes its description under the window JSON's keys, but every value is text the authors
        had ChatGPT write from the window (paper Appendix B.4). Its ``DATE`` and ``location`` are sentences
        about the window's dates and place, not timestamps or a county.

        Returns:
            The row, with its prompt and channel sentences composed and its report read.
        """
        published = as_object(published)
        description = as_object(field(published, "description"))
        sentences = tuple(text_field(description, channel.value) for channel in Channel)
        prompt = (
            f"Weather time series location: {text_field(description, 'location')} "
            f"Time range: {text_field(description, 'DATE')} "
            f"The weather is {text_field(description, 'labels')}. {_SENTENCE_JOINER.join(sentences)}"
        )
        events = field(published, "events")
        report = None
        if events is not None:
            events = as_object(events)
            report = Report(
                event_id=exact_int(field(events, EVENT_ID), EVENT_ID),
                event_type=event_label(field(events, "event_type"), "event_type"),
                narrative=text(field(events, "narrative"), "narrative"),
            )
        digest = hashlib.blake2b(json.dumps(published, sort_keys=True).encode(), digest_size=16).digest()
        return cls(exact_int(field(published, "id"), "id"), prompt, sentences, report, digest)

    def check_names(self, window: Window) -> None:
        """Check that the row has an event report exactly when its window is classified, and the right one.

        Raises:
            TimeFValidationError: If the window's event report is missing, mismatched or unexpected.
        """
        if window.is_classified:
            if self.report is None or self.report.event_type != window.label:
                raise TimeFValidationError(
                    f"{window.where}: no event report with event_type {window.label.name if window.label else None}"
                )
        elif self.report is not None:
            raise TimeFValidationError(f"{window.where}: event report for a window without an event-type task")


def _iter_rows(file: TraceFile) -> Iterator[RetrievalRow]:
    """Read and check a retrieval file one batch at a time, so no pass holds a whole file's descriptions.

    Yields:
        Each row, in file order.

    Raises:
        TimeFValidationError: If a row is invalid, naming the file and the row's id.
        TimeFFormatError: If the file cannot be read as parquet with the columns the connector needs.
    """
    try:
        with pq.ParquetFile(file.path) as parquet:
            for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=_COLUMNS):
                for published in batch.to_pylist():
                    try:
                        yield RetrievalRow.parse(published)
                    except TimeFValidationError as exc:
                        raise TimeFValidationError(f"{file.path} (id {published['id']!r}): {exc}") from exc
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(f"{file.path}: {exc}") from exc


class _Named(NamedTuple):
    """The window a retrieval row names, and the digest the row had when it was checked."""

    window: Window
    digest: bytes


class RetrievalIndex:
    """Which window each retrieval row names, file by file, checked against the window index.

    It keeps no descriptions or reports: iterating rereads the files, so the task stream holds one batch
    of descriptions at a time.
    """

    def __init__(self, named: dict[TraceFile, list[_Named]]) -> None:
        self._named = named

    @classmethod
    def read(cls, files: Sequence[TraceFile], windows: WindowIndex) -> "RetrievalIndex":
        """Check every row of the retrieval files against the indexed windows.

        Every row must name a window, once; every classified window must have its event report, and
        every report a classified window. The task stream rereads the rows and relies on these checks.

        Returns:
            The index.

        Raises:
            TimeFValidationError: If a row or a report is invalid.
        """
        named: dict[TraceFile, list[_Named]] = {}
        seen: set[int] = set()
        orphans: list[int] = []
        for file in files:
            named[file] = []
            for row in _iter_rows(file):
                if row.window_id in seen:
                    raise TimeFValidationError(f"{file.path}: window id {row.window_id} appears in more than one row")
                seen.add(row.window_id)
                window = windows.get(row.window_id)
                if window is None:
                    orphans.append(row.window_id)
                    continue
                row.check_names(window)
                named[file].append(_Named(window, row.digest))
        if orphans:
            raise TimeFValidationError(f"{len(orphans)} retrieval rows have no window, such as {min(orphans)}")
        for window in windows.windows.values():
            if window.is_classified and window.window_id not in seen:
                raise TimeFValidationError(f"{window.where}: classified window has no retrieval row")
        return cls(named)

    def test_windows(self) -> list[Window]:
        """Return the windows the retrieval test file's rows name, in file order.

        They come from every window file, train and val included, since the retrieval split is the
        authors' own.
        """
        return [item.window for file, items in self._named.items() if file.split is SplitName.TEST for item in items]

    def rows(self) -> Iterator[tuple[TraceFile, Window, RetrievalRow]]:
        """Reread the checked retrieval files, one batch at a time.

        Yields:
            Each row's file, the window it names, and the row, in file order.

        Raises:
            TimeFFormatError: If a file no longer holds the rows that were checked.
        """
        for file, items in self._named.items():
            rows = _iter_rows(file)
            for item in items:
                row = self._next_row(rows, file)
                if row is None or row.window_id != item.window.window_id or row.digest != item.digest:
                    raise TimeFFormatError(f"{file.path}: changed since it was checked, at window id {item.window.key}")
                yield file, item.window, row
            if self._next_row(rows, file) is not None:
                raise TimeFFormatError(f"{file.path}: changed since it was checked, it has more rows")

    @staticmethod
    def _next_row(rows: Iterator[RetrievalRow], file: TraceFile) -> RetrievalRow | None:
        try:
            return next(rows, None)
        except TimeFValidationError as exc:
            raise TimeFFormatError(f"{file.path}: changed since it was checked: {exc}") from exc
