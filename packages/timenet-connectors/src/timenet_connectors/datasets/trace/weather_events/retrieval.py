"""Read the retrieval files: each row is one window's generated description and, for an event, its report.

``RetrievalIndex.read`` checks every row against the indexed windows (pass one) and keeps only which
window each row names. ``RetrievalIndex.rows`` rereads the files for the task stream (pass two) and
checks the rows did not change.
"""

from collections.abc import Iterator, Sequence
import hashlib
import json
from typing import NamedTuple

import pyarrow as pa

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
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel, EventLabel, SourceField
from timenet_connectors.sources.parquet import iter_parquet_rows


_COLUMNS = [SourceField.ID.value, SourceField.DESCRIPTION.value, SourceField.EVENTS.value]
"""The retrieval columns read. The ``timeseries`` column copies a window's series, which a Record holds."""
# generate_dsp joins the channel sentences with this spacing.
_SENTENCE_JOINER = " \n "
# 16 bytes of BLAKE2b tell a changed row from the checked one; the digest is no security check.
_ROW_DIGEST_BYTES = 16


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
        description = as_object(field(published, SourceField.DESCRIPTION))
        sentences = tuple(text_field(description, channel.value) for channel in Channel)
        prompt = (
            f"Weather time series location: {text_field(description, SourceField.LOCATION)} "
            f"Time range: {text_field(description, SourceField.DATE)} "
            f"The weather is {text_field(description, SourceField.LABELS)}. {_SENTENCE_JOINER.join(sentences)}"
        )
        events = field(published, SourceField.EVENTS)
        report = None
        if events is not None:
            events = as_object(events)
            report = Report(
                event_id=exact_int(field(events, SourceField.EVENT_ID), SourceField.EVENT_ID),
                event_type=event_label(field(events, SourceField.EVENT_TYPE), SourceField.EVENT_TYPE),
                narrative=text(field(events, SourceField.NARRATIVE), SourceField.NARRATIVE),
            )
        serialized = json.dumps(published, sort_keys=True).encode()
        digest = hashlib.blake2b(serialized, digest_size=_ROW_DIGEST_BYTES).digest()
        window_id = exact_int(field(published, SourceField.ID), SourceField.ID)
        return cls(window_id, prompt, sentences, report, digest)

    def check_report_matches(self, window: Window) -> None:
        """Check that the row has an event report exactly when its window is classified, and the one for its label.

        Raises:
            TimeFValidationError: If the window's event report is missing, mismatched or unexpected.
        """
        label = window.label
        if label is not None and window.is_classified:
            if self.report is None or self.report.event_type is not label:
                raise TimeFValidationError(f"{window.error_context}: no event report with event_type {label.name}")
        elif self.report is not None:
            raise TimeFValidationError(f"{window.error_context}: event report for a window without an event-type task")


def _iter_rows(file: TraceFile) -> Iterator[RetrievalRow]:
    """Read and check a retrieval file a batch at a time, so no pass holds a whole file's descriptions.

    Yields:
        Each row, in file order.

    Raises:
        TimeFValidationError: If a row is invalid, naming the file and the row's id.
        TimeFFormatError: If the file cannot be read as parquet, or lacks a column the connector reads.
    """
    try:
        for published in iter_parquet_rows(file.path, columns=_COLUMNS):
            try:
                yield RetrievalRow.parse(published)
            except TimeFValidationError as exc:
                raise TimeFValidationError(f"{file.path} (id {published.get(SourceField.ID)!r}): {exc}") from exc
    except (OSError, pa.ArrowException) as exc:
        raise TimeFFormatError(f"{file.path}: {exc}") from exc


class _IndexedRow(NamedTuple):
    """The window a retrieval row names, and the digest the row had when it was checked."""

    window: Window
    digest: bytes


class CheckedRow(NamedTuple):
    """A reread retrieval row that matches the row pass one checked, with its file and its window."""

    file: TraceFile
    window: Window
    row: RetrievalRow


class RetrievalIndex:
    """Which window each retrieval row names, file by file, checked against the window index.

    It keeps no descriptions or reports: iterating rereads the files, so the task stream holds one batch
    of descriptions at a time.
    """

    def __init__(self, indexed: dict[TraceFile, list[_IndexedRow]]) -> None:
        self._indexed = indexed

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
        indexed: dict[TraceFile, list[_IndexedRow]] = {}
        seen: set[int] = set()
        orphans: list[int] = []
        for file in files:
            indexed[file] = []
            for row in _iter_rows(file):
                if row.window_id in seen:
                    raise TimeFValidationError(f"{file.path}: window id {row.window_id} appears in more than one row")
                seen.add(row.window_id)
                window = windows.get(row.window_id)
                if window is None:
                    orphans.append(row.window_id)
                    continue
                row.check_report_matches(window)
                indexed[file].append(_IndexedRow(window, row.digest))
        if orphans:
            raise TimeFValidationError(f"{len(orphans)} retrieval rows have no window, such as {min(orphans)}")
        for window in windows.windows.values():
            if window.is_classified and window.window_id not in seen:
                raise TimeFValidationError(f"{window.error_context}: classified window has no retrieval row")
        return cls(indexed)

    def test_windows(self) -> list[Window]:
        """Return the windows the retrieval test file's rows name, in file order.

        They come from every window file, train and val included, since the retrieval split is the
        authors' own.
        """
        windows: list[Window] = []
        for file, rows in self._indexed.items():
            if file.split is SplitName.TEST:
                windows.extend(row.window for row in rows)
        return windows

    def rows(self) -> Iterator[CheckedRow]:
        """Reread the checked retrieval files, a batch at a time.

        Yields:
            Each row with its file and the window it names, in file order.

        Raises:
            TimeFFormatError: If a file no longer holds the rows that were checked.
        """
        for file, indexed in self._indexed.items():
            rows = _iter_rows(file)
            for item in indexed:
                row = self._next_row(rows, file)
                if row is None or row.window_id != item.window.window_id or row.digest != item.digest:
                    raise TimeFFormatError(f"{file.path}: changed since it was checked, at window id {item.window.key}")
                yield CheckedRow(file, item.window, row)
            if self._next_row(rows, file) is not None:
                raise TimeFFormatError(f"{file.path}: changed since it was checked, it has more rows")

    @staticmethod
    def _next_row(rows: Iterator[RetrievalRow], file: TraceFile) -> RetrievalRow | None:
        try:
            return next(rows, None)
        except TimeFValidationError as exc:
            raise TimeFFormatError(f"{file.path}: changed since it was checked: {exc}") from exc
