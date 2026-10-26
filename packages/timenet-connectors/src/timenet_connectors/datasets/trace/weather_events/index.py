"""Index the archive's rows: the windows and the Measurements they share, the retrieval rows, and the Records."""

from collections.abc import Iterable, Iterator

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.trace.weather_events.archive import TraceFileRef, TraceSplit, check_pinned_counts
from timenet_connectors.datasets.trace.weather_events.records import (
    Measurement,
    TraceRecords,
    TraceWindow,
    WindowLoader,
    WindowTimeline,
)
from timenet_connectors.datasets.trace.weather_events.retrieval import RetrievalRow, read_retrieval_rows
from timenet_connectors.datasets.trace.weather_events.windows import WindowRef, WindowRow
from timenet_connectors.sources.json_stream import iter_object


class WindowIndex:
    """Every TraceWindow of the window files, by its key, and the Measurements they share."""

    def __init__(self) -> None:
        self._windows: dict[str, TraceWindow] = {}
        # Every distinct Measurement, by the digest of what makes windows the same measurements.
        self._measurements: dict[str, Measurement] = {}

    @classmethod
    def from_files(cls, files: Iterable[TraceFileRef]) -> "WindowIndex":
        """Index every window of the given window files.

        Returns:
            The index.
        """
        index = cls()
        for file in files:
            index._add_file(file)
        return index

    def _add_file(self, file: TraceFileRef) -> None:
        """Index one window file: group its windows by measurements, and check its pinned counts.

        An invalid window id, window, copy coordinates, or pinned count raises ``TimeFValidationError``.
        """
        rows = tasks = 0
        for key, published, offset, length in iter_object(file.path):
            ref = self._new_ref(file, key, offset, length)
            row = WindowRow.parse(published, window_id=ref.window_id, kind=file.kind, location=ref.context)
            measurement = self._measurement_for(row, ref)
            window = TraceWindow(ref, measurement, row.county, row.event_type)
            measurement.windows.append(window)
            self._windows[key] = window
            rows += 1
            tasks += window.is_eligible_for_classification() or window.is_eligible_for_forecasting()
        check_pinned_counts(file, rows, tasks)

    def _measurement_for(self, row: WindowRow, ref: WindowRef) -> Measurement:
        """Return the Measurement that holds a window's values: an earlier copy's, or a new one.

        Returns:
            The Measurement, created from this window if no earlier window held the same values. An
            earlier copy taken at other coordinates raises ``TimeFValidationError``.
        """
        key = row.measurement_key()
        if key in self._measurements:
            measurement = self._measurements[key]
            measurement.check_station(row.station, ref)
            return measurement
        measurement = Measurement(
            station=row.station,
            sampling=row.mode,
            timeline=WindowTimeline.of(row.readings.timestamps),
            value_source=ref,
        )
        self._measurements[key] = measurement
        return measurement

    def _new_ref(self, file: TraceFileRef, key: str, offset: int, length: int) -> WindowRef:
        # Retrieval rows give integer ids, so a key must read exactly as str(id), such as 12, not 012.
        if not key.isascii() or not key.isdecimal() or key != str(int(key)):
            raise TimeFValidationError(f"{file.path}: expected a canonical decimal window id, got {key!r}")
        if key in self._windows:
            raise TimeFValidationError(f"{file.path}: window {key} also appears in {self._windows[key].ref.file.path}")
        return WindowRef(file, key, offset, length)

    def window(self, key: str) -> TraceWindow:
        """Return the window with a key."""
        return self._windows[key]

    def __contains__(self, key: object) -> bool:
        return key in self._windows

    def __iter__(self) -> Iterator[TraceWindow]:
        return iter(self._windows.values())

    @property
    def measurement_count(self) -> int:
        """Return the number of distinct Measurements the windows hold."""
        return len(self._measurements)

    def build_records(self, loader: WindowLoader) -> TraceRecords:
        """Build the Record of every Measurement, now that every window is indexed.

        Returns:
            The Records, by Measurement.
        """
        return TraceRecords({measurement: measurement.to_record(loader) for measurement in self._measurements.values()})


class RetrievalIndex:
    """Which window each retrieval row names, file by file, checked against the window index.

    It keeps no descriptions or reports: iterating rereads the files, so the task stream holds one batch
    of descriptions at a time.
    """

    def __init__(self, named: dict[TraceFileRef, list[TraceWindow]]) -> None:
        self._named = named

    @classmethod
    def from_files(cls, files: tuple[TraceFileRef, ...], windows: WindowIndex) -> "RetrievalIndex":
        """Read the retrieval files, and check every row against the indexed windows.

        Every row must name a window, once; every classified window must have its event report, and
        every report a classified window. The task stream rereads the rows and relies on these checks.

        Returns:
            The index of the windows each file's rows name.

        Raises:
            TimeFValidationError: If a row, a report, or a file's pinned counts is invalid.
        """
        named: dict[TraceFileRef, list[TraceWindow]] = {}
        seen: set[str] = set()
        orphans: list[str] = []
        for file in files:
            named[file] = []
            rows = events = 0
            for row in read_retrieval_rows(file):
                if row.key in seen:
                    raise TimeFValidationError(f"{file.path}: window id {row.key} appears in more than one row")
                seen.add(row.key)
                rows += 1
                events += row.event_report is not None
                if row.key not in windows:
                    orphans.append(row.key)
                    continue
                window = windows.window(row.key)
                window.check_report(row.event_report)
                named[file].append(window)
            check_pinned_counts(file, rows, events)
        if orphans:
            raise TimeFValidationError(f"{len(orphans)} retrieval rows have no window, such as {min(orphans)}")
        for window in windows:
            if window.is_eligible_for_classification() and window.ref.key not in seen:
                raise TimeFValidationError(f"{window.ref.context}: classified window has no retrieval row")
        return cls(named)

    def test_windows(self) -> list[TraceWindow]:
        """Return the windows the retrieval test file's rows name, in file order.

        They come from every window file, train and val included, since the retrieval split is the
        authors' own.

        Returns:
            The windows.
        """
        return [window for file, windows in self._named.items() if file.split is TraceSplit.TEST for window in windows]

    def __iter__(self) -> Iterator[tuple[TraceFileRef, TraceWindow, RetrievalRow]]:
        """Reread the checked retrieval files, one batch at a time.

        Yields:
            Each row's file, the window it names, and the row, in file order.

        Raises:
            TimeFValidationError: If a file no longer holds the rows ``from_files`` read.
        """
        for file, windows in self._named.items():
            for window, row in zip(windows, read_retrieval_rows(file), strict=True):
                if row.key != window.ref.key:
                    raise TimeFValidationError(f"{file.path}: changed since it was checked, at window id {row.key}")
                yield file, window, row
