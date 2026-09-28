"""The :class:`Record` type: one logical unit of time-series data."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
import pyarrow as pa

from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, SupportsAnnotate, TimeInterval, TimeOrigin, TimePoint, new_id
from timenet.types.clock import check_int64, offset_us


@dataclass(kw_only=True)
class Record(SupportsAnnotate):
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    Created via :meth:`~timenet.dataset.TimeFDataset.add_record`. Mutable so ``task_ids`` and
    ``annotations`` can be populated after construction.

    A record has no metadata field. Record-level facts, such as a subject's age or the recording
    device, are :class:`~timenet.types.Annotation` objects with no ``span``. Such an annotation can
    also state a ``unit``, and it travels with every window drawn later from the record.
    """

    sources: tuple[Source, ...] = ()
    """The root sources that belong to this recording."""
    record_id: str = field(default_factory=new_id)
    """Unique id for the record (default: an auto-generated uuid7)."""
    subject_ids: tuple[str, ...] = ()
    """Subjects this record belongs to (empty for subject-less domains)."""
    task_ids: tuple[str, ...] = ()
    """Ids of the tasks attached to this record."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to the record."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Optional JSON-compatible recording metadata."""
    time_span: TimeInterval | None = None
    """The session's overall span on the source recording timeline: an :class:`~timenet.types.TimeInterval`
    covering the whole record, or ``None``. Declare it when the series have gaps and an event may fall in
    one, for example a note taken while every sensor was briefly off. This checks an unscoped span
    against it, rather than against the union of the series' windows. Its ``time_series_ids`` must be
    ``None``, and it must contain every series' window."""

    @property
    def id(self) -> str:
        """Return the record's stable public identifier."""
        return self.record_id

    def __post_init__(self) -> None:
        """Validate the record hierarchy and its optional single-clock session span.

        Raises:
            TimeFValidationError: If ``time_span`` is not a whole-record ``TimeInterval`` or does not
                contain some series' window.
        """
        if self.sources:
            tuple(self.walk_sources())
            signal_ids = [signal.id for signal in self.walk_signals()]
            if len(signal_ids) != len(set(signal_ids)):
                raise TimeFValidationError(f"record {self.record_id!r} contains duplicate signal IDs")
        if self.time_span is not None:
            if not isinstance(self.time_span, TimeInterval):
                raise TimeFValidationError(
                    f"Record.time_span must be a TimeInterval covering the whole record, got {self.time_span!r}"
                )
            if self.time_span.time_series_ids is not None:
                raise TimeFValidationError(
                    "Record.time_span covers the whole record, so its time_series_ids must be None"
                )
            origin = self._single_origin(None)
            if self.time_span.time_origin is not None and self.time_span.time_origin is not origin:
                raise TimeFValidationError("Record.time_span names a different source clock")
            for ts in self.signals:
                window = ts.span_us
                if window is not None and (window[0] < self.time_span.start_us or window[1] > self.time_span.end_us):
                    raise TimeFValidationError(
                        f"Record.time_span ({self.time_span.start_us}, {self.time_span.end_us}) us must contain "
                        f"every series' window, but {ts.id!r} covers {window} us"
                    )

    @property
    def signals(self) -> tuple[Signal, ...]:
        """Return every signal in this record's hierarchy."""
        return tuple(self.walk_signals())

    @property
    def signal_origins(self) -> dict[str, TimeOrigin]:
        """Map each signal ID to the clock of its direct source."""
        return {signal.id: source.start_time for source in self.walk_sources() for signal in source.signals}

    @property
    def start_time(self) -> int | None:
        """Return the earliest absolute timed sample, if every timed clock is known."""
        starts = []
        for source in self.walk_sources():
            for signal in source.signals:
                window = signal.span_us
                if window is not None:
                    if source.start_time.timestamp is None:
                        return None
                    starts.append(cast("int", source.start_time.timestamp) + window[0])
        if not starts:
            return None
        earliest = min(starts)
        check_int64("Record.start_time", earliest)
        return earliest

    def _single_origin(self, signal_ids: tuple[str, ...] | None) -> TimeOrigin:
        """Find the one timed source clock selected by signal IDs.

        Returns:
            The selected clock origin.

        Raises:
            TimeFValidationError: If the scope has no timed signal or more than one clock.
        """
        timed_ids = {signal.id for signal in self.signals if signal.span_us is not None}
        selected = timed_ids if signal_ids is None else timed_ids.intersection(signal_ids)
        if signal_ids is not None and set(signal_ids) != selected:
            raise TimeFValidationError("time scope must name timed signals on this record")
        origins = self.signal_origins
        clocks = {id(origins[signal_id]): origins[signal_id] for signal_id in selected}
        if len(clocks) != 1:
            raise TimeFValidationError("time scope requires exactly one source clock")
        return next(iter(clocks.values()))

    def _selected_origin(self, signal_ids: tuple[str, ...] | None, time_origin: TimeOrigin | None) -> TimeOrigin:
        """Resolve a clock from an explicit origin or the selected timed signals.

        Returns:
            The selected source clock.

        Raises:
            TimeFValidationError: If the explicit clock does not belong to the selected signals.
        """
        if time_origin is None:
            return self._single_origin(signal_ids)
        origins = self.signal_origins
        timed_ids = {signal.id for signal in self.signals if signal.span_us is not None}
        if not any(origin is time_origin for signal_id, origin in origins.items() if signal_id in timed_ids):
            raise TimeFValidationError("time_origin has no timed signals on this record")
        if signal_ids is not None and any(
            signal_id not in timed_ids or origins.get(signal_id) is not time_origin for signal_id in signal_ids
        ):
            raise TimeFValidationError("time scope must name timed signals on the selected source clock")
        return time_origin

    def walk_sources(self) -> Iterable[Source]:
        """Yield all sources in deterministic depth-first order.

        Yields:
            Every source in the record.

        Raises:
            TimeFValidationError: If a source is attached twice or the hierarchy contains a cycle.
        """  # noqa: DOC502 - raised by the nested traversal helper
        seen: set[int] = set()
        active: set[int] = set()

        def walk(source: Source) -> Iterable[Source]:
            identity = id(source)
            if identity in active:
                raise TimeFValidationError(f"source hierarchy contains a cycle at source {source.id!r}")
            if identity in seen:
                raise TimeFValidationError(f"source {source.id!r} is attached more than once")
            seen.add(identity)
            active.add(identity)
            yield source
            for child in sorted(source.sources, key=lambda item: (item.name, item.id)):
                yield from walk(child)
            active.remove(identity)

        for root in sorted(self.sources, key=lambda item: (item.name, item.id)):
            yield from walk(root)

    def walk_signals(self) -> Iterable[Signal]:
        """Yield every signal in deterministic source and signal order."""
        for source in self.walk_sources():
            yield from sorted(source.signals, key=lambda signal: (signal.name, signal.id))

    @property
    def has_absolute_time(self) -> bool:
        """Whether this record's relative timeline has a Unix-time anchor."""
        return self.start_time is not None

    def time_point(
        self,
        at: datetime,
        *,
        time_series_ids: tuple[str, ...] | None = None,
        time_origin: TimeOrigin | None = None,
    ) -> TimePoint:
        """Build a :class:`~timenet.types.TimePoint` at a wall-clock moment on this record's timeline.

        Places ``at`` on the selected source clock. A span's bounds are offsets on that clock, so
        the clock must have a known absolute origin. Name ``time_origin`` when the record has
        independent clocks and the point has no signal scope.

        Args:
            at: The wall-clock moment, timezone-aware.
            time_series_ids: Series the point is scoped to. ``None`` covers every series.
            time_origin: Source clock to use when this record has independent clocks.

        Returns:
            The point, in microseconds from this record's relative zero.
        """
        origin = self._selected_origin(time_series_ids, time_origin)
        return TimePoint(start_us=offset_us(at, origin.timestamp), time_series_ids=time_series_ids, time_origin=origin)

    def time_interval(
        self,
        start: datetime,
        end: datetime,
        *,
        time_series_ids: tuple[str, ...] | None = None,
        time_origin: TimeOrigin | None = None,
    ) -> TimeInterval:
        """Build a :class:`~timenet.types.TimeInterval` between two wall-clock moments on this timeline.

        Places ``start`` and ``end`` on the selected source clock. The clock must have a known
        absolute origin. Name ``time_origin`` when the record has independent clocks and the
        interval has no signal scope.

        Args:
            start: Wall-clock start, timezone-aware.
            end: Wall-clock end, exclusive and timezone-aware.
            time_series_ids: Series the interval is scoped to. ``None`` covers every series.
            time_origin: Source clock to use when this record has independent clocks.

        Returns:
            The half-open interval, in microseconds from this record's relative zero.
        """
        origin = self._selected_origin(time_series_ids, time_origin)
        return TimeInterval(
            start_us=offset_us(start, origin.timestamp),
            end_us=offset_us(end, origin.timestamp),
            time_series_ids=time_series_ids,
            time_origin=origin,
        )

    def annotate(self, annotation: Annotation, *, warn_when_outside: bool = True) -> Annotation:
        """Attach an annotation to the record and return it.

        Args:
            annotation: The annotation to attach.
            warn_when_outside: Warn and keep the span when it leaves its window, rather than raise.

        Returns:
            A new occurrence that shares the input annotation's reusable content.

        Raises:
            TimeFValidationError: If the annotation's span references a series not on this record. If a
                scoped span names a timeless series. If the span falls outside the window its scope
                selects: the intersection of named series, the record's ``time_span``, or the union of
                the series' windows and ``warn_when_outside`` is False.
        """  # noqa: DOC502 (raised by _validate_annotation, not directly here)
        self._validate_annotation(annotation, warn_when_outside=warn_when_outside)
        attached = annotation._new_occurrence()
        self.annotations = (*self.annotations, attached)
        return attached

    def add_annotations(
        self, annotations: Iterable[Annotation], *, warn_when_outside: bool = True
    ) -> tuple[Annotation, ...]:
        """Attach several annotations to the record, all together or not at all.

        The whole batch is validated before any of it is attached: if one annotation fails a check, the
        call raises and leaves the record unchanged. To keep the annotations before a failure attached,
        loop :meth:`annotate` instead.

        Args:
            annotations: The annotations to attach. Pass a single one to :meth:`annotate`.
            warn_when_outside: As on :meth:`annotate`.

        Returns:
            New occurrences that share the input annotations' reusable content, in the order given.

        Raises:
            TimeFValidationError: as documented on :meth:`annotate`.
        """  # noqa: DOC502 (raised by _validate_annotation, not directly here)
        batch = tuple(annotations)
        for annotation in batch:
            self._validate_annotation(annotation, warn_when_outside=warn_when_outside)
        attached = tuple(annotation._new_occurrence() for annotation in batch)
        self.annotations = (*self.annotations, *attached)
        return attached

    def _validate_annotation(self, annotation: Annotation, *, warn_when_outside: bool = True) -> None:
        """Run :meth:`annotate`'s checks without attaching it.

        Split out so :meth:`add_annotations` can validate a whole batch before committing it in one tuple
        concatenation.

        Args:
            annotation: The annotation to check.
            warn_when_outside: As on :meth:`annotate`.

        Raises:
            TimeFValidationError: as documented on :meth:`annotate`.
        """  # noqa: DOC502 (raised by check_span_within_window, not directly here)
        if annotation.span is not None:
            check_span_within_window(
                f"annotation {annotation.key!r}",
                annotation.span,
                self.signals,
                self.record_id,
                self.time_span,
                warn_when_outside=warn_when_outside,
                origins=self.signal_origins,
            )

    def to_arrow(self) -> pa.Array:
        """Read the sole signal's values as an Arrow array, for the common single-signal record.

        Returns:
            The single :class:`Signal`' values as a 1-D Arrow array.

        Raises:
            ValueError: If the record has more than one signal, read ``signals[i]`` explicitly then.
        """
        if len(self.signals) != 1:
            raise ValueError(
                f"Record.to_arrow() needs a single-signal record, but this one has "
                f"{len(self.signals)} signals; read record.signals[i].to_arrow() instead"
            )
        return self.signals[0].to_arrow()

    def to_numpy(self) -> np.ndarray:
        """Read the sole signal's values as a NumPy array (materializes :meth:`to_arrow`).

        Returns:
            The single :class:`Signal`' values as a 1-D ``np.ndarray``.
        """
        return self.to_arrow().to_numpy(zero_copy_only=False)
