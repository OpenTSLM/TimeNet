"""The :class:`Record` type: one logical unit of time-series data."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pyarrow as pa

from timenet.dataset.source import Source
from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types import Annotation, SupportsAnnotate, TimeInterval, TimeOrigin, TimePoint
from timenet.types.clock import check_int64, offset_us, unix_us
from timenet.types.ids import EntityId, assigned_id


@dataclass(kw_only=True)
class Record(SupportsAnnotate):
    """One logical unit of time-series data: a recording, a session, a sensor bundle, a market window.

    You create a record with :meth:`~timenet.dataset.TimeFDataset.add_record`. The record is mutable,
    so you can add ``annotations`` after construction.

    A record has no metadata field. Record-level facts, such as a subject's age or the recording
    device, are :class:`~timenet.types.Annotation` objects with no ``span``. Such an annotation can
    also state a ``unit``, and it travels with every window drawn later from the record.
    """

    sources: tuple[Source, ...] = ()
    """The root sources that belong to this recording."""
    start_time: TimeOrigin = field(default_factory=TimeOrigin)
    """The common relative zero, with an optional absolute timestamp."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to the record."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional JSON-compatible recording metadata."""
    time_span: TimeInterval | None = None
    """The session's overall span on the source recording timeline: an :class:`~timenet.types.TimeInterval`
    covering the whole record, or ``None``. Declare it when the series have gaps and an event may fall in
    one, for example a note taken while every sensor was briefly off. This checks an unscoped span
    against it, rather than against the union of the series' windows. Its ``time_series`` must be
    ``None``, and it must contain every series' window."""
    _id: EntityId | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the record hierarchy and its optional single-clock session span.

        Raises:
            TimeFValidationError: If ``time_span`` is not a whole-record ``TimeInterval`` or does not
                contain some series' window.
        """
        if not isinstance(self.start_time, TimeOrigin):
            raise TimeFValidationError("Record.start_time must be a TimeOrigin")
        self.metadata = validate_json_mapping(self.metadata)
        for signal in self.signals:
            if (window := signal.span_us) is not None:
                for endpoint in window:
                    check_int64(f"signal {signal.name!r} time offset", endpoint)
                    if self.start_time.timestamp is not None:
                        check_int64(
                            f"signal {signal.name!r} absolute time", unix_us(self.start_time.timestamp) + endpoint
                        )
        if self.time_span is not None:
            if not isinstance(self.time_span, TimeInterval):
                raise TimeFValidationError(
                    f"Record.time_span must be a TimeInterval covering the whole record, got {self.time_span!r}"
                )
            if self.time_span.time_series is not None:
                raise TimeFValidationError("Record.time_span covers the whole record, so its time_series must be None")
            for ts in self.signals:
                window = ts.span_us
                if window is not None and (window[0] < self.time_span.start_us or window[1] > self.time_span.end_us):
                    raise TimeFValidationError(
                        f"Record.time_span ({self.time_span.start_us}, {self.time_span.end_us}) us must contain "
                        f"every series' window, but {ts.name!r} covers {window} us"
                    )

        for annotation in self.annotations:
            self._validate_annotation(annotation)

    @property
    def id(self) -> EntityId:
        """The record's id, unique among the records of its dataset.

        :meth:`~timenet.dataset.TimeFDataset.add_record` sets it, and the reader sets it on read-back.

        Raises:
            TimeFValidationError: If the record is not in a dataset yet.
        """  # noqa: DOC502 (raised by assigned_id)
        return assigned_id(self._id, "Record")

    @property
    def _label(self) -> str:
        """Name the record in an error message, with its id once it has one."""
        return "record" if self._id is None else f"record {self._id!r}"

    @property
    def signals(self) -> tuple[Signal, ...]:
        """Return every signal in this record's hierarchy."""
        return tuple(self.walk_signals())

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
                raise TimeFValidationError(f"source hierarchy contains a cycle at source {source.name!r}")
            if identity in seen:
                raise TimeFValidationError(f"source {source.name!r} is attached more than once")
            seen.add(identity)
            active.add(identity)
            yield source
            for child in sorted(source.sources, key=lambda item: item.name):
                yield from walk(child)
            active.remove(identity)

        for root in sorted(self.sources, key=lambda item: item.name):
            yield from walk(root)

    def walk_signals(self) -> Iterable[Signal]:
        """Yield every signal in deterministic source and signal order."""
        for source in self.walk_sources():
            yield from sorted(source.signals, key=lambda signal: signal.name)

    def walk_annotations(self) -> Iterable[Annotation]:
        """Yield annotation occurrences from this Record, its Sources, and its Signals."""
        yield from self.annotations
        for source in self.walk_sources():
            yield from source.annotations
            for signal in source.signals:
                yield from signal.annotations

    @property
    def has_absolute_time(self) -> bool:
        """Whether this record's relative timeline has a Unix-time anchor."""
        return self.start_time.timestamp is not None

    def time_point(self, at: datetime, *, time_series: tuple[Signal, ...] | None = None) -> TimePoint:
        """Build a :class:`~timenet.types.TimePoint` at a wall-clock moment on this record's timeline.

        The Record must have a known absolute timestamp.

        Args:
            at: The wall-clock moment, timezone-aware.
            time_series: Series the point is scoped to. ``None`` covers every series.

        Returns:
            The point, in microseconds from this record's relative zero.
        """
        return TimePoint(start_us=offset_us(at, self.start_time.timestamp), time_series=time_series)

    def time_interval(
        self, start: datetime, end: datetime, *, time_series: tuple[Signal, ...] | None = None
    ) -> TimeInterval:
        """Build a :class:`~timenet.types.TimeInterval` between two wall-clock moments on this timeline.

        The Record must have a known absolute timestamp.

        Args:
            start: Wall-clock start, timezone-aware.
            end: Wall-clock end, exclusive and timezone-aware.
            time_series: Series the interval is scoped to. ``None`` covers every series.

        Returns:
            The half-open interval, in microseconds from this record's relative zero.
        """
        return TimeInterval(
            start_us=offset_us(start, self.start_time.timestamp),
            end_us=offset_us(end, self.start_time.timestamp),
            time_series=time_series,
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
                self._label,
                self.time_span,
                warn_when_outside=warn_when_outside,
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
        """Read the sole signal's values as NumPy, without building Arrow when they are NumPy.

        Returns:
            The single :class:`Signal`' values as an ``np.ndarray``.

        Raises:
            TimeFValidationError: If the record does not have exactly one signal.
        """
        signals = self.signals
        if len(signals) != 1:
            raise TimeFValidationError(
                f"Record.to_numpy() needs a single-signal record, but this one has "
                f"{len(signals)} signals; read record.signals[i].to_numpy() instead"
            )
        return signals[0].to_numpy()
