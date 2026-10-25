"""The :class:`Signal` leaf type: one logical stream with a lazy values loader."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import assert_never

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.dataset.materialized_values import (
    NumpyMaterializedValues,
    ValuesLoader,
    as_values_loader,
    materialized_values,
)
from timenet.dataset.span_validation import check_span_within_window
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types import (
    Annotation,
    Span,
    StepInterval,
    SupportsAnnotate,
    TimeInterval,
    TimeSeriesSpec,
)
from timenet.types.ids import EntityId, assigned_id


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class Signal(SupportsAnnotate):
    """Reference to one logical stream of time-series data, with optional windowing and a lazy loader.

    Each Signal has exactly one owning Source. Several Signals can share an immutable TimeAxis or
    TimeSeriesSpec. Consumers read values through :meth:`to_arrow` or :meth:`to_numpy`. The connector
    supplies ``loader`` at build, or :class:`~timenet.reader.TimeFReader` supplies it on read-back.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, units, dtype, and per-timestep shape."""
    name: str
    """Human-readable signal name within the source."""
    time_axis: TimeAxis
    """Where this series' values sit in time. A :class:`~timenet.dataset.axis.RegularAxis` gives a
    cadence, an :class:`~timenet.dataset.axis.IrregularAxis` stores per-value time offsets, and an
    :class:`~timenet.dataset.axis.OrdinalAxis` marks a sequence with no time at all."""
    loader: ValuesLoader
    """Values loader with explicit Arrow, NumPy, and step-range access."""
    time_offsets_loader: ValuesLoader | None = None
    """Values loader returning one int64 microsecond time offset per value, for an irregular series only.

    An :class:`~timenet.dataset.axis.IrregularAxis` requires this loader, and any other axis
    rejects it. A regular axis computes its time offsets and an ordinal one has none. A stream attached to
    either gives a second, conflicting answer to the same question."""
    n_values: int
    """The number of values the series holds, one per timestep. If ``spec`` gives each timestep a shape,
    the series counts one value per timestep, not one per scalar. This matches the count a chunk's
    ``n_values`` reports.

    """
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached directly to this signal."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional JSON-compatible signal metadata."""
    _id: EntityId | None = field(default=None, init=False, repr=False)

    def __init__(  # noqa: PLR0913 - a signal carries its complete public metadata
        self,
        *,
        spec: TimeSeriesSpec,
        time_axis: TimeAxis,
        name: str,
        data: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
        annotations: tuple[Annotation, ...] = (),
        metadata: JsonMapping | None = None,
    ) -> None:
        """Build a signal from values that are already in memory."""
        loader = materialized_values(spec, data)
        self._initialize(
            spec=spec,
            time_axis=time_axis,
            name=name,
            loader=loader,
            n_values=len(loader),
            annotations=annotations,
            metadata=metadata,
        )

    @classmethod
    def from_loader(  # noqa: PLR0913 - lazy signals carry their storage metadata
        cls,
        *,
        spec: TimeSeriesSpec,
        time_axis: TimeAxis,
        name: str,
        loader: ValuesLoader | Callable[[], pa.Array],
        n_values: int,
        time_offsets_loader: ValuesLoader | Callable[[], pa.Array] | None = None,
        annotations: tuple[Annotation, ...] = (),
        metadata: JsonMapping | None = None,
    ) -> "Signal":
        """Build a signal whose values are loaded only when they are read.

        Accept a values loader or an existing Arrow callable. Callables are adapted to the common
        interface without reading them. A callable's optional ``read_steps`` method stays available.

        Returns:
            A signal backed by ``loader`` without calling it.
        """
        instance = cls.__new__(cls)
        instance._initialize(
            spec=spec,
            time_axis=time_axis,
            name=name,
            loader=loader,
            n_values=n_values,
            time_offsets_loader=time_offsets_loader,
            annotations=annotations,
            metadata=metadata,
        )
        return instance

    def _initialize(  # noqa: PLR0913 - shared eager and lazy field initialization
        self,
        *,
        spec: TimeSeriesSpec,
        time_axis: TimeAxis,
        name: str,
        loader: ValuesLoader | Callable[[], pa.Array],
        n_values: int,
        time_offsets_loader: ValuesLoader | Callable[[], pa.Array] | None = None,
        annotations: tuple[Annotation, ...] = (),
        metadata: JsonMapping | None = None,
    ) -> None:
        """Set and validate fields shared by eager and lazy construction."""
        set_field = object.__setattr__
        set_field(self, "spec", spec)
        set_field(self, "name", name)
        set_field(self, "time_axis", time_axis)
        set_field(self, "loader", as_values_loader(loader))
        set_field(
            self, "time_offsets_loader", None if time_offsets_loader is None else as_values_loader(time_offsets_loader)
        )
        set_field(self, "_id", None)
        set_field(self, "n_values", n_values)
        set_field(self, "annotations", tuple(annotations))
        set_field(self, "metadata", {} if metadata is None else metadata)
        self.__post_init__()

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one annotation and return it.

        Args:
            annotation: The annotation to attach.

        Returns:
            The attached annotation.
        """
        self._validate_annotation(annotation)
        return self._attach_annotation(annotation)

    def _validate_annotation(self, annotation: Annotation) -> None:
        """Validate an annotation without changing this Signal."""
        if annotation.span is not None:
            check_span_within_window(
                f"annotation {annotation.key!r}", annotation.span, (self,), f"signal {self.name!r}"
            )

    def _attach_annotation(self, annotation: Annotation) -> Annotation:
        """Attach an already validated annotation as a new occurrence.

        Returns:
            The attached occurrence.
        """
        attached = annotation._new_occurrence()
        object.__setattr__(self, "annotations", (*self.annotations, attached))  # noqa: PLC2801
        return attached

    def __post_init__(self) -> None:
        """Validate the intrinsic per-series invariants.

        Raises:
            TimeFValidationError: If ``n_values`` is not a positive integer. Also if
                ``time_offsets_loader`` and the axis shape disagree about whether this series stores
                per-value time offsets.
            ValueError: If ``signal`` is empty. The axis validates itself.
        """
        if not self.name:
            raise ValueError("Signal.name must be non-empty")
        object.__setattr__(self, "metadata", validate_json_mapping(self.metadata))
        if isinstance(self.n_values, bool) or not isinstance(self.n_values, int) or self.n_values <= 0:
            raise TimeFValidationError(f"Signal.n_values must be a positive integer, got {self.n_values!r}")
        irregular = isinstance(self.time_axis, IrregularAxis)
        if irregular and self.time_offsets_loader is None:
            raise TimeFValidationError(
                "an IrregularAxis series must carry time_offsets_loader: the axis states no cadence, so "
                "nothing else can say where its values sit. Build it with Signal.from_irregular()"
            )
        if not irregular and self.time_offsets_loader is not None:
            raise TimeFValidationError(
                f"time_offsets_loader is only for an IrregularAxis series, but this one has "
                f"{type(self.time_axis).__name__}, which already determines every time offset"
            )

        for annotation in self.annotations:
            self._validate_annotation(annotation)

    @property
    def id(self) -> EntityId:
        """The signal's id, unique among the signals of its dataset.

        :meth:`~timenet.dataset.TimeFDataset.add_record` sets it, and the reader sets it on read-back.

        Raises:
            TimeFValidationError: If the signal's record is not in a dataset yet.
        """  # noqa: DOC502 (raised by assigned_id)
        return assigned_id(self._id, "Signal")

    @property
    def span_us(self) -> tuple[int, int] | None:
        """The half-open microsecond window this series covers, or ``None`` if it has no timeline.

        The axis derives this window, so the window and the axis cannot disagree. The dispatch ends in
        :func:`~typing.assert_never`, so a new axis shape breaks this method at type-check time.

        Returns:
            ``(first time_offset, one past the last)`` in microseconds, or ``None`` for an ordinal series.
        """
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            return (axis.time_offset_us(0), axis.time_offset_us(self.n_values))
        if isinstance(axis, IrregularAxis):
            # One microsecond past the last stored time offset. The axis states no cadence, so there is
            # no next time offset to end on. A microsecond is the finest unit the format addresses.
            return (axis.first_us, axis.last_us + 1)
        if isinstance(axis, OrdinalAxis):
            return None
        assert_never(axis)

    @classmethod
    def from_values(
        cls,
        values: np.ndarray | Sequence[bool | int | float | str | None],
        *,
        spec: TimeSeriesSpec,
        name: str,
        time_axis: TimeAxis,
    ) -> "Signal":
        """Build a series from already-materialized values and wrap them in a loader for the spec's dtype.

        Numeric NumPy arrays retain their buffers when dtype and layout already match the spec.
        NumPy reads return fresh read-only views; treat the input buffers as read-only while in use.
        Boolean inputs are copied, and NumPy reads return writable copies to keep Arrow reads consistent.
        Arrow conversion happens only on an Arrow read. Other inputs become Arrow arrays immediately.
        The array length sets ``n_values``.

        For files or remote sources, use :meth:`from_loader` and supply the length. That constructor
        does not read the values immediately.

        Args:
            values: The signal values to convert to the spec's dtype. Text and enum specs take
                strings. Python ``None`` marks a missing timestep when ``spec.nullable`` is true.
            spec: The series' measurement-modality spec.
            name: The signal name.
            time_axis: Where the values sit in time.

        Returns:
            The constructed :class:`Signal`.
        """
        return cls(
            spec=spec,
            name=name,
            data=values,
            time_axis=time_axis,
        )

    @classmethod
    def from_irregular(
        cls,
        values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
        *,
        time_offsets_us: np.ndarray | Sequence[int],
        spec: TimeSeriesSpec,
        name: str,
        full_validation: bool = True,
    ) -> "Signal":
        """Build an irregular series from materialized values and their time offsets.

        The axis endpoints come from the stream itself, so the two cannot disagree. You cannot state a
        first or last time offset that the time offsets do not have. To convert wall-clock moments, use
        :func:`~timenet.dataset.axis.time_offsets_from_datetimes` before you call.
        Numeric NumPy values and normalized time offsets retain their buffers for NumPy reads.
        Arrow conversion happens only on an Arrow read. Treat retained input buffers as read-only.
        Boolean inputs are copied, and NumPy reads return writable copies.

        Args:
            values: The signal values to convert to the spec's dtype. Text and enum specs take
                strings. Python ``None`` marks a missing timestep when ``spec.nullable`` is true.
            time_offsets_us: One time offset per value, in microseconds from the record's relative zero.
            spec: The series' measurement-modality spec.
            name: The signal name.
            full_validation: Check timestamp ordering by default. Set to ``False`` only for previously
                validated timestamps to skip that scan. Both modes check type, shape, integer range,
                endpoints, and one timestamp per value. Conversion can still allocate when needed.

        Returns:
            The constructed :class:`Signal`.

        Raises:
            TimeFValidationError: If the time offsets are unusable, or there is not exactly one per
                value.
        """
        if not isinstance(full_validation, bool):
            raise TimeFValidationError("full_validation must be a bool")
        loader = materialized_values(spec, values, validate_arrow=True)
        time_offsets = to_time_offsets_us(time_offsets_us, validate_order=full_validation)
        if len(time_offsets) != len(loader):
            raise TimeFValidationError(
                f"an irregular series needs one time offset per value, got {len(time_offsets)} time offsets for {len(loader)} values"
            )
        return cls.from_loader(
            spec=spec,
            name=name,
            time_axis=IrregularAxis(first_us=int(time_offsets[0]), last_us=int(time_offsets[-1])),
            loader=loader,
            time_offsets_loader=NumpyMaterializedValues(time_offsets),
            n_values=len(loader),
        )

    def time_offsets_us(self) -> np.ndarray:
        """Read this series' per-value time offsets.

        Only an irregular series has this. A regular axis computes its time offsets without a read, and an
        ordinal one has none. So neither has a stream to return.

        Returns:
            One int64 microsecond time offset per value.

        Raises:
            TimeFValidationError: If this series' axis is not an
                :class:`~timenet.dataset.axis.IrregularAxis`.
        """
        if self.time_offsets_loader is None:
            raise TimeFValidationError(
                f"only an IrregularAxis series stores time offsets, and this one has {type(self.time_axis).__name__}"
            )
        return self.time_offsets_loader.to_numpy()

    def to_arrow(self) -> pa.Array:
        """Read the series' values as an Arrow array.

        Returns:
            A primitive array for scalar values or a fixed-shape tensor array for N-D values.
        """
        return self.loader.to_arrow()

    def to_numpy(self) -> Shaped[np.ndarray, " time *value"]:
        """Read the series' values as a NumPy array.

        Returns:
            The series' values with shape ``(n_steps, *spec.value_shape)``.

        Raises:
            TimeFValidationError: If a timestep is missing or only partially present. Use
                :meth:`to_arrow` to preserve missingness. A nullable spec without actual nulls
                remains supported, as do NaN and infinity values.
        """  # noqa: DOC502 - raised by the values loader
        return self.loader.to_numpy()

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open temporal step range as Arrow without forcing a NumPy conversion.

        Range-aware storage loaders read only the intersecting chunks. A connector loader that implements
        only the no-argument callable stays compatible through a full-read slice.

        Args:
            start: First temporal step, inclusive.
            stop: Last temporal step, exclusive.

        Returns:
            A primitive Arrow array for scalar series or a fixed-shape tensor array for N-D series.

        Raises:
            TimeFValidationError: If the range is negative or reversed.
        """
        if start < 0 or stop < start:
            raise TimeFValidationError(f"expected 0 <= start <= stop, got start={start}, stop={stop}")
        return self.loader.read_steps(start, stop)

    def step_range(self, span: Span) -> tuple[int, int]:
        """Return the half-open step range ``(start, stop)`` of this series that ``span`` covers.

        The bridge to step-based forecasting libraries: ``stop - start`` is the horizon ``h`` that
        GluonTS, Nixtla, and fev speak in. The pair feeds :meth:`read_steps` to read the ground
        truth. A step span already counts in this series' own steps, so it is the range, bounded by the
        series' length. The axis locates a time span instead: the steps whose time offsets fall in
        ``[start_us, end_us)``, each rounded up to the next step. An ordinal series has no timeline, so a
        time span has no answer on it.

        Args:
            span: The interval to locate. A step span must name this series.

        Returns:
            ``(start, stop)`` step indices, half-open, from this series' first step.

        Raises:
            TimeFValidationError: If ``span`` is a point. If a step span does not name this series or
                runs past its length. If an ordinal series receives a time span, or the span resolves
                past the series' steps. If the located range is empty.
        """
        if isinstance(span, StepInterval):
            if span.time_series is not self:
                raise TimeFValidationError(
                    f"a step span counts on series {span.time_series.name!r}, not this series {self.name!r}"
                )
            if span.stop > self.n_values:
                raise TimeFValidationError(
                    f"step span ({span.start}, {span.stop}) runs past this series' {self.n_values} steps"
                )
            return (span.start, span.stop)
        if not isinstance(span, TimeInterval):  # a point (TimePoint or StepPoint) names no range
            raise TimeFValidationError(f"step_range needs an interval span, not a point, got {span!r}")
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            start, stop = axis.index_at_or_after(span.start_us), axis.index_at_or_after(span.end_us)
        elif isinstance(axis, IrregularAxis):
            offsets = self.time_offsets_us()
            start = int(np.searchsorted(offsets, span.start_us, side="left"))
            stop = int(np.searchsorted(offsets, span.end_us, side="left"))
        elif isinstance(axis, OrdinalAxis):
            raise TimeFValidationError(
                f"a seconds span has no step range on ordinal series {self.name!r}, which has "
                f"no timeline; name the horizon in steps instead"
            )
        else:
            assert_never(axis)
        if start < 0 or stop > self.n_values:
            raise TimeFValidationError(
                f"span {span!r} runs past the {self.n_values} steps of series {self.name!r}: "
                f"it resolves to ({start}, {stop})"
            )
        if stop <= start:
            raise TimeFValidationError(
                f"span {span!r} covers no steps of series {self.name!r}: it resolves to the "
                f"empty range ({start}, {stop}). A forecast horizon needs at least one step"
            )
        return (start, stop)
