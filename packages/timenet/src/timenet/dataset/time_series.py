"""The :class:`Signal` leaf type: one logical stream whose values sit behind an array handle."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, TypeGuard, assert_never

from jaxtyping import Shaped
import numpy as np
import pyarrow as pa

from timenet.dataset.axis import IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.dataset.span_validation import check_span_within_window
from timenet.errors import TimeFNullValuesError, TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types import (
    Annotation,
    Span,
    StepInterval,
    StepSpan,
    SupportsAnnotate,
    TimeInterval,
    TimeSeriesSpec,
    TimeSpan,
)
from timenet.types.handles import ArrayHandle, ArrowArray, CallableArray, NumpyArray
from timenet.types.ids import EntityId, assigned_id


def _validate_enum_values(
    spec: TimeSeriesSpec,
    values: Iterable[object],
) -> None:
    """Reject enum values outside the spec's codebook.

    Args:
        spec: The series' spec with its ``categories``.
        values: The label values to check.

    Raises:
        TimeFValidationError: If any value is not in ``spec.categories``.
    """
    allowed = set(spec.categories)
    unknown = {v for v in values if v not in allowed and not (v is None and spec.nullable)}
    if unknown:
        raise TimeFValidationError(
            f"enum series for {spec.spec_type!r} has values outside its categories "
            f"({list(spec.categories)!r}): {sorted(unknown, key=str)}"
        )


def _can_use_numpy_storage(
    values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
    spec: TimeSeriesSpec,
) -> TypeGuard[np.ndarray]:
    """Check whether in-memory values can stay NumPy under this spec.

    Object arrays go through Arrow, since they can hold ``None``. Text and enum specs go through
    Arrow for its string and dictionary encodings.

    Returns:
        Whether the values are a NumPy array that a :class:`~timenet.types.handles.NumpyArray` can hold.
    """
    return isinstance(values, np.ndarray) and values.dtype.kind != "O" and spec.dtype not in {"str", "enum"}


def _arrow_from_values(
    spec: TimeSeriesSpec, values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array
) -> pa.Array:
    """Build a scalar Arrow array in the spec's dtype and preserve missing values supplied as ``None``.

    Returns:
        Values converted to the spec's dtype, with a separate bit that marks whether each timestep is present.

    Raises:
        TimeFValidationError: If a non-nullable spec receives a null value, or an enum value is
            outside the spec's categories.
    """
    arrow_type = pa.string() if spec.dtype in {"str", "enum"} else pa.from_numpy_dtype(np.dtype(spec.dtype))
    if isinstance(values, pa.Array):
        array = values.cast(arrow_type)
    elif spec.dtype == "bool":
        # Convert present values to booleans as NumPy does. Preserve None as an Arrow null.
        array = pa.array([None if value is None else bool(value) for value in values], type=arrow_type)
    else:
        # pa.array preserves None as a null. The null count below avoids a separate scan.
        array = pa.array(values, type=arrow_type)
    if array.null_count and not spec.nullable:
        raise TimeFValidationError(f"series for {spec.spec_type!r} has null values but nullable=False")
    if spec.dtype == "enum":
        # Encode the labels first. The dictionary stores each distinct label once.
        # Compare these labels with the allowed categories, without creating an object for every value.
        array = array.dictionary_encode()
        _validate_enum_values(spec, array.dictionary.to_pylist())
    return array


def _in_memory(
    spec: TimeSeriesSpec,
    values: np.ndarray | Sequence[bool | int | float | str | None] | pa.Array,
    *,
    validate_arrow: bool = False,
) -> NumpyArray | ArrowArray:
    """Hold in-memory values in the spec's dtype, as NumPy where they can stay NumPy.

    Numeric and boolean NumPy values stay NumPy, so a NumPy read builds no Arrow. Other values
    become an Arrow array. Arrow values are kept as they are unless ``validate_arrow`` asks for the
    same conversion and checks.

    Returns:
        A :class:`~timenet.types.handles.NumpyArray` or an :class:`~timenet.types.handles.ArrowArray`.

    Raises:
        TimeFValidationError: If NumPy values do not have the spec's per-timestep shape, a
            non-nullable spec receives a null value, or an enum value is outside the categories.
    """
    if isinstance(values, pa.Array) and not validate_arrow:
        return ArrowArray(values)
    if _can_use_numpy_storage(values, spec):
        if values.ndim == 0 or values.shape[1:] != spec.value_shape:
            raise TimeFValidationError(
                f"series for {spec.spec_type!r} needs shape (n_steps, *{spec.value_shape}), got {values.shape}"
            )
        # Keep NumPy conversion rules, including truncation that pa.array rejects.
        return NumpyArray(np.asarray(values, dtype=np.dtype(spec.dtype)))
    return ArrowArray(_arrow_from_values(spec, values))


def _array_handle(values: ArrayHandle | Callable[[], pa.Array]) -> ArrayHandle:
    """Keep a value handle as it is, and wrap a plain callable to read through it on every read.

    Returns:
        The handle, or a :class:`~timenet.types.handles.CallableArray` over the callable.

    Raises:
        TimeFValidationError: If a plain callable has its own ``read_steps``, which a
            ``CallableArray`` would never call.
    """
    if isinstance(values, ArrayHandle):
        return values
    if hasattr(values, "read_steps"):
        raise TimeFValidationError(
            f"{type(values).__name__} has its own read_steps(); subclass ArrayHandle to keep its range reads"
        )
    return CallableArray(values)


@dataclass(frozen=True, eq=False, slots=True)
class _SlicedArray(ArrayHandle):
    """Array handle that reads only the steps ``[start, stop)`` of another handle, when it is read.

    :meth:`Signal.slice` uses it to stay lazy. A :class:`CallableArray` would read every value of
    ``base`` and then cut them, and reading the steps at once would read them when the slice is
    made. This handle turns each read into a range read on ``base`` instead.

    Steps count from the slice's own first step. A slice of a slice reads ``base`` directly, so
    slicing again adds no layer.
    """

    base: ArrayHandle
    start: int
    stop: int

    @classmethod
    def from_handle(cls, handle: ArrayHandle, start: int, stop: int) -> "_SlicedArray":
        """Return a handle on the steps ``[start, stop)`` of ``handle``.

        If ``handle`` is itself a sliced array, the new handle reads its base directly.

        Returns:
            The handle, which reads nothing until it is read.
        """
        if isinstance(handle, _SlicedArray):
            return cls(handle.base, handle.start + start, handle.start + stop)
        return cls(handle, start, stop)

    @property
    def is_loaded(self) -> bool:
        """Return whether the underlying handle holds its values in memory."""
        return self.base.is_loaded

    def read(self) -> pa.Array:
        """Return the slice's steps, read from the underlying handle."""
        return self.base.read_steps(self.start, self.stop)

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Return the slice's steps ``[start, stop)`` by asking the underlying handle for only those.

        Like the default, a ``stop`` past the slice's end reads to the end of the slice.

        Returns:
            The steps' values, in the form :meth:`read` returns.
        """
        stop = min(self.start + stop, self.stop)
        return self.base.read_steps(min(self.start + start, stop), stop)

    def read_numpy(self) -> np.ndarray:
        """Return the slice's values as a read-only NumPy array.

        Over in-memory NumPy values this is a view, with no Arrow and no copy. Other handles convert
        :meth:`read`, as the default does.

        Returns:
            The values, shaped ``(n_steps, *value_shape)``.
        """
        if isinstance(self.base, NumpyArray):
            return self.base.read_numpy()[self.start : self.stop]
        return ArrayHandle.read_numpy(self)


class _StepRange(NamedTuple):
    """A half-open range of steps ``[start, stop)`` of a series, and the time offsets read to find it."""

    start: int
    """The first step in the range."""
    stop: int
    """The step just past the range."""
    offsets: np.ndarray | None
    """All time offsets of an irregular series, if they were read to find the range, else ``None``.

    The caller can reuse them instead of reading them again.
    """


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class Signal(SupportsAnnotate):
    """Reference to one logical stream of time-series data, with optional windowing and lazy values.

    Each Signal has exactly one owning Source. Several Signals can share an immutable TimeAxis or
    TimeSeriesSpec. Consumers read values through :meth:`to_arrow` or :meth:`to_numpy`, which apply the
    spec. The values sit behind an internal :class:`~timenet.types.ArrayHandle`: the constructor holds
    them in memory, and :meth:`from_loader` takes a handle or a callable.
    """

    spec: TimeSeriesSpec
    """Measurement-modality contract: type tag, units, dtype, and per-timestep shape."""
    name: str
    """Human-readable signal name within the source."""
    time_axis: TimeAxis
    """Where this series' values sit in time. A :class:`~timenet.dataset.axis.RegularAxis` gives a
    cadence, an :class:`~timenet.dataset.axis.IrregularAxis` stores per-value time offsets, and an
    :class:`~timenet.dataset.axis.OrdinalAxis` marks a sequence with no time at all."""
    _values: ArrayHandle = field(repr=False)
    """Internal handle to the series' values. The accessors read it and apply the spec."""
    _time_offsets: ArrayHandle | None = field(default=None, repr=False)
    """Internal handle to one int64 microsecond time offset per value, for an irregular series only.

    An :class:`~timenet.dataset.axis.IrregularAxis` requires these time offsets, and any other axis
    rejects them. A regular axis computes its time offsets and an ordinal one has none. A stream attached to
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
        held = _in_memory(spec, data)
        self._initialize(
            spec=spec,
            time_axis=time_axis,
            name=name,
            values=held,
            n_values=len(held.values),
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
        loader: ArrayHandle | Callable[[], pa.Array],
        n_values: int,
        time_offsets_loader: ArrayHandle | Callable[[], pa.Array] | None = None,
        annotations: tuple[Annotation, ...] = (),
        metadata: JsonMapping | None = None,
    ) -> "Signal":
        """Build a signal whose values are loaded only when they are read.

        ``loader`` and ``time_offsets_loader`` each take a value handle, which the signal keeps as it
        is, or a plain callable returning an Arrow array, which the signal calls on every read. A
        callable with its own ``read_steps()`` is rejected: subclass ``ArrayHandle`` to keep its range
        reads.

        Returns:
            A signal backed by ``loader`` without reading it.
        """
        instance = cls.__new__(cls)
        instance._initialize(
            spec=spec,
            time_axis=time_axis,
            name=name,
            values=_array_handle(loader),
            n_values=n_values,
            time_offsets=None if time_offsets_loader is None else _array_handle(time_offsets_loader),
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
        values: ArrayHandle,
        n_values: int,
        time_offsets: ArrayHandle | None = None,
        annotations: tuple[Annotation, ...] = (),
        metadata: JsonMapping | None = None,
    ) -> None:
        """Set and validate fields shared by eager and lazy construction."""
        set_field = object.__setattr__
        set_field(self, "spec", spec)
        set_field(self, "name", name)
        set_field(self, "time_axis", time_axis)
        set_field(self, "_values", values)
        set_field(self, "_time_offsets", time_offsets)
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
            TimeFValidationError: If ``n_values`` is not a positive integer. Also if the time
                offsets and the axis shape disagree about whether this series stores per-value time
                offsets.
            ValueError: If ``signal`` is empty. The axis validates itself.
        """
        if not self.name:
            raise ValueError("Signal.name must be non-empty")
        object.__setattr__(self, "metadata", validate_json_mapping(self.metadata))
        if isinstance(self.n_values, bool) or not isinstance(self.n_values, int) or self.n_values <= 0:
            raise TimeFValidationError(f"Signal.n_values must be a positive integer, got {self.n_values!r}")
        irregular = isinstance(self.time_axis, IrregularAxis)
        if irregular and self._time_offsets is None:
            raise TimeFValidationError(
                "an IrregularAxis series must carry time_offsets_loader: the axis states no cadence, so "
                "nothing else can say where its values sit. Build it with Signal.from_irregular()"
            )
        if not irregular and self._time_offsets is not None:
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

    @property
    def is_ordinal(self) -> bool:
        """Whether this series counts steps only and has no timeline, so time spans cannot place it."""
        return isinstance(self.time_axis, OrdinalAxis)

    @classmethod
    def from_values(
        cls,
        values: np.ndarray | Sequence[bool | int | float | str | None],
        *,
        spec: TimeSeriesSpec,
        name: str,
        time_axis: TimeAxis,
    ) -> "Signal":
        """Build a series from already-materialized values and hold them in memory in the spec's dtype.

        Numeric NumPy arrays retain their buffers when dtype and layout already match the spec.
        NumPy reads return fresh read-only views; treat the input buffers as read-only while in use.
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
        Numeric NumPy values and normalized time offsets retain their buffers for NumPy reads, which
        return read-only arrays. Arrow conversion happens only on an Arrow read. Treat retained input
        buffers as read-only.

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
        held = _in_memory(spec, values, validate_arrow=True)
        n_values = len(held.values)
        time_offsets = to_time_offsets_us(time_offsets_us, validate_order=full_validation)
        if len(time_offsets) != n_values:
            raise TimeFValidationError(
                f"an irregular series needs one time offset per value, got {len(time_offsets)} time offsets for {n_values} values"
            )
        return cls.from_loader(
            spec=spec,
            name=name,
            time_axis=IrregularAxis(first_us=int(time_offsets[0]), last_us=int(time_offsets[-1])),
            loader=held,
            time_offsets_loader=NumpyArray(time_offsets),
            n_values=n_values,
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
        if self._time_offsets is None:
            raise TimeFValidationError(
                f"only an IrregularAxis series stores time offsets, and this one has {type(self.time_axis).__name__}"
            )
        return self._time_offsets.read_numpy()

    def to_arrow(self) -> pa.Array:
        """Read the series' values as an Arrow array.

        Returns:
            A primitive array for scalar values or a fixed-shape tensor array for N-D values.
        """
        return self._values.read()

    def to_numpy(self) -> Shaped[np.ndarray, " time *value"]:
        """Read the series' values as a read-only NumPy array.

        Returns:
            The series' values with shape ``(n_steps, *spec.value_shape)``.

        Raises:
            TimeFNullValuesError: If a timestep is missing or only partially present. Use
                :meth:`to_arrow` to preserve missingness. A nullable spec without actual nulls
                remains supported, as do NaN and infinity values.
        """
        try:
            return self._values.read_numpy()
        except TimeFNullValuesError as error:
            detail = str(error)
            raise TimeFNullValuesError(
                f"series contains null values. {detail[:1].upper()}{detail[1:]}. Use to_arrow() to preserve missingness"
            ) from error

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a half-open temporal step range as Arrow without forcing a NumPy conversion.

        A series whose stored values were loaded, or that holds its values in memory, is sliced. An
        unloaded stored series reads only the chunks that intersect the range, and one from a loader
        callable is read in full first.

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
        return self._values.read_steps(start, stop)

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
            self._check_step_span_counts_on_this(span)
            if span.stop > self.n_values:
                raise TimeFValidationError(
                    f"step span ({span.start}, {span.stop}) runs past this series' {self.n_values} steps"
                )
            return (span.start, span.stop)
        if not isinstance(span, TimeInterval):  # a point (TimePoint or StepPoint) names no range
            raise TimeFValidationError(f"step_range needs an interval span, not a point, got {span!r}")
        steps = self._steps_in_time_range(span.start_us, span.end_us)
        if steps.start < 0 or steps.stop > self.n_values:
            raise TimeFValidationError(
                f"span {span!r} runs past the {self.n_values} steps of series {self.name!r}: "
                f"it resolves to ({steps.start}, {steps.stop})"
            )
        if steps.stop <= steps.start:
            raise TimeFValidationError(
                f"span {span!r} covers no steps of series {self.name!r}: it resolves to the "
                f"empty range ({steps.start}, {steps.stop}). A forecast horizon needs at least one step"
            )
        return (steps.start, steps.stop)

    def _check_step_span_counts_on_this(self, span: StepInterval) -> None:
        """Reject a step span that counts on another series, whose steps mean nothing on this one.

        Raises:
            TimeFValidationError: If ``span`` does not name this series.
        """
        if self.is_selected_by(span):
            return
        other = span.time_series.name
        if other == self.name:
            # A slice keeps its original's name, so a step span on the original lands here.
            raise TimeFValidationError(
                f"step span ({span.start}, {span.stop}) counts on another series named {other!r}. A slice "
                f"is a new series with its original's name, so build the step span on this series itself"
            )
        raise TimeFValidationError(
            f"step span ({span.start}, {span.stop}) counts on series {other!r}, not on series {self.name!r}"
        )

    def _steps_in_time_range(self, start_us: int, end_us: int) -> _StepRange:
        """Return the steps whose time offsets fall in ``[start_us, end_us)``, without range checks.

        The caller checks the result. On a regular axis, the range can start before step 0 or end
        past :attr:`n_values`, and it is empty when no step falls in the time range.

        Returns:
            The step range. Its ``offsets`` are set on an irregular axis only.

        Raises:
            TimeFValidationError: If the series is ordinal, which has no timeline.
        """
        axis = self.time_axis
        if isinstance(axis, RegularAxis):
            return _StepRange(axis.index_at_or_after(start_us), axis.index_at_or_after(end_us), None)
        if isinstance(axis, IrregularAxis):
            offsets = self.time_offsets_us()
            start = int(np.searchsorted(offsets, start_us, side="left"))
            return _StepRange(start, int(np.searchsorted(offsets, end_us, side="left")), offsets)
        if isinstance(axis, OrdinalAxis):
            raise TimeFValidationError(
                f"a seconds span has no step range on ordinal series {self.name!r}, which has "
                f"no timeline; name the horizon in steps instead"
            )
        assert_never(axis)

    def overlaps_with(self, span: TimeSpan) -> bool:
        """Return whether at least one value of this series falls inside ``span``.

        A record can hold several series that cover different parts of its timeline, such as two
        segments of one recording. Use this to find the series a time span reaches. A value is inside
        the span when its time offset falls in ``[start_us, exclusive_end)``, the same rule
        :meth:`slice` uses. So for a time interval, ``slice(span, clip=True)`` keeps at least one value
        whenever this returns ``True``.

        This compares times only and ignores the span's ``time_series``. Ask :meth:`is_selected_by`
        whether the span applies to this series. On an irregular series this reads the time offsets,
        unless the span lies outside the window :attr:`span_us` reports.

        Args:
            span: A time interval or time point on the record timeline.

        Returns:
            ``True`` if a value of this series falls inside the span.

        Raises:
            TimeFValidationError: If ``span`` is not a :class:`~timenet.types.TimeSpan`. If the series is
                ordinal: it has no timeline, so check :attr:`is_ordinal` first.
        """
        if not isinstance(span, TimeSpan):
            raise TimeFValidationError(f"overlaps_with needs a time span, got {type(span).__name__} {span!r}")
        window = self.span_us
        if window is None:
            raise TimeFValidationError(
                f"overlaps_with needs a series with a timeline, but series {self.name!r} is ordinal; "
                "check is_ordinal first"
            )
        first_us, end_us = window
        if span.exclusive_end <= first_us or span.start_us >= end_us:
            return False  # outside the window, so no value can be inside, and no offsets are read
        steps = self._steps_in_time_range(span.start_us, span.exclusive_end)
        # A regular axis can locate steps outside the series, so count only the ones it holds.
        return min(steps.stop, self.n_values) > max(steps.start, 0)

    def is_selected_by(self, span: Span) -> bool:
        """Return whether ``span`` applies to this series.

        A step span selects only the series it counts on. A time span with no ``time_series``
        selects every series, and a time span with ``time_series`` set selects only the series it
        lists. Series are compared by identity, not by name, so a span does not select a slice of a
        series it names: the slice is a new series.

        This checks the scope only, not the time or steps the span covers. :meth:`overlaps_with` and
        :meth:`slice` compare times only, so call this first when the scope matters.

        Args:
            span: A time or step span, point or interval.

        Returns:
            ``True`` if the span applies to this series.

        Raises:
            TimeFValidationError: If ``span`` is neither a time span nor a step span.
        """
        if isinstance(span, StepSpan):
            return span.time_series is self
        if isinstance(span, TimeSpan):
            return span.time_series is None or any(series is self for series in span.time_series)
        raise TimeFValidationError(f"is_selected_by needs a time or step span, got {type(span).__name__} {span!r}")

    def slice(self, span: Span, *, clip: bool = False) -> "Signal":
        """Return a new lazy series that holds only the steps ``span`` covers.

        The span must hold at least one of this series' steps. A step span must count on this series,
        because its steps mean nothing on another. A time span is placed by time only, and its
        ``time_series`` is not checked, so ask :meth:`is_selected_by` first when the scope matters.
        That also lets you cut a slice again with the span it was cut with.

        The slice keeps this series' name, spec, metadata, and place on the timeline, and drops its
        annotations, which can fall outside the slice. It reads no values until it is read, and then
        reads only its own steps where this series supports range reads (see :meth:`read_steps`).

        To cut every series of a record that a time interval selects and reaches::

            for signal in record.walk_signals():
                if signal.is_selected_by(interval) and not signal.is_ordinal and signal.overlaps_with(interval):
                    parts.append(signal.slice(interval, clip=True))

        Args:
            span: The interval to keep: a time interval within :attr:`span_us`, or a step interval on
                this series that ends at or before :attr:`n_values`.
            clip: Keep the part of the span inside this series instead of raising when the span reaches
                past it.

        Returns:
            A lazy :class:`Signal` that holds the steps the span covers.

        Raises:
            TimeFValidationError: If ``span`` is a point, a step span on another series, or a time
                span on an ordinal series. If the span reaches past this series and ``clip`` is
                false, or covers no step of this series.
        """  # noqa: DOC502 (raised by _slice_range)
        steps = self._slice_range(span, clip=clip)
        axis = self.time_axis
        time_offsets: NumpyArray | None = None
        if isinstance(axis, RegularAxis):
            axis = axis.at_index(steps.start)
        elif isinstance(axis, IrregularAxis):
            # A step span does not read the offsets to locate itself.
            offsets = self.time_offsets_us() if steps.offsets is None else steps.offsets
            sliced = offsets[steps.start : steps.stop]
            axis = IrregularAxis(first_us=int(sliced[0]), last_us=int(sliced[-1]))
            # Copy the slice's offsets, so a kept slice does not hold the original's whole array alive.
            time_offsets = NumpyArray(sliced.copy())
        elif not isinstance(axis, OrdinalAxis):  # an ordinal axis counts steps only, so it stays as is
            assert_never(axis)
        return Signal.from_loader(
            spec=self.spec,
            time_axis=axis,
            name=self.name,
            loader=_SlicedArray.from_handle(self._values, steps.start, steps.stop),
            n_values=steps.stop - steps.start,
            time_offsets_loader=time_offsets,
            metadata=self.metadata,
        )

    def _slice_range(self, span: Span, *, clip: bool) -> _StepRange:
        """Check ``span`` for :meth:`slice` and return the steps the slice keeps.

        It checks that the span is an interval, and that a step span counts on this series. It then
        turns a time span into steps, after it cuts the span to this series (``clip=True``) or checks
        that the span lies inside it. A step span needs no conversion, only the same cut or check.

        Returns:
            The steps to keep. Its ``offsets`` are the time offsets read to find a time span's steps
            on an irregular axis, so :meth:`slice` does not read them again.

        Raises:
            TimeFValidationError: For each span :meth:`slice` rejects.
        """
        if isinstance(span, StepInterval):
            if not clip:  # step_range checks the series and the length
                return _StepRange(*self.step_range(span), None)
            self._check_step_span_counts_on_this(span)
            stop = min(span.stop, self.n_values)
            if stop <= span.start:
                raise TimeFValidationError(
                    f"step span ({span.start}, {span.stop}) covers no steps of the {self.n_values} "
                    f"steps of series {self.name!r}"
                )
            return _StepRange(span.start, stop, None)
        if not isinstance(span, TimeInterval):  # a point (TimePoint or StepPoint) names no range
            raise TimeFValidationError(f"slice needs an interval span, not a point, got {span!r}")
        window = self.span_us
        if window is None:
            raise TimeFValidationError(
                f"a time interval cannot select part of ordinal series {self.name!r}, which has no "
                "timeline; scope in steps instead"
            )
        first_us, end_us = window
        start_us, stop_us = span.start_us, span.end_us
        if clip:
            start_us, stop_us = max(start_us, first_us), min(stop_us, end_us)
        elif start_us < first_us or stop_us > end_us:
            raise TimeFValidationError(
                f"span {span!r} runs past series {self.name!r}, which covers [{first_us}, {end_us}) "
                "microseconds; pass clip=True to keep the part inside it"
            )
        if stop_us <= start_us:
            raise TimeFValidationError(f"span {span!r} does not overlap series {self.name!r}")
        steps = self._steps_in_time_range(start_us, stop_us)
        if steps.stop <= steps.start:
            raise TimeFValidationError(
                f"span {span!r} covers no steps of series {self.name!r}: no value falls inside it"
            )
        return steps
