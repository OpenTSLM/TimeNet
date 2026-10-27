"""The :class:`Signal` leaf type: one logical stream whose values sit behind an array handle."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeGuard, assert_never

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
    SupportsAnnotate,
    TimeInterval,
    TimeSeriesSpec,
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

        A stored series reads only the chunks that intersect the range. A series held in memory is
        sliced, and one from a loader callable is read in full first.

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
