"""Validated target values and catalog rows."""

from typing import Self

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from timenet.dataset.record import Record
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.task_target_kind import TaskTargetKind, is_step_span
from timenet.types import StepInterval, StepPoint, TimeInterval, TimePoint
from timenet.types._model import TimeFModel
from timenet.types.text import read_value


_REQUIRED_COLUMNS: dict[TaskTargetKind, frozenset[str]] = {
    TaskTargetKind.TEXT: frozenset(("text_value",)),
    TaskTargetKind.INTEGER: frozenset(("integer_value",)),
    TaskTargetKind.FLOAT: frozenset(("float_value",)),
    TaskTargetKind.BOOLEAN: frozenset(("boolean_value",)),
    TaskTargetKind.RECORD: frozenset(("record_id",)),
    TaskTargetKind.SIGNAL: frozenset(("signal_id",)),
    TaskTargetKind.TIME_POINT: frozenset(("span_start",)),
    TaskTargetKind.TIME_INTERVAL: frozenset(("span_start", "span_end")),
    TaskTargetKind.STEP_POINT: frozenset(("span_start", "span_signal_ids")),
    TaskTargetKind.STEP_INTERVAL: frozenset(("span_start", "span_end", "span_signal_ids")),
}
_PAYLOAD_COLUMNS = frozenset().union(*_REQUIRED_COLUMNS.values())


class TargetValueRow(TimeFModel):
    """Represent a public target with validated values and public object IDs."""

    target_kind: TaskTargetKind
    text_value: StrictStr | None = None
    integer_value: StrictInt | None = None
    float_value: StrictFloat | None = None
    boolean_value: StrictBool | None = None
    span_start: StrictInt | None = None
    span_end: StrictInt | None = None
    record_id: StrictInt | None = None
    signal_id: StrictInt | None = None
    span_signal_ids: tuple[StrictInt, ...] | None = None

    @model_validator(mode="after")
    def _check_kind_columns(self) -> Self:
        kind = self.target_kind
        scope = self.span_signal_ids
        if is_step_span(kind) and (scope is None or len(scope) != 1):
            raise TimeFValidationError(f"task target kind {kind.value!r} requires exactly one span Signal")
        populated = {name for name in _PAYLOAD_COLUMNS if getattr(self, name) is not None}
        required = _REQUIRED_COLUMNS[kind]
        allowed = required
        if kind in {TaskTargetKind.TIME_POINT, TaskTargetKind.TIME_INTERVAL}:
            allowed = required | {"span_signal_ids"}
        if missing := required - populated:
            raise TimeFValidationError(f"task target kind {kind.value!r} requires {', '.join(sorted(missing))}")
        if unexpected := populated - allowed:
            raise TimeFValidationError(
                f"task target kind {kind.value!r} does not allow {', '.join(sorted(unexpected))}"
            )
        return self

    @classmethod
    def from_target(cls, target: object) -> Self:  # noqa: PLR0911 - one branch per target kind
        """Construct a validated row for one public target.

        Returns:
            The target's value columns, with unrelated columns left NULL.

        Raises:
            TimeFFormatError: If the target type is unsupported.
        """
        target = read_value(target)
        if isinstance(target, bool):
            return cls(target_kind=TaskTargetKind.BOOLEAN, boolean_value=target)
        elif isinstance(target, str):
            return cls(target_kind=TaskTargetKind.TEXT, text_value=target)
        elif isinstance(target, int):
            return cls(target_kind=TaskTargetKind.INTEGER, integer_value=target)
        elif isinstance(target, float):
            return cls(target_kind=TaskTargetKind.FLOAT, float_value=target)
        elif isinstance(target, Record):
            return cls(target_kind=TaskTargetKind.RECORD, record_id=target.id)
        elif isinstance(target, Signal):
            return cls(target_kind=TaskTargetKind.SIGNAL, signal_id=target.id)
        elif isinstance(target, TimePoint):
            return cls(
                target_kind=TaskTargetKind.TIME_POINT,
                span_start=target.start_us,
                span_signal_ids=None if target.time_series is None else tuple(s.id for s in target.time_series),
            )
        elif isinstance(target, TimeInterval):
            return cls(
                target_kind=TaskTargetKind.TIME_INTERVAL,
                span_start=target.start_us,
                span_end=target.end_us,
                span_signal_ids=None if target.time_series is None else tuple(s.id for s in target.time_series),
            )
        elif isinstance(target, StepPoint):
            return cls(
                target_kind=TaskTargetKind.STEP_POINT,
                span_start=target.start,
                span_signal_ids=(target.time_series.id,),
            )
        elif isinstance(target, StepInterval):
            return cls(
                target_kind=TaskTargetKind.STEP_INTERVAL,
                span_start=target.start,
                span_end=target.stop,
                span_signal_ids=(target.time_series.id,),
            )
        else:
            raise TimeFFormatError(f"cannot encode unsupported target type {type(target).__name__}")
