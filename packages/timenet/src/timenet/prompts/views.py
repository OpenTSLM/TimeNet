"""Views of a classification task that hold only plain values, for prompt templates.

A template sees a view, never the task. The input view leaves out the targets, the rationale, the split,
IDs, Record metadata and the annotations that the task does not select as input.
"""

from collections.abc import Iterable
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import NamedTuple

import pyarrow as pa

from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping
from timenet.types import (
    Annotation,
    ClassificationTask,
    DatasetMetadata,
    LabelSet,
    Span,
    Task,
    TextHandle,
    TextListHandle,
    TimeInterval,
    ureg,
)
from timenet.types.clock import US_PER_S
from timenet.types.units import normalize_unit


ALLOWED_METADATA_KEYS = ("device", "placement")
"""The metadata keys of Sources and Signals that a template can see. Other keys can hold IDs."""

_UNIT_SYMBOLS = {"standard_gravity": "g"}
"""Symbols that differ from pint's. pint writes standard gravity as ``g_0``."""


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One selected input annotation, as text."""

    key: str
    text: str
    """The annotation's text, or one numbered line per item of a text list."""
    is_list: bool

    @property
    def line(self) -> str:
        """The key and the text, such as ``age_band: 38-52``. A text list starts on the next line."""
        return f"{self.key}:\n{self.text}" if self.is_list else f"{self.key}: {self.text}"


@dataclass(frozen=True, slots=True)
class ChannelView:
    """One Signal of an input Record, cut to the task's window."""

    name: str
    spec_name: str
    spec_type: str
    unit: str | None
    sampling_rate_hz: int | float | None
    """Values per second, or ``None`` if the Signal is not sampled regularly."""
    n_values: int
    """The number of values in the window."""
    context: tuple[ContextItem, ...]
    """The selected input annotations on this Signal."""

    @property
    def unit_symbol(self) -> str:
        """The unit's symbol, such as ``mV``, or an empty string without a unit."""
        if self.unit is None:
            return ""
        return _UNIT_SYMBOLS.get(self.unit) or f"{ureg.Unit(self.unit):~P}"


@dataclass(frozen=True, slots=True)
class SourceView:
    """One Source of an input Record that has a channel in the window."""

    name: str
    metadata: JsonMapping
    """Only the keys in ``ALLOWED_METADATA_KEYS``."""
    channels: tuple[ChannelView, ...]


@dataclass(frozen=True, slots=True)
class RecordView:
    """One input Record."""

    sources: tuple[SourceView, ...]
    context: tuple[ContextItem, ...]
    """The selected input annotations on the Record, its Sources, or Signals outside the window."""

    @property
    def channels(self) -> tuple[ChannelView, ...]:
        """The channels of all Sources, in order."""
        return tuple(channel for source in self.sources for channel in source.channels)


@dataclass(frozen=True, slots=True)
class DatasetView:
    """The dataset card's name and description, if the caller passes a card."""

    name: str | None
    description: str | None


@dataclass(frozen=True, slots=True)
class WindowView:
    """The task's window."""

    duration_s: float | None
    """The duration of a time scope, or ``None`` without a scope or for a scope in steps."""


@dataclass(frozen=True, slots=True)
class LabelView:
    """One label of the label set."""

    name: str
    description: str | None


@dataclass(frozen=True, slots=True)
class LabelSetView:
    """The task's label set."""

    name: str | None
    """The label set's ``name`` annotation."""
    description: str | None
    """The label set's ``description`` annotation."""
    multi_label: bool
    labels: tuple[LabelView, ...]

    @property
    def options_line(self) -> str:
        """The label names on one line, separated by commas.

        Raises:
            TimeFValidationError: If a name holds a comma or a line break.
        """
        for label in self.labels:
            if "," in label.name or "\n" in label.name:
                raise TimeFValidationError(
                    f"label {label.name!r} holds a comma or a line break, so one line would list it as several options"
                )
        return ", ".join(label.name for label in self.labels)


@dataclass(frozen=True, slots=True)
class ClassificationInputView:
    """What the input section of a template sees."""

    dataset: DatasetView
    window: WindowView
    records: tuple[RecordView, ...]
    """The input Records, in the task's order."""
    context: tuple[ContextItem, ...]
    """The selected input annotations outside the input Records."""
    prompt: str | None
    """The task's own prompt."""
    label_set: LabelSetView


@dataclass(frozen=True, slots=True)
class ClassificationTargetView(ClassificationInputView):
    """What the target section of a template sees: the input view, the labels and the rationale."""

    targets: tuple[str, ...]
    rationale: str | None


class TaskView(NamedTuple):
    """The input view, and the windowed Signal of each channel."""

    values: ClassificationInputView
    windows: dict[int, Signal]
    """The stripped window of each channel, by the ``id()`` of its ``ChannelView``."""


def build_input_view(task: Task, dataset: DatasetMetadata | None = None) -> TaskView:
    """Build the input view of a classification task.

    The views follow Aionic's renderer: each Signal claims its selected annotations first, then each
    Record claims those on itself and its Sources, and the rest go to the top-level context.

    Args:
        task: The task.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The view and the stripped window of each channel.

    Raises:
        TimeFValidationError: If the task is not a classification task, its scope is a point or covers
            no input Signal, its inputs are empty or repeat, an annotation is input and target, a selected
            annotation is not text without a span, or its prompt is blank.
    """
    if not isinstance(task, ClassificationTask):
        raise TimeFValidationError(f"prompt templates support ClassificationTask only, got {type(task).__name__}")
    scope = task.scope
    if scope is not None and scope.is_point:
        raise TimeFValidationError("a task scoped to a point names no window of its inputs")
    _check_inputs(task)
    prompt = None if task.prompt is None else task.prompt.read()
    if prompt is not None and not prompt.strip():
        raise TimeFValidationError("the task's prompt is blank")

    pending = list(task.input_annotations)
    windows: dict[int, Signal] = {}
    records = []
    for record in task.inputs:
        sources = []
        for source in record.walk_sources():
            channels = []
            for signal in sorted(source.signals, key=lambda signal: signal.name):
                window = _window(signal, scope)
                if window is None:
                    continue
                channel = ChannelView(
                    name=signal.name,
                    spec_name=signal.spec.name,
                    spec_type=signal.spec.spec_type,
                    unit=normalize_unit(signal.spec.unit_value),
                    sampling_rate_hz=_sampling_rate_hz(signal.time_axis),
                    n_values=window.n_values,
                    context=_context(_claim(pending, signal.annotations)),
                )
                windows[id(channel)] = _stripped(window)
                channels.append(channel)
            if channels:
                sources.append(
                    SourceView(name=source.name, metadata=_allowed(source.metadata), channels=tuple(channels))
                )
        context = _context(_claim(pending, record.walk_annotations()))
        records.append(RecordView(sources=tuple(sources), context=context))
    if scope is not None and not windows:
        raise TimeFValidationError(f"the task's scope {scope!r} covers no input signal")

    values = ClassificationInputView(
        dataset=DatasetView(
            name=None if dataset is None else dataset.name,
            description=None if dataset is None else dataset.description,
        ),
        window=WindowView(
            duration_s=(scope.end_us - scope.start_us) / US_PER_S if isinstance(scope, TimeInterval) else None
        ),
        records=tuple(records),
        context=_context(pending),
        prompt=prompt,
        label_set=_label_set_view(task.label_set),
    )
    return TaskView(values=values, windows=windows)


def build_target_view(task: ClassificationTask, view: ClassificationInputView) -> ClassificationTargetView:
    """Add the task's labels and rationale to its input view.

    Returns:
        The target view.

    Raises:
        TimeFValidationError: If the task has both targets and target annotations or neither, a label is not
            text, is blank, holds a line break or repeats, or the rationale is blank.
    """
    labels = _target_labels(task)
    for label in labels:
        if "\n" in label:
            raise TimeFValidationError(f"label {label!r} holds a line break, but the target lists one label per line")
    if len(set(labels)) != len(labels):
        raise TimeFValidationError(f"the target repeats a label: {labels!r}")
    rationale = None if task.rationale is None else task.rationale.read()
    if rationale is not None and not rationale.strip():
        raise TimeFValidationError("the task's rationale is blank")
    inputs = {field.name: getattr(view, field.name) for field in fields(view)}
    return ClassificationTargetView(**inputs, targets=tuple(labels), rationale=rationale)


def _check_inputs(task: Task) -> None:
    """Make sure that the task has unique input Records and no annotation as input and target.

    Raises:
        TimeFValidationError: If a check fails.
    """
    if not task.inputs:
        raise TimeFValidationError("the task needs at least one input Record")
    if len({id(record) for record in task.inputs}) != len(task.inputs):
        raise TimeFValidationError("the task lists an input Record more than once")
    targets = {id(annotation) for annotation in task.target_annotations}
    for annotation in task.input_annotations:
        if id(annotation) in targets:
            raise TimeFValidationError(f"annotation {annotation.key!r} cannot be used as input and target")


def _claim(pending: list[Annotation], owned: Iterable[Annotation]) -> list[Annotation]:
    """Remove the pending annotations that are among ``owned``, compared by identity.

    Returns:
        The removed annotations, in the order of selection.
    """
    owned_ids = {id(annotation) for annotation in owned}
    claimed = [annotation for annotation in pending if id(annotation) in owned_ids]
    pending[:] = [annotation for annotation in pending if id(annotation) not in owned_ids]
    return claimed


def _context(annotations: Iterable[Annotation]) -> tuple[ContextItem, ...]:
    """Turn selected input annotations into context items.

    Returns:
        One item per annotation.

    Raises:
        TimeFValidationError: If an annotation has a span or holds no text, blank text, an empty text list,
            or a list item that is blank or holds a line break.
    """
    items = []
    for annotation in annotations:
        key = annotation.key
        if annotation.span is not None:
            raise TimeFValidationError(f"input annotation {key!r} has a span, and templates take none")
        value = annotation.value
        if isinstance(value, TextListHandle):
            entries = value.read()
            if not entries or any(not entry.strip() for entry in entries):
                raise TimeFValidationError(f"input annotation {key!r} needs at least one item, each with text")
            if any("\n" in entry for entry in entries):
                raise TimeFValidationError(f"input annotation {key!r} has an item with a line break")
            text = "\n".join(f"{number}. {entry}" for number, entry in enumerate(entries, start=1))
            items.append(ContextItem(key=key, text=text, is_list=True))
            continue
        if not isinstance(value, TextHandle):
            raise TimeFValidationError(f"input annotation {key!r} holds {type(value).__name__}, not text")
        text = value.read()
        if not text.strip():
            raise TimeFValidationError(f"input annotation {key!r} is blank")
        items.append(ContextItem(key=key, text=text, is_list=False))
    return tuple(items)


def _window(signal: Signal, scope: Span | None) -> Signal | None:
    """Cut a Signal to the scope, as Aionic does.

    Returns:
        The Signal without a scope, its part inside the scope, or ``None`` if the scope leaves it out.
    """
    if scope is None:
        return signal
    if not signal.is_selected_by(scope):
        return None
    if isinstance(scope, TimeInterval) and scope.time_series is None and not signal.overlaps_with(scope):
        return None
    return signal.slice(scope, clip=True)


def _stripped(window: Signal) -> Signal:
    """Copy a window without annotations, with allowed metadata only, on a time axis that starts at zero.

    Returns:
        A lazy Signal that reads the window's values.
    """
    axis: TimeAxis = window.time_axis
    time_offsets = None
    if isinstance(axis, RegularAxis):
        axis = RegularAxis(period_us=axis.period_us)
    elif isinstance(axis, IrregularAxis):
        offsets = window.time_offsets_us()
        first = int(offsets[0])
        axis = IrregularAxis(first_us=0, last_us=axis.last_us - first)
        shifted = pa.array(offsets - first, type=pa.int64())

        def time_offsets() -> pa.Array:
            return shifted

    return Signal.from_loader(
        spec=window.spec,
        time_axis=axis,
        name=window.name,
        loader=lambda: window.read_steps(0, window.n_values),
        n_values=window.n_values,
        time_offsets_loader=time_offsets,
        metadata=_allowed(window.metadata),
    )


def _allowed(metadata: JsonMapping) -> JsonMapping:
    """Keep only the metadata keys in ``ALLOWED_METADATA_KEYS``.

    Returns:
        The kept keys and values.
    """
    return {key: metadata[key] for key in ALLOWED_METADATA_KEYS if key in metadata}


def _sampling_rate_hz(axis: TimeAxis) -> int | float | None:
    """Return the rate of a regular axis, as an int if it is whole.

    Returns:
        Values per second, or ``None`` for an axis that is not regular.
    """
    if not isinstance(axis, RegularAxis):
        return None
    rate = Fraction(US_PER_S) / axis.period_us
    return rate.numerator if rate.denominator == 1 else float(rate)


def _label_set_view(label_set: LabelSet) -> LabelSetView:
    """Build the view of a label set.

    Returns:
        The view, with the text of the ``name`` and ``description`` annotations.
    """
    texts = {
        annotation.key: annotation.value.read()
        for annotation in reversed(label_set.annotations)
        if isinstance(annotation.value, TextHandle)
    }
    return LabelSetView(
        name=texts.get("name"),
        description=texts.get("description"),
        multi_label=label_set.multi_label,
        labels=tuple(LabelView(name=label.name, description=label.description) for label in label_set.labels),
    )


def _target_labels(task: ClassificationTask) -> list[str]:
    """Read the task's labels from its targets or its target annotations.

    Returns:
        The labels, in the order that the task stores them.

    Raises:
        TimeFValidationError: If the task has both kinds of targets or neither, a target annotation has a span or
            holds a text list, or a label is not text or is blank.
    """
    if task.targets is not None and task.target_annotations:
        raise TimeFValidationError("the task cannot have both targets and target annotations")
    if task.targets is not None:
        values = list(task.targets)
    elif task.target_annotations:
        for annotation in task.target_annotations:
            if annotation.span is not None:
                raise TimeFValidationError(f"target annotation {annotation.key!r} has a span, and templates take none")
            if isinstance(annotation.value, TextListHandle):
                raise TimeFValidationError(f"target annotation {annotation.key!r} holds a text list, not one label")
        values = [annotation.value for annotation in task.target_annotations]
    else:
        raise TimeFValidationError("the task needs targets or target annotations")
    labels = []
    for value in values:
        if not isinstance(value, TextHandle):
            raise TimeFValidationError(f"each label must be text, got {type(value).__name__}")
        label = value.read()
        if not label.strip():
            raise TimeFValidationError("a label is blank")
        labels.append(label)
    return labels
