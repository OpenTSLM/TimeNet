"""Render a task with a prompt renderer into model-neutral parts.

A renderer is a subclass of :class:`PromptRenderer`. Its methods get a view of the task, never the task,
and return a :class:`PromptBuilder` with text and series.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self, TypeAlias

from timenet.dataset import Signal
from timenet.errors import TimeFValidationError
from timenet.prompts.views import (
    ChannelView,
    ClassificationInputView,
    ClassificationTargetView,
    TaskView,
    build_input_view,
    build_target_view,
)
from timenet.types import ClassificationTask, DatasetMetadata, Task


@dataclass(frozen=True)
class TextPart:
    """Text for the model."""

    text: str


@dataclass(frozen=True)
class SeriesPart:
    """One channel's window: a stripped copy of the Signal."""

    series: Signal


@dataclass(frozen=True)
class SeriesTextPart:
    """One channel's window, with the text of the annotations on its Signal."""

    series: Signal
    text: str


PromptPart: TypeAlias = TextPart | SeriesPart | SeriesTextPart


@dataclass(frozen=True)
class RenderedPrompt:
    """The parts of the input and of the expected reply."""

    input: tuple[PromptPart, ...]
    target: tuple[PromptPart, ...]


@dataclass(frozen=True)
class _Series:
    """A channel that the builder resolves to its window when the prompt is built."""

    channel: ChannelView
    text: str


class PromptBuilder:
    """Collect the lines of text and the series of one prompt, in order.

    A renderer adds a channel of its view as a reference. :func:`render` swaps each channel for its window.
    Lines of text between two series become one text part. Its lines are joined with line breaks, and the
    part is stripped. Text that is blank after stripping is left out.
    """

    def __init__(self) -> None:
        self._items: list[str | _Series] = []

    def add_text(self, *lines: str) -> Self:
        """Add lines of text to the prompt."""
        self._items.extend(lines)
        return self

    def add_series(self, channel: ChannelView, text: str = "") -> Self:
        """Add a channel's window.

        Args:
            channel: A channel of the view that the renderer got.
            text: Text that belongs to this series, such as the selected annotations on its Signal. With
                text, the part is a ``SeriesTextPart`` that keeps the two together. Without it, the part is
                a ``SeriesPart``.

        Returns:
            The builder, so that calls can be chained.
        """
        self._items.append(_Series(channel, text))
        return self

    def build(self, windows: dict[int, Signal]) -> tuple[PromptPart, ...]:
        """Resolve each channel to its window and return the parts. :func:`render` calls this.

        Args:
            windows: The window of each channel, by the ``id()`` of its ``ChannelView``.

        Returns:
            The parts, in the order that the renderer added them.

        Raises:
            TimeFValidationError: If a channel is not a channel of the view that the renderer got.
        """
        parts: list[PromptPart] = []
        lines: list[str] = []

        def flush() -> None:
            text = "\n".join(lines).strip()
            lines.clear()
            if text:
                parts.append(TextPart(text))

        for item in self._items:
            if isinstance(item, str):
                lines.append(item)
                continue
            window = windows.get(id(item.channel))
            if window is None:
                raise TimeFValidationError(
                    "add_series() takes a channel of the task's view, such as input_records[0].channels[0]"
                )
            flush()
            parts.append(SeriesTextPart(window, item.text) if item.text else SeriesPart(window))
        flush()
        return tuple(parts)


class PromptRenderer(ABC):
    """A named prompt renderer. A subclass sets ``name`` and builds the prompt of each section."""

    name: ClassVar[str]
    """The renderer's name. Log it with the installed TimeNet version, which fixes the renderer's text."""

    @abstractmethod
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        """Build what the model receives. The input view has no targets and no rationale."""

    @abstractmethod
    def render_target_prompt(self, view: ClassificationTargetView) -> PromptBuilder:
        """Build the expected reply."""


def render_input(
    task: Task, renderer: PromptRenderer, *, dataset: DatasetMetadata | None = None
) -> tuple[PromptPart, ...]:
    """Render only the input. This never reads the targets or the rationale.

    Args:
        task: A classification task.
        renderer: The renderer.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts.

    Raises:
        TimeFValidationError: If the task is not a classification task, does not fit the renderer, or breaks a
            rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _render_input
    return _render_input(_classification(task), renderer, dataset)[1]


def render(task: Task, renderer: PromptRenderer, *, dataset: DatasetMetadata | None = None) -> RenderedPrompt:
    """Render the input and the expected reply, for training.

    Args:
        task: A classification task.
        renderer: The renderer.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts and the target parts.

    Raises:
        TimeFValidationError: If the task is not a classification task, does not fit the renderer, or breaks a
            rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers
    task = _classification(task)
    view, input_parts = _render_input(task, renderer, dataset)
    target_parts = renderer.render_target_prompt(build_target_view(task, view.values)).build(view.windows)
    return RenderedPrompt(input=input_parts, target=target_parts)


def _classification(task: Task) -> ClassificationTask:
    """Return the task as a classification task.

    Raises:
        TimeFValidationError: If it is another task type.
    """
    if not isinstance(task, ClassificationTask):
        raise TimeFValidationError(f"prompt renderers support ClassificationTask only, got {type(task).__name__}")
    return task


def _render_input(
    task: ClassificationTask, renderer: PromptRenderer, dataset: DatasetMetadata | None
) -> tuple[TaskView, tuple[PromptPart, ...]]:
    """Build the input view and render the input.

    Returns:
        The view, for the target, and the input parts.
    """
    view = build_input_view(task, dataset)
    return view, renderer.render_input_prompt(view.values).build(view.windows)
