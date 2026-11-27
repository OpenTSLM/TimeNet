"""Render a task with a prompt renderer into model-neutral parts.

A renderer is a subclass of :class:`PromptRenderer`. Its methods get a view of the task, never the task,
and add text and series to a :class:`Parts` builder.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, TypeAlias, cast

from timenet.dataset import Signal
from timenet.errors import TimeFValidationError
from timenet.prompts.views import (
    ChannelView,
    ClassificationInputView,
    ClassificationTargetView,
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


Part: TypeAlias = TextPart | SeriesPart | SeriesTextPart


@dataclass(frozen=True)
class RenderedPrompt:
    """The parts of the input and of the expected reply."""

    input: tuple[Part, ...]
    target: tuple[Part, ...]


class Parts:
    """Collect the parts that a renderer adds, in order.

    Lines of text between two series become one text part. Its lines are joined with line breaks, and
    the part is stripped. Text that is blank after stripping is left out.
    """

    def __init__(self, windows: dict[int, Signal]) -> None:
        self._windows = windows
        self._parts: list[Part] = []
        self._lines: list[str] = []

    def text(self, *lines: str) -> None:
        """Add lines of text."""
        self._lines.extend(lines)

    def series(self, channel: ChannelView, text: str = "") -> None:
        """Add a channel's window, with ``text`` if it is not empty.

        Raises:
            TimeFValidationError: If the channel is not a channel of the view that the renderer got.
        """
        window = self._windows.get(id(channel))
        if window is None:
            raise TimeFValidationError("series() takes a channel of the task's view, such as records[0].channels[0]")
        self._flush()
        self._parts.append(SeriesTextPart(window, text) if text else SeriesPart(window))

    def build(self) -> tuple[Part, ...]:
        """Return the parts.

        Returns:
            The parts, in the order that the renderer added them.
        """
        self._flush()
        return tuple(self._parts)

    def _flush(self) -> None:
        """Turn the pending lines into a text part."""
        text = "\n".join(self._lines).strip()
        self._lines = []
        if text:
            self._parts.append(TextPart(text))


class PromptRenderer(ABC):
    """A named prompt renderer. A subclass sets ``name`` and adds the parts of each section."""

    name: ClassVar[str]
    """The renderer's name, with a version such as ``-v1``, so that a training run can log it."""

    @abstractmethod
    def add_input(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add what the model receives. The input view has no targets and no rationale."""

    @abstractmethod
    def add_target(self, view: ClassificationTargetView, parts: Parts) -> None:
        """Add the expected reply."""


def render_input(task: Task, renderer: PromptRenderer, *, dataset: DatasetMetadata | None = None) -> tuple[Part, ...]:
    """Render only the input. This never reads the targets or the rationale.

    Args:
        task: A classification task.
        renderer: The renderer.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts.

    Raises:
        TimeFValidationError: If the task does not fit the renderer or breaks a rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by build_input_view and the renderer
    view = build_input_view(task, dataset)
    parts = Parts(view.windows)
    renderer.add_input(view.values, parts)
    return parts.build()


def render(task: Task, renderer: PromptRenderer, *, dataset: DatasetMetadata | None = None) -> RenderedPrompt:
    """Render the input and the expected reply, for training.

    Args:
        task: A classification task.
        renderer: The renderer.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts and the target parts.

    Raises:
        TimeFValidationError: If the task does not fit the renderer or breaks a rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the view builders and the renderer
    view = build_input_view(task, dataset)
    # build_input_view accepts only classification tasks.
    target_view = build_target_view(cast("ClassificationTask", task), view.values)
    input_parts, target_parts = Parts(view.windows), Parts(view.windows)
    renderer.add_input(view.values, input_parts)
    renderer.add_target(target_view, target_parts)
    return RenderedPrompt(input=input_parts.build(), target=target_parts.build())
