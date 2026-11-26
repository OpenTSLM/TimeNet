"""Render a task with a prompt template into model-neutral parts.

Jinja renders each section in its immutable sandbox. The ``series()`` helper puts a marker into the text,
and the renderer splits the text at the markers into text parts and series parts. The marker holds a
random token for each render, so text in a prompt or an annotation cannot forge a series part.
"""

from dataclasses import dataclass, fields
from functools import lru_cache
import re
import secrets
from typing import TypeAlias, cast

from jinja2 import StrictUndefined, Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from timenet.dataset import Signal
from timenet.errors import TimeFValidationError
from timenet.prompts.views import ChannelView, ClassificationInputView, build_input_view, build_target_view
from timenet.types import ClassificationTask, DatasetMetadata, Task


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompt template with one Jinja template for each section."""

    name: str
    input: str
    """Jinja template for what the model receives. It sees only the input view."""
    target: str
    """Jinja template for the expected reply. It sees the target view."""


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


def render_input(task: Task, template: PromptTemplate, *, dataset: DatasetMetadata | None = None) -> tuple[Part, ...]:
    """Render only the input section. This never reads the targets or the rationale.

    Args:
        task: A classification task.
        template: The template.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts.

    Raises:
        TimeFValidationError: If the task does not fit the template or breaks a rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by build_input_view and _render
    view = build_input_view(task, dataset)
    return _render(template.name, "input", template.input, view.values, view.windows)


def render(task: Task, template: PromptTemplate, *, dataset: DatasetMetadata | None = None) -> RenderedPrompt:
    """Render the input and the expected reply, for training.

    Args:
        task: A classification task.
        template: The template.
        dataset: The dataset card, for the dataset's name and description.

    Returns:
        The input parts and the target parts.

    Raises:
        TimeFValidationError: If the task does not fit the template or breaks a rule of the views.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the view builders and _render
    view = build_input_view(task, dataset)
    # build_input_view accepts only classification tasks.
    target_view = build_target_view(cast("ClassificationTask", task), view.values)
    return RenderedPrompt(
        input=_render(template.name, "input", template.input, view.values, view.windows),
        target=_render(template.name, "target", template.target, target_view, view.windows),
    )


def _required(value: object, what: str) -> object:
    """Return the value, or fail if it is missing or blank, for templates that need a field.

    Returns:
        The value.

    Raises:
        TimeFValidationError: If the value is ``None`` or blank text.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise TimeFValidationError(f"the template needs {what}")
    return value


_ENVIRONMENT = ImmutableSandboxedEnvironment(
    undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True
)
_ENVIRONMENT.filters["required"] = _required


@lru_cache(maxsize=256)
def _compile(source: str) -> Template:
    """Compile a template once.

    Returns:
        The compiled template.
    """
    return _ENVIRONMENT.from_string(source)


def _render(
    name: str, section: str, source: str, values: ClassificationInputView, windows: dict[int, Signal]
) -> tuple[Part, ...]:
    """Render one section over a view and split it into parts.

    Returns:
        The parts, in order. Text between two series is stripped, and empty text is left out.

    Raises:
        TimeFValidationError: If the template fails, for example on a field that the view does not have.
    """
    token = secrets.token_hex(16)
    placed: list[tuple[Signal, str]] = []

    def series(channel: object, text: str = "") -> str:
        window = windows.get(id(channel)) if isinstance(channel, ChannelView) else None
        if window is None:
            raise TimeFValidationError("series() takes a channel of the task's view, such as records[0].channels[0]")
        placed.append((window, text))
        return f"<<series:{token}:{len(placed) - 1}>>"

    context = {field.name: getattr(values, field.name) for field in fields(values)}
    try:
        text = _compile(source).render(**context, series=series)
    except TemplateError as exc:
        raise TimeFValidationError(f"template {name!r}, {section} section: {exc}") from exc

    parts: list[Part] = []
    for index, piece in enumerate(re.split(rf"<<series:{token}:(\d+)>>", text)):
        if index % 2:
            window, annotations = placed[int(piece)]
            parts.append(SeriesTextPart(window, annotations) if annotations else SeriesPart(window))
        elif piece.strip():
            parts.append(TextPart(piece.strip()))
    return tuple(parts)
