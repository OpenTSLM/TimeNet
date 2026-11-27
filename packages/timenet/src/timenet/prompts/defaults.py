"""TimeNet's default prompt renderers for classification tasks.

Each name ends with a version. A change to a renderer's text gets a new version, so that a training run can
log which text it used. The consumer chooses the renderer: none of them chooses between the task's own
prompt and the question from the label set.
"""

from abc import abstractmethod

from typing_extensions import override

from timenet.errors import TimeFValidationError
from timenet.prompts.render import Parts, PromptRenderer
from timenet.prompts.views import (
    ChannelView,
    ClassificationInputView,
    ClassificationTargetView,
    ContextItem,
    LabelSetView,
)


def _lines(items: tuple[ContextItem, ...]) -> list[str]:
    """Return each context item as one line, such as ``age_band: 38-52``."""
    return [item.line for item in items]


def _series(parts: Parts, channel: ChannelView) -> None:
    """Add a channel's series with the text of the selected annotations on its Signal."""
    parts.series(channel, text="\n".join(_lines(channel.context)))


def _inputs(view: ClassificationInputView, parts: Parts) -> None:
    """Add Aionic's layout.

    Each Record's series come with the text of their annotations, then the Record's other selected
    annotations. The selected annotations outside the input Records come last.
    """
    for record in view.records:
        for channel in record.channels:
            _series(parts, channel)
        parts.text(*_lines(record.context))
    parts.text(*_lines(view.context))


def _answer(label_set: LabelSetView) -> str:
    """Return the sentence that says how many labels to answer with."""
    if label_set.multi_label:
        return "Answer with every label that applies, one per line."
    return "Answer with one label."


def _options(label_set: LabelSetView) -> str:
    """Return the options on one line and the answer sentence, as Aionic writes them."""
    return f"Options: {label_set.options_line}. {_answer(label_set)}"


def _prompt(view: ClassificationInputView) -> str:
    """Return the task's own prompt.

    Raises:
        TimeFValidationError: If the task has no prompt.
    """
    if view.prompt is None:
        raise TimeFValidationError("the renderer needs a task prompt, and the task has none")
    return view.prompt


def _channel_line(channel: ChannelView) -> str:
    """Describe a channel, such as ``Channel x: Wrist acceleration, unit g, 100 Hz.``."""
    details = [channel.spec_name or channel.spec_type]
    if channel.unit_symbol:
        details.append(f"unit {channel.unit_symbol}")
    if channel.sampling_rate_hz:
        details.append(f"{channel.sampling_rate_hz:g} Hz")
    return f"Channel {channel.name}: {', '.join(details)}."


class _AionicRenderer(PromptRenderer):
    """Aionic's layout, then a question that each subclass adds, and Aionic's reply."""

    @override
    def add_input(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the series and the selected annotations as Aionic does, then the question."""
        _inputs(view, parts)
        self.add_question(view, parts)

    @abstractmethod
    def add_question(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the question after the inputs."""

    @override
    def add_target(self, view: ClassificationTargetView, parts: Parts) -> None:
        """Add Aionic's reply: the rationale and an empty line, then one label per line."""
        if view.rationale is not None:
            parts.text(view.rationale, "")
        parts.text(*view.targets)


class _LabelQuestion(_AionicRenderer):
    name = "classification-label-question-v1"

    @override
    def add_question(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the question from the label set, with the options on one line."""
        parts.text(f"Which label matches the input? {_options(view.label_set)}")


class _PromptQuestion(_AionicRenderer):
    name = "classification-prompt-question-v1"

    @override
    def add_question(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the task's own prompt."""
        parts.text(_prompt(view))


class _PromptWithOptions(_AionicRenderer):
    name = "classification-prompt-options-v1"

    @override
    def add_question(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the task's own prompt, then the options on one line."""
        parts.text(_prompt(view), _options(view.label_set))


class _Described(_AionicRenderer):
    name = "classification-described-v1"

    @override
    def add_input(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the window, each Source and channel before its series, the annotations, then the question."""
        if view.window.duration_s:
            parts.text(f"Input: {view.window.duration_s:g} s window.")
        for record in view.records:
            for source in record.sources:
                metadata = ", ".join(f"{key}: {value}" for key, value in source.metadata.items())
                parts.text(f"Source: {source.name}" + (f" ({metadata})" if metadata else ""))
                for channel in source.channels:
                    parts.text(_channel_line(channel))
                    _series(parts, channel)
            parts.text(*_lines(record.context))
        parts.text(*_lines(view.context))
        self.add_question(view, parts)

    @override
    def add_question(self, view: ClassificationInputView, parts: Parts) -> None:
        """Add the question from the label set, with one option per line and its description."""
        parts.text("Question: Which label matches the input?", "Options:")
        for label in view.label_set.labels:
            parts.text(f"- {label.name}" + (f": {label.description}" if label.description else ""))
        parts.text(_answer(view.label_set))


LABEL_QUESTION = _LabelQuestion()
"""Aionic's question from the label set. It ignores the task's own prompt."""

PROMPT_QUESTION = _PromptQuestion()
"""Aionic's use of the task's own prompt as the question. It fails for a task without a prompt."""

PROMPT_WITH_OPTIONS = _PromptWithOptions()
"""The task's own prompt, then the options. It fails for a task without a prompt."""

DESCRIBED = _Described()
"""The window, each Source and channel with its quantity, unit and rate, then the options with their
descriptions."""

DEFAULT_RENDERERS: tuple[PromptRenderer, ...] = (LABEL_QUESTION, PROMPT_QUESTION, PROMPT_WITH_OPTIONS, DESCRIBED)
"""All default renderers for classification tasks."""
