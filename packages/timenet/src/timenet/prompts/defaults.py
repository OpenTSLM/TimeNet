"""TimeNet's default prompt renderers for classification tasks.

The consumer chooses the renderer: none of them chooses between the task's own prompt and the question from
the label set.
"""

from abc import abstractmethod

from typing_extensions import override

from timenet.errors import TimeFValidationError
from timenet.prompts.render import PromptBuilder, PromptRenderer
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


def _annotation_text(channel: ChannelView) -> str:
    """Return the selected annotations on a channel's Signal, one per line."""
    return "\n".join(_lines(channel.context))


def _inputs(view: ClassificationInputView) -> PromptBuilder:
    """Start a prompt with Aionic's layout.

    Each Record's series come with the text of their annotations, then the Record's other selected
    annotations. The selected annotations outside the input Records come last.

    Returns:
        The builder.
    """
    prompt = PromptBuilder()
    for record in view.records:
        for channel in record.channels:
            prompt.series(channel, text=_annotation_text(channel))
        prompt.text(*_lines(record.context))
    return prompt.text(*_lines(view.context))


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
    """Aionic's layout, then a question that each subclass writes, and Aionic's reply."""

    @override
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        """Build the series and the selected annotations as Aionic does, then the question."""
        return _inputs(view).text(*self.question(view))

    @abstractmethod
    def question(self, view: ClassificationInputView) -> list[str]:
        """Return the lines of the question after the inputs."""

    @override
    def render_target_prompt(self, view: ClassificationTargetView) -> PromptBuilder:
        """Build Aionic's reply: the rationale and an empty line, then one label per line."""
        prompt = PromptBuilder()
        if view.rationale is not None:
            prompt.text(view.rationale, "")
        return prompt.text(*view.targets)


class _LabelQuestion(_AionicRenderer):
    name = "classification-label-question"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the question from the label set, with the options on one line."""
        return [f"Which label matches the input? {_options(view.label_set)}"]


class _PromptQuestion(_AionicRenderer):
    name = "classification-prompt-question"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the task's own prompt."""
        return [_prompt(view)]


class _PromptWithOptions(_AionicRenderer):
    name = "classification-prompt-options"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the task's own prompt, then give the options on one line."""
        return [_prompt(view), _options(view.label_set)]


class _Described(_AionicRenderer):
    name = "classification-described"

    @override
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        """Build the window, each Source and channel before its series, the annotations, then the question."""
        prompt = PromptBuilder()
        if view.window.duration_s:
            prompt.text(f"Input: {view.window.duration_s:g} s window.")
        for record in view.records:
            for source in record.sources:
                metadata = ", ".join(f"{key}: {value}" for key, value in source.metadata.items())
                prompt.text(f"Source: {source.name}" + (f" ({metadata})" if metadata else ""))
                for channel in source.channels:
                    prompt.text(_channel_line(channel)).series(channel, text=_annotation_text(channel))
            prompt.text(*_lines(record.context))
        return prompt.text(*_lines(view.context), *self.question(view))

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the question from the label set, with one option per line and its description."""
        options = [
            f"- {label.name}" + (f": {label.description}" if label.description else "")
            for label in view.label_set.labels
        ]
        return ["Question: Which label matches the input?", "Options:", *options, _answer(view.label_set)]


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
