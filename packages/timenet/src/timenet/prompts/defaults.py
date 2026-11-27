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
    LabelSetView,
)


def _annotation_text(channel: ChannelView) -> str:
    """Return the selected annotations on a channel's Signal, one per line."""
    return "\n".join(item.line for item in channel.context)


def _inputs(view: ClassificationInputView) -> PromptBuilder:
    """Start a prompt with Aionic's layout.

    Each Record's series come with the text of their annotations, then the Record's other selected
    annotations. The selected annotations outside the input Records come last.

    Returns:
        The builder.
    """
    prompt = PromptBuilder()
    for record in view.input_records:
        for channel in record.channels:
            prompt.add_series(channel, text=_annotation_text(channel))
        for item in record.context:
            prompt.add_text(item.line)
    for item in view.context:
        prompt.add_text(item.line)
    return prompt


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


class _DefaultRendererBase(PromptRenderer):
    """The base of the default renderers: Aionic's layout, then a question from each subclass, and Aionic's reply."""

    @override
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        """Build the series and the selected annotations as Aionic does, then the question."""
        return _inputs(view).add_text(*self.question(view))

    @abstractmethod
    def question(self, view: ClassificationInputView) -> list[str]:
        """Return the lines of the question after the inputs."""

    @override
    def render_target_prompt(self, view: ClassificationTargetView) -> PromptBuilder:
        """Build Aionic's reply: the rationale and an empty line, then one label per line."""
        prompt = PromptBuilder()
        if view.rationale is not None:
            prompt.add_text(view.rationale, "")
        return prompt.add_text(*view.targets)


class LabelQuestionRenderer(_DefaultRendererBase):
    """Aionic's question from the label set, with the options on one line. It ignores the task's own prompt."""

    name = "classification-label-question"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the question from the label set, with the options on one line."""
        return [f"Which label matches the input? {_options(view.label_set)}"]


class PromptQuestionRenderer(_DefaultRendererBase):
    """Aionic's use of the task's own prompt as the question. It fails for a task without a prompt."""

    name = "classification-prompt-question"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the task's own prompt."""
        return [_prompt(view)]


class PromptWithOptionsRenderer(_DefaultRendererBase):
    """The task's own prompt, then the options on one line. It fails for a task without a prompt."""

    name = "classification-prompt-options"

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the task's own prompt, then give the options on one line."""
        return [_prompt(view), _options(view.label_set)]


class DescribedRenderer(_DefaultRendererBase):
    """The window, each Source and channel with its unit and rate, then the options with their descriptions."""

    name = "classification-described"

    @override
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        """Build the window, each Source and channel before its series, the annotations, then the question."""
        prompt = PromptBuilder()
        if view.window.duration_s:
            prompt.add_text(f"Input: {view.window.duration_s:g} s window.")
        for record in view.input_records:
            for source in record.sources:
                metadata = ", ".join(f"{key}: {value}" for key, value in source.metadata.items())
                prompt.add_text(f"Source: {source.name}" + (f" ({metadata})" if metadata else ""))
                for channel in source.channels:
                    prompt.add_text(_channel_line(channel)).add_series(channel, text=_annotation_text(channel))
            for item in record.context:
                prompt.add_text(item.line)
        for item in view.context:
            prompt.add_text(item.line)
        return prompt.add_text(*self.question(view))

    @override
    def question(self, view: ClassificationInputView) -> list[str]:
        """Ask the question from the label set, with one option per line and its description."""
        options = [
            f"- {label.name}" + (f": {label.description}" if label.description else "")
            for label in view.label_set.labels
        ]
        return ["Question: Which label matches the input?", "Options:", *options, _answer(view.label_set)]


DEFAULT_RENDERERS: tuple[PromptRenderer, ...] = (
    LabelQuestionRenderer(),
    PromptQuestionRenderer(),
    PromptWithOptionsRenderer(),
    DescribedRenderer(),
)
"""One instance of each default renderer for classification tasks."""
