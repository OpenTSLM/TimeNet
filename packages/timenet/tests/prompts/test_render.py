from dataclasses import replace

import pytest
from typing_extensions import override

from timenet.errors import TimeFValidationError
from timenet.prompts import PromptBuilder, PromptRenderer, SeriesPart, SeriesTextPart, TextPart, render, render_input
from timenet.prompts.views import ClassificationInputView, ClassificationTargetView
from timenet.testing import make_dataset
from timenet.types import AnswerTask


class _Question(PromptRenderer):
    """Asks a question without series and replies with the labels."""

    name = "question"

    @override
    def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
        return PromptBuilder().text("Question?")

    @override
    def render_target_prompt(self, view: ClassificationTargetView) -> PromptBuilder:
        return PromptBuilder().text(*view.targets)


def test_series_and_text_parts(capture):
    class Series(_Question):
        @override
        def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
            prompt = PromptBuilder()
            for channel in view.records[0].channels:
                prompt.series(channel, text="\n".join(item.line for item in channel.context))
            return prompt.text("Question?")

    result = render(capture.task, Series())

    first, second, third, question = result.input
    assert isinstance(first, SeriesTextPart)
    assert first.text == "axis_note: x points along the forearm"
    assert first.series.n_values == 1000
    assert first.series.annotations == ()
    assert isinstance(second, SeriesPart)
    assert isinstance(third, SeriesPart)
    assert question == TextPart("Question?")
    assert result.target == (TextPart("sleep"),)


def test_text_between_series_merges_into_one_part(capture):
    class Lines(_Question):
        @override
        def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
            prompt = PromptBuilder().text("one", "two").text("  ")
            return prompt.series(view.records[0].channels[0]).text("", "three", "")

    parts = render_input(capture.task, Lines())

    assert [type(part) for part in parts] == [TextPart, SeriesPart, TextPart]
    assert (parts[0], parts[2]) == (TextPart("one\ntwo"), TextPart("three"))


def test_render_input_does_not_read_targets(capture):
    task = capture.replace(targets=("light", "light"), label_set=capture.label_set(multi_label=True))

    assert render_input(task, _Question()) == (TextPart("Question?"),)
    with pytest.raises(TimeFValidationError, match="repeats"):
        render(task, _Question())


def test_series_needs_a_channel_of_the_view(capture):
    class Copy(_Question):
        @override
        def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
            return PromptBuilder().series(replace(view.records[0].channels[0]))

    with pytest.raises(TimeFValidationError, match="channel"):
        render_input(capture.task, Copy())


def test_dataset_card_reaches_the_view(capture):
    class Dataset(_Question):
        @override
        def render_input_prompt(self, view: ClassificationInputView) -> PromptBuilder:
            return PromptBuilder().text(view.dataset.name or "")

    assert render_input(capture.task, Dataset(), dataset=make_dataset().metadata) == (TextPart("Hello World"),)


def test_other_task_types_are_rejected(capture):
    task = AnswerTask(inputs=[capture.record], prompt="What?", targets="sleep")

    with pytest.raises(TimeFValidationError, match="ClassificationTask"):
        render_input(task, _Question())
