from dataclasses import replace

import pytest

from timenet.errors import TimeFValidationError
from timenet.prompts import Parts, PromptRenderer, SeriesPart, SeriesTextPart, TextPart, render, render_input
from timenet.prompts.views import ClassificationInputView, ClassificationTargetView
from timenet.testing import make_dataset


def _labels(view: ClassificationTargetView, parts: Parts) -> None:
    parts.text(*view.targets)


def _renderer(input) -> PromptRenderer:
    return PromptRenderer(name="r", input=input, target=_labels)


def test_series_and_text_parts(capture):
    def input(view: ClassificationInputView, parts: Parts) -> None:
        for channel in view.records[0].channels:
            parts.series(channel, text="\n".join(item.line for item in channel.context))
        parts.text("Question?")

    result = render(capture.task, _renderer(input))

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
    def input(view: ClassificationInputView, parts: Parts) -> None:
        parts.text("one", "two")
        parts.text("  ")
        parts.series(view.records[0].channels[0])
        parts.text("", "three", "")

    parts = render_input(capture.task, _renderer(input))

    assert [type(part) for part in parts] == [TextPart, SeriesPart, TextPart]
    assert (parts[0], parts[2]) == (TextPart("one\ntwo"), TextPart("three"))


def test_render_input_does_not_read_targets(capture):
    task = capture.replace(targets=("light", "light"), label_set=capture.label_set(multi_label=True))
    renderer = _renderer(lambda view, parts: parts.text("Question?"))

    assert render_input(task, renderer) == (TextPart("Question?"),)
    with pytest.raises(TimeFValidationError, match="repeats"):
        render(task, renderer)


def test_series_needs_a_channel_of_the_view(capture):
    def input(view: ClassificationInputView, parts: Parts) -> None:
        parts.series(replace(view.records[0].channels[0]))

    with pytest.raises(TimeFValidationError, match="channel"):
        render_input(capture.task, _renderer(input))


def test_dataset_card_reaches_the_view(capture):
    renderer = _renderer(lambda view, parts: parts.text(view.dataset.name or ""))

    assert render_input(capture.task, renderer, dataset=make_dataset().metadata) == (TextPart("Hello World"),)
