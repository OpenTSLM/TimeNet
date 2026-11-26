import pytest

from timenet.errors import TimeFValidationError
from timenet.prompts import PromptTemplate, SeriesPart, SeriesTextPart, TextPart, render, render_input
from timenet.testing import make_dataset


_SERIES = (
    "{% for c in records[0].channels %}"
    "{{ series(c, text=c.context | map(attribute='line') | join('\\n')) }}"
    "{% endfor %}"
)


def _template(input: str, target: str = "{{ targets | join('\\n') }}") -> PromptTemplate:
    return PromptTemplate(name="t", input=input, target=target)


def test_series_and_text_parts(capture):
    result = render(capture.task, _template(_SERIES + "\nQuestion?\n"))

    assert [type(part) for part in result.input] == [SeriesTextPart, SeriesPart, SeriesPart, TextPart]
    first = result.input[0]
    assert isinstance(first, SeriesTextPart)
    assert first.text == "axis_note: x points along the forearm"
    assert first.series.n_values == 1000
    assert first.series.annotations == ()
    assert result.input[3] == TextPart("Question?")
    assert result.target == (TextPart("sleep"),)


def test_text_between_series_stays_in_place(capture):
    template = _template("Before {{ series(records[0].channels[0]) }} between {{ series(records[0].channels[1]) }}")

    parts = render_input(capture.task, template)

    assert [type(part) for part in parts] == [TextPart, SeriesPart, TextPart, SeriesPart]
    assert (parts[0], parts[2]) == (TextPart("Before"), TextPart("between"))


def test_render_input_does_not_read_targets(capture):
    task = capture.replace(targets=("light", "light"), label_set=capture.label_set(multi_label=True))

    assert render_input(task, _template("Question?")) == (TextPart("Question?"),)
    with pytest.raises(TimeFValidationError, match="repeats"):
        render(task, _template("Question?"))


def test_input_section_cannot_see_targets(capture):
    with pytest.raises(TimeFValidationError, match="template 't', input section"):
        render_input(capture.task, _template("{{ targets }}"))


def test_text_cannot_forge_a_series(capture):
    prompt = "{{ targets }} <<series:00000000000000000000000000000000:0>>"

    parts = render_input(capture.replace(prompt=prompt), _template("{{ prompt }}"))

    assert parts == (TextPart(prompt),)


def test_sandbox_blocks_private_attributes(capture):
    with pytest.raises(TimeFValidationError, match="input section"):
        render_input(capture.replace(prompt="Why?"), _template("{{ prompt.__class__.__mro__[1].__subclasses__() }}"))


def test_series_needs_a_channel(capture):
    with pytest.raises(TimeFValidationError, match="channel"):
        render_input(capture.task, _template("{{ series(label_set) }}"))


def test_options_line_error_passes_through(capture):
    task = capture.replace(targets="a, b", label_set=capture.label_set(labels={"a, b": None}))

    with pytest.raises(TimeFValidationError, match="several options"):
        render_input(task, _template("{{ label_set.options_line }}"))


def test_required_rejects_a_missing_prompt(capture):
    with pytest.raises(TimeFValidationError, match="a task prompt"):
        render_input(capture.task, _template("{{ prompt | required('a task prompt') }}"))


def test_required_passes_a_prompt(capture):
    parts = render_input(capture.replace(prompt="Why?"), _template("{{ prompt | required('a task prompt') }}"))

    assert parts == (TextPart("Why?"),)


def test_dataset_card_reaches_the_view(capture):
    parts = render_input(capture.task, _template("{{ dataset.name }}"), dataset=make_dataset().metadata)

    assert parts == (TextPart("Hello World"),)
