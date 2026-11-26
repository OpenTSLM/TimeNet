import pytest

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.prompts import (
    DEFAULT_TEMPLATES,
    DESCRIBED,
    LABEL_QUESTION,
    PROMPT_QUESTION,
    PROMPT_WITH_OPTIONS,
    SeriesPart,
    SeriesTextPart,
    TextPart,
    render,
    render_input,
)
from timenet.testing import make_dataset
from timenet.types import ClassificationTask, TimeSeriesSpec


_QUESTION = "What is the subject doing?"


def _text(parts) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextPart))


def test_label_question_matches_aionic(capture):
    x, y, z, question = render_input(capture.task, LABEL_QUESTION)

    assert isinstance(x, SeriesTextPart)
    assert x.text == "axis_note: x points along the forearm"
    assert isinstance(y, SeriesPart)
    assert isinstance(z, SeriesPart)
    assert question == TextPart(
        "age_band: 38-52\n"
        "instructions: Use only the acceleration.\n"
        "Which label matches the input? Options: light, moderate-vigorous, sedentary, sleep. Answer with one label."
    )


def test_label_question_ignores_the_prompt(capture):
    assert _text(render_input(capture.replace(prompt=_QUESTION), LABEL_QUESTION)).endswith(
        "Options: light, moderate-vigorous, sedentary, sleep. Answer with one label."
    )


def test_multi_label_answer_sentence(capture):
    task = capture.replace(targets=("sleep",), label_set=capture.label_set(multi_label=True))

    assert _text(render_input(task, LABEL_QUESTION)).endswith("Answer with every label that applies, one per line.")


def test_prompt_question_uses_the_prompt(capture):
    text = _text(render_input(capture.replace(prompt=_QUESTION), PROMPT_QUESTION))

    assert text.endswith(f"instructions: Use only the acceleration.\n{_QUESTION}")


def test_prompt_question_needs_a_prompt(capture):
    with pytest.raises(TimeFValidationError, match="a task prompt"):
        render_input(capture.task, PROMPT_QUESTION)


def test_prompt_with_options(capture):
    text = _text(render_input(capture.replace(prompt=_QUESTION), PROMPT_WITH_OPTIONS))

    assert text.endswith(f"{_QUESTION}\nOptions: light, moderate-vigorous, sedentary, sleep. Answer with one label.")


def test_described(capture):
    parts = render_input(capture.task, DESCRIBED, dataset=make_dataset().metadata)
    text = _text(parts)

    assert sum(isinstance(part, SeriesPart | SeriesTextPart) for part in parts) == 3
    for expected in (
        "10 s window",
        "Axivity AX3",
        "placement: dominant wrist",
        "unit g",
        "100 Hz",
        "- sedentary: Waking behavior at 1.5 METs or less in a sitting, lying or reclining posture.",
        "Answer with one label.",
    ):
        assert expected in text
    for hidden in ("walmsley_2020", "12345", "P001", "sleeping"):
        assert hidden not in text


def test_described_leaves_out_empty_fields(capture):
    steps = TimeSeriesSpec(spec_type="count", name="Step count", unit_value=None, dtype="int32")
    signal = Signal(spec=steps, time_axis=OrdinalAxis(), name="steps", data=[1, 2, 3])
    task = ClassificationTask(
        inputs=[Record(sources=[Source("pedometer", signals=[signal])])],
        targets="sleep",
        label_set=capture.label_set(labels={"sleep": None, "awake": None}),
    )

    text = _text(render_input(task, DESCRIBED))

    for missing in ("unit", "Hz", "window", "(", "sleep:"):
        assert missing not in text
    assert "- sleep\n- awake" in text


@pytest.mark.parametrize("template", DEFAULT_TEMPLATES, ids=lambda template: template.name)
def test_target_with_rationale(capture, template):
    task = capture.replace(prompt=_QUESTION, rationale="Still for 10 s.")

    assert render(task, template).target == (TextPart("Still for 10 s.\n\nsleep"),)


def test_default_names_are_unique_and_versioned():
    names = [template.name for template in DEFAULT_TEMPLATES]

    assert len(set(names)) == len(names) == 4
    assert all(name.endswith("-v1") for name in names)
