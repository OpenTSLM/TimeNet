import pytest

from timenet.errors import TimeFValidationError
from timenet.types import PromptTemplate


CHOICES = PromptTemplate(
    source=(
        "{% block prompt %}Classify the series.\n"
        "{% block answer_instruction %}Respond ONLY with the {{ key_noun }} of the correct choice "
        '({{ keys | join(", ") }}).\n\n{% endblock %}'
        "{% block options %}Choices:\n{% for option in options %}{{ option.marker }} {{ option.text }}\n"
        "{% endfor %}{% endblock %}{% endblock %}"
        "{% block answer %}{{ option.key }}{% endblock %}"
    )
)
UP = {"marker": "A:", "key": "A", "text": "up"}
DOWN = {"marker": "B:", "key": "B", "text": "down"}


def test_a_template_writes_the_prompt_and_the_answer_from_the_same_facts():
    facts = {"options": [UP, DOWN], "keys": ["A", "B"], "key_noun": "letter", "option": DOWN}

    assert CHOICES.render_prompt(facts) == (
        "Classify the series.\nRespond ONLY with the letter of the correct choice (A, B).\n\nChoices:\nA: up\nB: down\n"
    )
    assert CHOICES.render_answer(facts) == "B"
    assert CHOICES.variables == {"options", "keys", "key_noun", "option"}
    assert CHOICES.sections == ("answer_instruction", "options")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("{% block prompt %}{{ options }{% endblock %}", "is not valid Jinja"),
        ("{% block prompt %}Which?{% endblock %}", r"needs the blocks \['answer'\]"),
        ("Which? {% block prompt %}{% endblock %}{% block answer %}{% endblock %}", "nothing outside them"),
        (
            "{% block prompt %}{% endblock %}{% block answer %}{% endblock %}{% block other %}{% endblock %}",
            "nothing outside",
        ),
        ("{% block prompt %}{{ options.pop() }}{% endblock %}{% block answer %}{% endblock %}", r"\['Call'\]"),
        ("{% block prompt %}{% set key = 1 %}{% endblock %}{% block answer %}{% endblock %}", r"\['Assign'\]"),
        ('{% extends "other" %}{% block prompt %}{% endblock %}{% block answer %}{% endblock %}', r"\['Extends'\]"),
        ('{% block prompt %}{% include "other" %}{% endblock %}{% block answer %}{% endblock %}', r"\['Include'\]"),
        (
            "{% block prompt %}{{ options | random }}{% endblock %}{% block answer %}{% endblock %}",
            r"filters \['random'\]",
        ),
    ],
)
def test_a_template_rejects_a_source_that_is_not_in_the_allowed_part_of_jinja(source, message):
    with pytest.raises(ValueError, match=message):
        PromptTemplate(source=source)


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ({"options": [UP], "keys": ["A"], "option": UP}, "'key_noun' is undefined"),
        ({"options": [{"text": "up"}], "keys": [], "key_noun": "letter", "option": UP}, "has no attribute 'marker'"),
    ],
)
def test_a_fact_that_the_template_reads_but_does_not_get_raises_an_error(facts, message):
    with pytest.raises(TimeFValidationError, match=message):
        CHOICES.render_prompt(facts)


def test_a_template_cannot_read_a_private_attribute_of_a_fact():
    template = PromptTemplate(
        source="{% block prompt %}{{ options.__class__ }}{% endblock %}{% block answer %}{% endblock %}"
    )

    with pytest.raises(TimeFValidationError, match="unsafe"):
        template.render_prompt({"options": []})


def test_escaped_text_is_written_as_it_is():
    text = (
        'Output JSON:\n{\n    "direction": "[forward|reversed]"\n}\nKeep {{ this }}, {% that %} and {# those #}.\n'
        "A line that ends with a carriage return and a line feed.\r\nAnd one with a carriage return alone.\r"
    )
    template = PromptTemplate(
        source="{% block prompt %}"
        + PromptTemplate.escape_text(text)
        + "{% endblock %}{% block answer %}{% endblock %}"
    )

    assert template.render_prompt({}) == text
    assert template.variables == frozenset()
