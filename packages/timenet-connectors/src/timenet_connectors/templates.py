"""Templates that connectors share for classification tasks whose prompt prints no options.

A source can give the options of a question apart from its text, or give no question at all. A model
still needs the options. These templates print them, so that each consumer shows such a task in the same
way, and so that the options can be shown in another order.

The options of these templates have no keys. The answer is the text of the correct option, as the source
gives it.
"""

from collections.abc import Sequence

from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTemplate, LabelSet, PromptTemplate, TemplateOption


_ANSWER_BLOCK = "{% block answer %}{{ option.text }}{% endblock %}"

DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
    source=(
        "{% block prompt %}"
        "{% block question %}Which label matches the input?{% endblock %} "
        '{% block options %}Options: {{ options | map(attribute="text") | join(", ") }}.{% endblock %} '
        "{% block answer_instruction %}"
        "{% if multi_label %}Answer with every label that applies, one per line."
        "{% else %}Answer with one label.{% endif %}"
        "{% endblock %}"
        "{% endblock %}" + _ANSWER_BLOCK
    )
)
"""The template of a task that has no prompt. It asks for the label and lists the labels on one line."""

PROMPT_THEN_OPTIONS_TEMPLATE = PromptTemplate(
    source=(
        "{% block prompt %}"
        "{% block question %}{{ prompt }}{% endblock %}\n\n"
        "{% block options %}Options:\n{% for option in options %}- {{ option.text }}\n{% endfor %}{% endblock %}\n"
        "{% block answer_instruction %}"
        "{% if multi_label %}Answer with every option that applies, one per line."
        "{% else %}Answer with one option.{% endif %}"
        "{% endblock %}"
        "{% endblock %}" + _ANSWER_BLOCK
    )
)
"""The template of a task whose prompt does not print its options. It lists them after the stored prompt."""


def get_default_template_of_label_set(label_set: LabelSet) -> ClassificationTemplate:
    """Return the template of a task that has no prompt, with the labels of a label set as its options.

    The template lists the labels on one line, separated by commas::

        Which label matches the input? Options: biking, lying, running. Answer with one label.

    Build the template once for a label set and give the same object to each task.

    Args:
        label_set: The label set of the tasks.

    Returns:
        The template, with one option per label in the order of the label set.

    Raises:
        TimeNetConnectorError: If the name of a label holds a comma or a line break. The list of the labels
            would then read as more labels.
    """
    not_listable = [name for name in label_set.names if "," in name or "\n" in name]
    if not_listable:
        raise TimeNetConnectorError(
            f"the labels {not_listable} hold a comma or a line break, so the default template cannot list them"
        )
    return ClassificationTemplate(
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        options=[TemplateOption(label=name) for name in label_set.names],
    )


def get_template_that_lists_options_after_prompt(options: Sequence[str]) -> ClassificationTemplate:
    """Return the template of a task whose stored prompt does not print its options.

    The template writes the stored prompt, and then one option per line::

        Is there an anomaly in the series?

        Options:
        - Yes
        - No

        Answer with one option.

    The template adds text to the stored prompt, so it does not reproduce the published prompt.

    Args:
        options: The labels of the options, in the order of the source.

    Returns:
        The template.

    Raises:
        TimeNetConnectorError: If an option holds a line break. It would then read as more options.
    """
    not_listable = [option for option in options if "\n" in option]
    if not_listable:
        raise TimeNetConnectorError(f"the options {not_listable} hold a line break, so the template cannot list them")
    return ClassificationTemplate(
        prompt_template=PROMPT_THEN_OPTIONS_TEMPLATE,
        options=[TemplateOption(label=option) for option in options],
    )
