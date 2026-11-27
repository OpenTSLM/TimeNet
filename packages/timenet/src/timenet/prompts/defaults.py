"""TimeNet's default prompt templates for classification tasks.

Each name ends with a version. A change to a template's text gets a new version, so that a training run can
log which text it used. The consumer chooses the template: none of them chooses between the task's own
prompt and the question from the label set.
"""

from timenet.prompts.render import PromptTemplate


# Aionic's layout: each Record's series with the text of their annotations, then the Record's other
# selected annotations, then the selected annotations outside the input Records.
_INPUTS = """\
{% for record in records %}
    {% for channel in record.channels %}
{{ series(channel, text=channel.context | map(attribute="line") | join("\\n")) }}
    {% endfor %}
    {% for item in record.context %}
{{ item.line }}
    {% endfor %}
{% endfor %}
{% for item in context %}
{{ item.line }}
{% endfor %}
"""

_ANSWER = (
    '{{ "Answer with every label that applies, one per line." if label_set.multi_label else "Answer with one label." }}'
)

# Aionic's reply: the rationale and an empty line, then one label per line.
_TARGET = """\
{% if rationale %}
{{ rationale }}

{% endif %}
{{ targets | join("\\n") }}
"""

LABEL_QUESTION = PromptTemplate(
    name="classification-label-question-v1",
    input=_INPUTS + "Which label matches the input? Options: {{ label_set.options_line }}. " + _ANSWER + "\n",
    target=_TARGET,
)
"""Aionic's question from the label set. It ignores the task's own prompt."""

PROMPT_QUESTION = PromptTemplate(
    name="classification-prompt-question-v1",
    input=_INPUTS + '{{ prompt | required("a task prompt") }}\n',
    target=_TARGET,
)
"""Aionic's use of the task's own prompt as the question. It fails for a task without a prompt."""

PROMPT_WITH_OPTIONS = PromptTemplate(
    name="classification-prompt-options-v1",
    input=_INPUTS
    + '{{ prompt | required("a task prompt") }}\nOptions: {{ label_set.options_line }}. '
    + _ANSWER
    + "\n",
    target=_TARGET,
)
"""The task's own prompt, then the options. It fails for a task without a prompt."""

DESCRIBED = PromptTemplate(
    name="classification-described-v1",
    input="""\
{% if window.duration_s %}
Input: {{ "%g" % window.duration_s }} s window.
{% endif %}
{% for record in records %}
    {% for source in record.sources %}
Source: {{ source.name }}
        {%- if source.metadata %} ({% for key, value in source.metadata.items() %}{{ key }}: {{ value }}
            {%- if not loop.last %}, {% endif %}{% endfor %}){% endif %}

        {% for channel in source.channels %}
Channel {{ channel.name }}: {{ channel.spec_name or channel.spec_type }}
            {%- if channel.unit_symbol %}, unit {{ channel.unit_symbol }}{% endif %}
            {%- if channel.sampling_rate_hz %}, {{ channel.sampling_rate_hz }} Hz{% endif %}.
{{ series(channel, text=channel.context | map(attribute="line") | join("\\n")) }}
        {% endfor %}
    {% endfor %}
    {% for item in record.context %}
{{ item.line }}
    {% endfor %}
{% endfor %}
{% for item in context %}
{{ item.line }}
{% endfor %}
Question: Which label matches the input?
Options:
{% for label in label_set.labels %}
- {{ label.name }}{% if label.description %}: {{ label.description }}{% endif %}

{% endfor %}
"""
    + _ANSWER
    + "\n",
    target=_TARGET,
)
"""The window, each Source and channel with its quantity, unit and rate, then the options with their
descriptions."""

DEFAULT_TEMPLATES = (LABEL_QUESTION, PROMPT_QUESTION, PROMPT_WITH_OPTIONS, DESCRIBED)
"""All default templates for classification tasks."""
