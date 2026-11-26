"""Render TimeNet tasks into model-neutral prompts with Jinja templates.

A template has an input section and a target section. :func:`render_input` renders the input for inference,
and :func:`render` renders the input and the expected reply for training. Both return parts that a model turns
into its own format.
"""

from timenet.prompts.defaults import (
    DEFAULT_TEMPLATES,
    DESCRIBED,
    LABEL_QUESTION,
    PROMPT_QUESTION,
    PROMPT_WITH_OPTIONS,
)
from timenet.prompts.render import (
    Part,
    PromptTemplate,
    RenderedPrompt,
    SeriesPart,
    SeriesTextPart,
    TextPart,
    render,
    render_input,
)


__all__ = [
    "DEFAULT_TEMPLATES",
    "DESCRIBED",
    "LABEL_QUESTION",
    "PROMPT_QUESTION",
    "PROMPT_WITH_OPTIONS",
    "Part",
    "PromptTemplate",
    "RenderedPrompt",
    "SeriesPart",
    "SeriesTextPart",
    "TextPart",
    "render",
    "render_input",
]
