"""Render TimeNet tasks into model-neutral prompts with Python renderers.

A renderer subclasses :class:`PromptRenderer` and adds the parts of the input and of the expected reply. :func:`render_input` renders the input for inference,
and :func:`render` renders the input and the expected reply for training. Both return parts that a model turns
into its own format.
"""

from timenet.prompts.defaults import (
    DEFAULT_RENDERERS,
    DESCRIBED,
    LABEL_QUESTION,
    PROMPT_QUESTION,
    PROMPT_WITH_OPTIONS,
)
from timenet.prompts.render import (
    Part,
    Parts,
    PromptRenderer,
    RenderedPrompt,
    SeriesPart,
    SeriesTextPart,
    TextPart,
    render,
    render_input,
)


__all__ = [
    "DEFAULT_RENDERERS",
    "DESCRIBED",
    "LABEL_QUESTION",
    "PROMPT_QUESTION",
    "PROMPT_WITH_OPTIONS",
    "Part",
    "Parts",
    "PromptRenderer",
    "RenderedPrompt",
    "SeriesPart",
    "SeriesTextPart",
    "TextPart",
    "render",
    "render_input",
]
