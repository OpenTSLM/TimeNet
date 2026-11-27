"""Render TimeNet tasks into model-neutral prompts with Python renderers.

A renderer subclasses :class:`PromptRenderer` and builds the prompt of the input and of the expected reply. :func:`render_input` renders the input for inference,
and :func:`render` renders the input and the expected reply for training. Both return parts that a model turns
into its own format.
"""

from timenet.prompts.defaults import (
    DEFAULT_RENDERERS,
    DescribedRenderer,
    LabelQuestionRenderer,
    PromptQuestionRenderer,
    PromptWithOptionsRenderer,
)
from timenet.prompts.render import (
    PromptBuilder,
    PromptPart,
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
    "DescribedRenderer",
    "LabelQuestionRenderer",
    "PromptBuilder",
    "PromptPart",
    "PromptQuestionRenderer",
    "PromptRenderer",
    "PromptWithOptionsRenderer",
    "RenderedPrompt",
    "SeriesPart",
    "SeriesTextPart",
    "TextPart",
    "render",
    "render_input",
]
