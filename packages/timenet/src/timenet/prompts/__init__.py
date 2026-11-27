"""Render TimeNet tasks into model-neutral prompts with Python renderers.

A renderer has an input function and a target function. :func:`render_input` renders the input for inference,
and :func:`render` renders the input and the expected reply for training. Both return parts that a model turns
into its own format.
"""

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
