"""A template that writes the prompt and the answer of a task.

A template is written in a small part of the Jinja language. It has two blocks. The ``prompt`` block writes
the text that a model gets, and the ``answer`` block writes one item of the answer that the model must give::

    {% block prompt %}Which trend?
    {% for option in options %}{{ option.marker }} {{ option.text }}
    {% endfor %}{% endblock %}
    {% block answer %}{{ option.key }}{% endblock %}

A template holds no data of a task. The task gives the template its facts, for example the options, when
the template is rendered. The same facts are given to both blocks, so the prompt and the answer agree.

The ``prompt`` block can hold more blocks, for example ``{% block options %}``. Such a block names one
section of the prompt.
"""

from collections.abc import Mapping
from functools import lru_cache
import re
from typing import NamedTuple, Self

import jinja2
from jinja2 import meta, nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.wire import NonEmptyString


PROMPT_BLOCK = "prompt"
"""The name of the block that writes the prompt."""
ANSWER_BLOCK = "answer"
"""The name of the block that writes one item of the answer."""

_ALLOWED_NODES = (
    nodes.Template,
    nodes.Block,
    nodes.Output,
    nodes.TemplateData,
    nodes.Name,
    nodes.Getattr,
    nodes.Getitem,
    nodes.Const,
    nodes.For,
    nodes.If,
    nodes.CondExpr,
    nodes.Filter,
    nodes.Keyword,
    nodes.Compare,
    nodes.Operand,
    nodes.Not,
    nodes.And,
    nodes.Or,
)
"""The parts of the Jinja language that a template can use: values, loops, conditions, blocks and filters.

A template is stored in a dataset, and a reader renders it. So a template cannot call a function, set a
variable, or load another template."""
_ALLOWED_FILTERS = frozenset({"capitalize", "first", "join", "last", "length", "lower", "map", "trim", "upper"})
"""The filters that a template can use. None of them is random, so the same facts give the same text."""
_NOT_PLAIN_TEXT = re.compile(r"\{[{%#]|\r")
"""What Jinja does not keep as it is in plain text: the start of an expression, statement, or comment, and a
carriage return, which Jinja reads as a line feed."""

# The sandbox refuses access to private attributes and changes of the facts. An undefined value raises an error.
_ENVIRONMENT = ImmutableSandboxedEnvironment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)


class _ParsedTemplate(NamedTuple):
    """What the source of a template gives when it is parsed."""

    variables: frozenset[str]
    sections: tuple[str, ...]


def _check_top_level(tree: nodes.Template) -> nodes.Block:
    """Make sure that a template holds one ``prompt`` block, one ``answer`` block, and no other text.

    Returns:
        The ``prompt`` block.

    Raises:
        ValueError: If a block is missing or repeated, or the template has text or a block outside them.
    """
    blocks: dict[str, nodes.Block] = {}
    for node in tree.body:
        if isinstance(node, nodes.Block) and node.name in {PROMPT_BLOCK, ANSWER_BLOCK} and node.name not in blocks:
            blocks[node.name] = node
            continue
        is_whitespace = isinstance(node, nodes.Output) and all(
            isinstance(part, nodes.TemplateData) and not part.data.strip() for part in node.nodes
        )
        if not is_whitespace:
            raise ValueError(
                f"a prompt template holds one {PROMPT_BLOCK!r} block and one {ANSWER_BLOCK!r} block, "
                "and nothing outside them"
            )
    missing = sorted({PROMPT_BLOCK, ANSWER_BLOCK} - blocks.keys())
    if missing:
        raise ValueError(f"a prompt template needs the blocks {missing}")
    return blocks[PROMPT_BLOCK]


@lru_cache(maxsize=4096)
def _parse(source: str) -> _ParsedTemplate:
    """Parse the source of a template and make sure that it uses only the allowed parts of Jinja.

    Many tasks of a dataset share one template, so the result is kept.

    Returns:
        The names of the facts that the template reads, and the names of its sections.

    Raises:
        ValueError: If the source is not valid Jinja, uses a part of Jinja that is not allowed, or does not
            have the two blocks.
    """
    try:
        tree = _ENVIRONMENT.parse(source)
    except jinja2.TemplateSyntaxError as error:
        raise ValueError(f"the prompt template is not valid Jinja: {error}") from error
    not_allowed = sorted(
        {type(node).__name__ for node in tree.find_all(nodes.Node) if type(node) not in _ALLOWED_NODES}
    )
    if not_allowed:
        raise ValueError(f"a prompt template cannot use these parts of Jinja: {not_allowed}")
    filters = sorted({node.name for node in tree.find_all(nodes.Filter)} - _ALLOWED_FILTERS)
    if filters:
        raise ValueError(f"a prompt template cannot use the filters {filters}. It can use {sorted(_ALLOWED_FILTERS)}")
    prompt_block = _check_top_level(tree)
    sections = tuple(block.name for block in prompt_block.find_all(nodes.Block))
    return _ParsedTemplate(frozenset(meta.find_undeclared_variables(tree)), sections)


@lru_cache(maxsize=4096)
def _compile(source: str) -> jinja2.Template:
    """Compile the source of a template for rendering, after the check of :func:`_parse`.

    A template is compiled when it is first rendered, not when it is built. A reader builds the template of
    each task, and many readers never render them.

    Returns:
        The compiled template.
    """
    _parse(source)
    return _ENVIRONMENT.from_string(source)


class PromptTemplate(TimeFModel):
    """A template that writes the prompt and the answer of a task from the facts of the task.

    See the module docstring for the form of a template. The source is checked when the template is built.
    """

    source: NonEmptyString
    """The template, in the allowed part of the Jinja language."""

    @model_validator(mode="after")
    def _is_valid_jinja(self) -> Self:
        _parse(self.source)
        return self

    @property
    def variables(self) -> frozenset[str]:
        """The names of the facts that the template reads, for example ``options``."""
        return _parse(self.source).variables

    @property
    def sections(self) -> tuple[str, ...]:
        """The names of the blocks inside the ``prompt`` block, in the order of the template."""
        return _parse(self.source).sections

    def render_prompt(self, facts: Mapping[str, object]) -> str:
        """Write the prompt from the facts of a task.

        Args:
            facts: The value of each fact that the template reads, by name.

        Returns:
            The prompt.

        Raises:
            TimeFValidationError: If the template reads a fact that is absent, or the text cannot be written.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _render_block)
        return self._render_block(PROMPT_BLOCK, facts)

    def render_answer(self, facts: Mapping[str, object]) -> str:
        """Write one item of the answer from the facts of a task.

        Args:
            facts: The value of each fact that the template reads, by name.

        Returns:
            The item of the answer.

        Raises:
            TimeFValidationError: If the template reads a fact that is absent, or the text cannot be written.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _render_block)
        return self._render_block(ANSWER_BLOCK, facts)

    def _render_block(self, block: str, facts: Mapping[str, object]) -> str:
        """Write the text of one of the two blocks.

        Returns:
            The text.

        Raises:
            TimeFValidationError: If the template reads a fact that is absent, or the text cannot be written.
        """
        template = _compile(self.source)
        try:
            return "".join(template.blocks[block](template.new_context(dict(facts))))
        except jinja2.TemplateError as error:
            raise TimeFValidationError(
                f"the {block!r} block of the prompt template cannot be written: {error}"
            ) from error

    @staticmethod
    def escape_text(text: str) -> str:
        """Return the Jinja source that writes a text as it is.

        Use this function to build the source of a template from text of a dataset. Jinja reads ``{{``,
        ``{%`` and ``{#`` as its own syntax, and it reads a carriage return as a line feed. The function
        writes them so that they stay as they are.

        Returns:
            The source.
        """
        return _NOT_PLAIN_TEXT.sub(lambda found: '{{ "\\r" }}' if found[0] == "\r" else f'{{{{ "{found[0]}" }}}}', text)
