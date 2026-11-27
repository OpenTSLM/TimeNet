"""How a classification task is shown to a model: a template, the options, and their keys.

A :class:`ClassificationTemplate` holds a :class:`~timenet.types.PromptTemplate` and the facts that the
template needs. It writes the prompt and the answer of a task with the options in the order of the source,
or in a random order with keys in a random style. The answer follows the options in both cases.

A template gets these facts:

- ``options``: The options in the order in which they are shown. Each option has a ``label`` and a ``text``.
  If the options have keys, an option also has a ``key``, for example ``B``, and a ``marker``, for example
  ``B:``.
- ``keys``: The key of each option, in the same order. Only if the options have keys.
- ``key_noun``: The word for a key, for example ``letter``. Only if the options have keys.
- ``multi_label``: Whether more than one option can be correct.
- ``prompt``: The stored prompt of the task. Only if the task has a prompt.
- ``option``: Only in the ``answer`` block: the correct option that the block writes.
"""

from collections.abc import Collection, Sequence
import random
from typing import Self

from pydantic import StrictBool, StrictStr, model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.option_keys import ALL_CLASSIFICATION_ENUMERATORS, ClassificationEnumerator, OptionKeyStyle
from timenet.types.prompt_templates import PromptTemplate
from timenet.types.wire import NonEmptyString


class TemplateOption(TimeFModel):
    """One label of the label set of a task, as an option of its template."""

    label: NonEmptyString
    """The name of the label that the option stands for."""
    text: NonEmptyString | None = None
    """The text that the prompt shows for the option. ``None`` means the name of the label."""
    pinned: StrictBool = False
    """Whether the option must keep its place, for example ``None of the above``."""


class OptionPresentation(TimeFModel):
    """The prompt and the answer of a classification task for one order of its options and one key style."""

    prompt: StrictStr
    """The prompt."""
    answers: tuple[StrictStr, ...]
    """The answer: one item for each correct option, in the order in which the options are shown."""
    option_labels: tuple[NonEmptyString, ...]
    """The label of each option, in the order in which the options are shown."""
    option_answers: tuple[StrictStr, ...]
    """The item of the answer that names each option, in the same order."""
    key_style: OptionKeyStyle | None
    """How the keys of the options are written. ``None`` if the options have no keys."""

    def get_label_of_answer(self, answer: str) -> str:
        """Return the label of the option that one item of an answer names.

        Use this method to compare the answer of a model with the target labels of the task.

        Args:
            answer: One item of an answer, written as an item of ``answers``.

        Returns:
            The label of the option.

        Raises:
            TimeFValidationError: If no option, or more than one option, has this answer.
        """
        labels = [label for label, item in zip(self.option_labels, self.option_answers, strict=True) if item == answer]
        if len(labels) != 1:
            raise TimeFValidationError(
                f"the answer {answer!r} names {len(labels)} options. The answers of the options are "
                f"{list(self.option_answers)}"
            )
        return labels[0]


class ClassificationTemplate(TimeFModel):
    """A template of a classification task, with the options and the key style that the template shows.

    The options are in the order of the source, and the key style is the one of the source. The source
    ``Which trend? A: up B: down`` with the answer ``B`` has this template::

        ClassificationTemplate(
            prompt_template=PromptTemplate(
                source=(
                    "{% block prompt %}Which trend? {{ options[0].marker }} {{ options[0].text }} "
                    "{{ options[1].marker }} {{ options[1].text }}{% endblock %}"
                    "{% block answer %}{{ option.key }}{% endblock %}"
                )
            ),
            options=[TemplateOption(label="A", text="up"), TemplateOption(label="B", text="down")],
            key_style=OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":"),
            reproduces_published_prompt=True,
        )
    """

    prompt_template: PromptTemplate
    """The template that writes the prompt and the answer."""
    options: tuple[TemplateOption, ...]
    """The options in the order of the source."""
    key_style: OptionKeyStyle | None = None
    """How the source writes the keys of the options. ``None`` if the options have no keys."""
    allowable_enumerators: frozenset[ClassificationEnumerator] = ALL_CLASSIFICATION_ENUMERATORS
    """The enumerators that can replace the enumerator of the ``key_style``.

    Restrict them if the template fits only some of them. For example, a template with the text
    ``Answer with the letter`` allows only the two Latin enumerators."""
    reproduces_published_prompt: StrictBool = False
    """Whether the template, with the options and keys of the source, writes the stored prompt and targets.

    If ``True``, the dataset checks this when it adds the task. Set ``False`` for a template that adds text to
    the stored prompt, or for a task without a prompt."""

    @model_validator(mode="after")
    def _is_consistent(self) -> Self:
        if not self.options:
            raise ValueError("a classification template must hold at least one option")
        labels = [option.label for option in self.options]
        if len(set(labels)) != len(labels):
            raise ValueError(f"a classification template names a label more than once: {labels}")
        if self.key_style is not None and self.key_style.enumerator not in self.allowable_enumerators:
            raise ValueError(
                f"the enumerator {self.key_style.enumerator.value!r} of the key style is not one of the "
                f"allowable enumerators {sorted(self.allowable_enumerators)}"
            )
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        """The label of each option, in the order of the source."""
        return tuple(option.label for option in self.options)

    def get_presentation_in_published_form(
        self, target_labels: Collection[str], *, multi_label: bool, prompt: str | None
    ) -> OptionPresentation:
        """Write the prompt and the answer with the options in the order of the source and with its keys.

        Args:
            target_labels: The labels of the correct options.
            multi_label: Whether more than one option can be correct.
            prompt: The stored prompt of the task, or ``None`` if the task has no prompt.

        Returns:
            The presentation.

        Raises:
            TimeFValidationError: If a target label has no option, or the template cannot be written.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _present)
        return self._present(range(len(self.options)), self.key_style, target_labels, multi_label, prompt)

    def get_presentation_with_shuffled_label_keys(
        self, target_labels: Collection[str], *, multi_label: bool, prompt: str | None, rng: random.Random
    ) -> OptionPresentation:
        """Write the prompt and the answer with the options in a random order and, if they have keys, in a random key style.

        A pinned option keeps its place. The key style is one of the variants that
        :meth:`~timenet.types.OptionKeyStyle.get_variants_with_common_markers` gives for the allowable
        enumerators that have enough symbols.

        Args:
            target_labels: The labels of the correct options.
            multi_label: Whether more than one option can be correct.
            prompt: The stored prompt of the task, or ``None`` if the task has no prompt.
            rng: The generator of the random choices. The same generator state gives the same presentation.

        Returns:
            The presentation.

        Raises:
            TimeFValidationError: If a target label has no option, or the template cannot be written.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by _present)
        free_positions = [position for position, option in enumerate(self.options) if not option.pinned]
        sources = list(free_positions)
        # Fisher-Yates shuffle. The steps are written out so that the result depends only on ``randrange``.
        for index in range(len(sources) - 1, 0, -1):
            other = rng.randrange(index + 1)
            sources[index], sources[other] = sources[other], sources[index]
        order = list(range(len(self.options)))
        for position, source in zip(free_positions, sources, strict=True):
            order[position] = source
        key_style = self.key_style
        if key_style is not None:
            enumerators = frozenset(
                enumerator
                for enumerator in self.allowable_enumerators
                if enumerator.has_symbols_for_option_count(len(self.options))
            )
            variants = key_style.get_variants_with_common_markers(enumerators)
            key_style = variants[rng.randrange(len(variants))]
        return self._present(order, key_style, target_labels, multi_label, prompt)

    def _present(
        self,
        order: Sequence[int],
        key_style: OptionKeyStyle | None,
        target_labels: Collection[str],
        multi_label: bool,
        prompt: str | None,
    ) -> OptionPresentation:
        """Write the prompt and the answer for an order of the options and a key style.

        Args:
            order: For each place, the position in ``options`` of the option that is shown there.
            key_style: The style of the keys, or ``None`` if the options have no keys.
            target_labels: The labels of the correct options.
            multi_label: Whether more than one option can be correct.
            prompt: The stored prompt of the task, or ``None`` if the task has no prompt.

        Returns:
            The presentation.

        Raises:
            TimeFValidationError: If a target label has no option.
        """
        without_option = sorted(set(target_labels) - set(self.labels))
        if without_option:
            raise TimeFValidationError(f"the target labels {without_option} are not options of the template")
        shown: list[dict[str, str]] = []
        for place, source in enumerate(order):
            option = self.options[source]
            facts_of_option = {"label": option.label, "text": option.label if option.text is None else option.text}
            if key_style is not None:
                facts_of_option["key"] = key_style.get_reference_to_option_at(place)
                facts_of_option["marker"] = key_style.get_marker_of_option_at(place)
            shown.append(facts_of_option)
        facts: dict[str, object] = {"options": shown, "multi_label": multi_label}
        if key_style is not None:
            facts["keys"] = [option["key"] for option in shown]
            facts["key_noun"] = key_style.enumerator.noun
        if prompt is not None:
            facts["prompt"] = prompt
        option_answers = tuple(self.prompt_template.render_answer({**facts, "option": option}) for option in shown)
        return OptionPresentation(
            prompt=self.prompt_template.render_prompt(facts),
            answers=tuple(
                answer for option, answer in zip(shown, option_answers, strict=True) if option["label"] in target_labels
            ),
            option_labels=tuple(option["label"] for option in shown),
            option_answers=option_answers,
            key_style=key_style,
        )
