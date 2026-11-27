"""One way to show the options of a classification task, with the answer for it.

A classification task stores its prompt and its answer as the source publishes them. An
:class:`OptionPresentation` holds the prompt, the options and the answer together, so that the options can
be shown in another order or with other keys while the answer stays correct.
"""

from collections.abc import Sequence
import random
from typing import NamedTuple, Self

from pydantic import StrictStr, model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.option_keys import OptionKeyStyle
from timenet.types.prompt_options import OptionKeyReference, PromptOption, PromptOptions, PromptSpan
from timenet.types.wire import NonEmptyString


class _Replacement(NamedTuple):
    """New text for one part of a prompt."""

    span: PromptSpan
    """The part of the prompt that is replaced."""
    text: str
    """The text that takes its place."""


def _replace_spans(prompt: str, replacements: Sequence[_Replacement]) -> tuple[str, list[PromptSpan]]:
    """Write a prompt with new text in some of its parts.

    The replaced parts must not overlap.

    Returns:
        The new prompt, and the span of each new text in it, in the order of ``replacements``.
    """
    pieces: list[str] = []
    new_spans: dict[int, PromptSpan] = {}
    end_of_previous, length = 0, 0
    in_prompt_order = sorted(range(len(replacements)), key=lambda index: replacements[index].span.start)
    for index in in_prompt_order:
        span, text = replacements[index]
        unchanged = prompt[end_of_previous : span.start]
        pieces.extend((unchanged, text))
        length += len(unchanged)
        new_spans[index] = PromptSpan(start=length, end=length + len(text))
        length += len(text)
        end_of_previous = span.end
    pieces.append(prompt[end_of_previous:])
    return "".join(pieces), [new_spans[index] for index in range(len(replacements))]


class OptionPresentation(TimeFModel):
    """One way to show the options of a classification task, with the answer for it.

    :meth:`~timenet.types.ClassificationTask.get_option_presentation` builds the presentation of a task.
    The methods that change the order or the keys return a new presentation. They change the prompt and the
    answer together.

    The presentation holds no reference to its task, so it can be stored and changed without the task.
    """

    prompt: StrictStr | None
    """The prompt as this presentation shows it, or ``None`` if the task has no prompt."""
    prompt_options: PromptOptions
    """The options in the order of this presentation, with their places in ``prompt``."""
    target_labels: tuple[NonEmptyString, ...]
    """The labels of the correct options. A change of the order or of the keys does not change them."""

    @model_validator(mode="after")
    def _is_consistent(self) -> Self:
        self.prompt_options.check_spans_in_prompt(self.prompt)
        if self.prompt_options.key_style is not None:
            not_shown = sorted(set(self.target_labels) - set(self.prompt_options.labels))
            if not_shown:
                raise ValueError(f"the target labels {not_shown} have no option, so they have no key")
        return self

    def get_answer(self) -> tuple[str, ...]:
        """Return the answer as this presentation asks for it.

        Returns:
            One item per target label, in the order of ``target_labels``. If the options have keys, an
            item is the reference to the key of the correct option, for example ``B``. If the options
            have no keys, an item is the target label.
        """
        style = self.prompt_options.key_style
        if style is None:
            return self.target_labels
        positions = {label: position for position, label in enumerate(self.prompt_options.labels)}
        return tuple(style.get_reference_to_option_at(positions[label]) for label in self.target_labels)

    def get_label_of_answer(self, answer: str) -> str:
        """Return the label of the option that an answer names in this presentation.

        Use this method to compare the answer of a model with ``target_labels``.

        Args:
            answer: One answer, written as :meth:`get_answer` writes an item.

        Returns:
            The label of the option.

        Raises:
            TimeFValidationError: If the answer names no option of this presentation.
        """
        labels = self.prompt_options.labels
        style = self.prompt_options.key_style
        if style is None:
            if answer not in labels:
                raise TimeFValidationError(f"the answer {answer!r} is not one of the options {list(labels)}")
            return answer
        position = style.get_position_of_option_with_reference(answer)
        if position >= len(labels):
            raise TimeFValidationError(f"the answer {answer!r} names no option. There are {len(labels)} options")
        return labels[position]

    def with_options_in_order(self, order: Sequence[int]) -> Self:
        """Return a presentation that shows the options in another order.

        The keys keep their places and the texts of the options move. For example, the order ``[1, 0]``
        turns ``A: up B: down`` into ``A: down B: up``.

        Args:
            order: For each place, the current position of the option that moves to it.

        Returns:
            The new presentation.

        Raises:
            TimeFValidationError: If ``order`` does not hold each position once, or a pinned option moves.
        """
        options = self.prompt_options.options
        if sorted(order) != list(range(len(options))):
            raise TimeFValidationError(f"the order {list(order)} must hold each of the {len(options)} positions once")
        moved = [position for position, source in enumerate(order) if position != source and options[source].pinned]
        if moved:
            raise TimeFValidationError(f"the order {list(order)} moves the pinned options to the positions {moved}")
        return self._with_order_and_key_style(order, self.prompt_options.key_style)

    def with_key_style(self, key_style: OptionKeyStyle) -> Self:
        """Return a presentation that writes the keys in another style.

        The markers of the options and the key references change. The order of the options stays.

        Args:
            key_style: The style of the keys in the new presentation.

        Returns:
            The new presentation.

        Raises:
            TimeFValidationError: If the options have no keys, the enumerator of the style is not
                allowable, or the enumerator has too few symbols for the options.
        """
        options = self.prompt_options
        if options.key_style is None:
            raise TimeFValidationError("the options have no keys, so their key style cannot change")
        if key_style.enumerator not in options.allowable_enumerators:
            raise TimeFValidationError(
                f"the enumerator {key_style.enumerator.value!r} is not one of the allowable enumerators "
                f"{sorted(options.allowable_enumerators)}"
            )
        if not key_style.enumerator.has_symbols_for_option_count(len(options.options)):
            raise TimeFValidationError(
                f"the enumerator {key_style.enumerator.value!r} has too few symbols for {len(options.options)} options"
            )
        return self._with_order_and_key_style(range(len(options.options)), key_style)

    def with_shuffled_label_keys(self, rng: random.Random) -> Self:
        """Return a presentation with the options in a random order and, if they have keys, in a random key style.

        A pinned option keeps its place. The key style is one of the variants that
        :meth:`~timenet.types.OptionKeyStyle.get_variants_with_common_markers` gives for the allowable
        enumerators that have enough symbols.

        Args:
            rng: The generator of the random choices. The same generator state gives the same presentation.

        Returns:
            The new presentation.
        """
        options = self.prompt_options
        free_positions = [position for position, option in enumerate(options.options) if not option.pinned]
        sources = list(free_positions)
        # Fisher-Yates shuffle. The steps are written out so that the result depends only on ``randrange``.
        for index in range(len(sources) - 1, 0, -1):
            other = rng.randrange(index + 1)
            sources[index], sources[other] = sources[other], sources[index]
        order = list(range(len(options.options)))
        for position, source in zip(free_positions, sources, strict=True):
            order[position] = source
        key_style = options.key_style
        if key_style is not None:
            enumerators = frozenset(
                enumerator
                for enumerator in options.allowable_enumerators
                if enumerator.has_symbols_for_option_count(len(options.options))
            )
            variants = key_style.get_variants_with_common_markers(enumerators)
            key_style = variants[rng.randrange(len(variants))]
        return self._with_order_and_key_style(order, key_style)

    def _with_order_and_key_style(self, order: Sequence[int], key_style: OptionKeyStyle | None) -> Self:
        """Build the presentation for an order of the options and a key style.

        Args:
            order: For each place, the current position of the option that moves to it.
            key_style: The style of the keys, or ``None`` if the options have no keys.

        Returns:
            The new presentation.
        """
        old = self.prompt_options
        shown = [old.options[source] for source in order]
        if self.prompt is None or not old.spans:
            options = tuple(PromptOption(label=option.label, pinned=option.pinned) for option in shown)
            return type(self)(
                prompt=self.prompt,
                prompt_options=old.model_copy(update={"options": options}),
                target_labels=self.target_labels,
            )
        prompt = self.prompt
        marker_replacements: list[_Replacement] = []
        text_replacements: list[_Replacement] = []
        for position, place in enumerate(old.options):
            moved_text = shown[position].text
            if place.text is None or moved_text is None:
                continue
            text_replacements.append(_Replacement(place.text, moved_text.read_from(prompt)))
            if place.marker is not None and key_style is not None:
                marker_replacements.append(_Replacement(place.marker, key_style.get_marker_of_option_at(position)))
        reference_replacements = [
            _Replacement(reference.span, key_style.get_reference_to_option_at(reference.option_position))
            for reference in old.key_references
            if key_style is not None
        ]
        new_prompt, new_spans = _replace_spans(
            prompt, [*text_replacements, *marker_replacements, *reference_replacements]
        )
        text_spans = new_spans[: len(text_replacements)]
        marker_spans = new_spans[len(text_replacements) : len(text_replacements) + len(marker_replacements)]
        reference_spans = new_spans[len(text_replacements) + len(marker_replacements) :]
        options = tuple(
            PromptOption(
                label=option.label,
                marker=marker_spans[position] if marker_spans else None,
                text=text_spans[position],
                pinned=option.pinned,
            )
            for position, option in enumerate(shown)
        )
        references = tuple(
            OptionKeyReference(option_position=reference.option_position, span=span)
            for reference, span in zip(old.key_references, reference_spans, strict=True)
        )
        return type(self)(
            prompt=new_prompt,
            prompt_options=PromptOptions(
                options=options,
                key_style=key_style,
                key_references=references,
                allowable_enumerators=old.allowable_enumerators,
            ),
            target_labels=self.target_labels,
        )
