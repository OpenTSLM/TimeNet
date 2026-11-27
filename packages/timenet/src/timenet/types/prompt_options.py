"""How the prompt of a classification task shows the options of the task.

A prompt can print its options, for example ``A: up`` and ``B: down``. :class:`PromptOptions` says where
each option is in the prompt and which key the prompt gives it. With this information, a consumer can show
the options in another order or with other keys, and the answer stays correct.
"""

import itertools
from typing import Self

from pydantic import StrictBool, model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.option_keys import ALL_CLASSIFICATION_ENUMERATORS, ClassificationEnumerator, OptionKeyStyle
from timenet.types.wire import NonEmptyString, StrictNonNegativeInt, StrictPositiveInt


class PromptSpan(TimeFModel):
    """A part of a prompt: the characters from ``start`` up to, but not including, ``end``.

    The positions count the characters of the prompt text. The first character has position 0.
    """

    start: StrictNonNegativeInt
    """The position of the first character of the part."""
    end: StrictPositiveInt
    """The position after the last character of the part."""

    @model_validator(mode="after")
    def _holds_a_character(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"a prompt span must hold at least one character, got {self.start} to {self.end}")
        return self

    def read_from(self, prompt: str) -> str:
        """Return the part of the prompt that the span holds."""
        return prompt[self.start : self.end]


class PromptOption(TimeFModel):
    """One label of the label set of a task, as the prompt of the task shows it."""

    label: NonEmptyString
    """The name of the label that the option stands for."""
    marker: PromptSpan | None = None
    """Where the marker of the option is in the prompt, for example ``A:``. ``None`` if the option has no key."""
    text: PromptSpan | None = None
    """Where the text of the option is in the prompt. ``None`` if the prompt does not print the option."""
    pinned: StrictBool = False
    """Whether the option must keep its place, for example ``None of the above``."""


class OptionKeyReference(TimeFModel):
    """A place in a prompt that refers to the key of an option, outside the list of options.

    For example, ``Answer with A, B or C.`` holds three references.
    """

    option_position: StrictNonNegativeInt
    """The position of the option that the reference names. The first option has position 0."""
    span: PromptSpan
    """Where the reference is in the prompt."""


class PromptOptions(TimeFModel):
    """How the prompt of a classification task shows the options of the task.

    The options are in the order of the prompt. They have one of three forms:

    - The prompt prints each option with a key. Each option has a ``marker`` and a ``text``, and the
      ``key_style`` is set.
    - The prompt prints each option without a key. Each option has only a ``text``.
    - The prompt does not print the options. No option has a ``marker`` or a ``text``. A consumer that shows
      the task must print the options.

    The prompt ``Which trend? A: up B: down. Answer with A or B.`` has these options::

        PromptOptions(
            options=[
                PromptOption(label="A", marker=PromptSpan(start=13, end=15), text=PromptSpan(start=16, end=18)),
                PromptOption(label="B", marker=PromptSpan(start=19, end=21), text=PromptSpan(start=22, end=26)),
            ],
            key_style=OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":"),
            key_references=[
                OptionKeyReference(option_position=0, span=PromptSpan(start=40, end=41)),
                OptionKeyReference(option_position=1, span=PromptSpan(start=45, end=46)),
            ],
        )
    """

    options: tuple[PromptOption, ...]
    """The options in the order of the prompt."""
    key_style: OptionKeyStyle | None = None
    """How the prompt writes the keys of the options. ``None`` if the options have no keys."""
    key_references: tuple[OptionKeyReference, ...] = ()
    """The places in the prompt that refer to a key, outside the list of options."""
    allowable_enumerators: frozenset[ClassificationEnumerator] = ALL_CLASSIFICATION_ENUMERATORS
    """The enumerators that can replace the enumerator of the ``key_style``.

    A prompt can restrict them. For example, a prompt that says ``Answer with the letter`` allows only the
    two Latin enumerators."""

    @model_validator(mode="after")
    def _is_consistent(self) -> Self:
        if not self.options:
            raise ValueError("prompt options must hold at least one option")
        labels = self.labels
        if len(set(labels)) != len(labels):
            raise ValueError(f"prompt options name a label more than once: {list(labels)}")
        self._check_forms_of_options()
        self._check_keys()
        self._check_order_of_spans()
        return self

    def _check_forms_of_options(self) -> None:
        """Make sure that all options have the same form.

        Raises:
            ValueError: If only some options have a text or a marker, or an option has a marker but no text.
        """
        with_text = [option.text is not None for option in self.options]
        with_marker = [option.marker is not None for option in self.options]
        if any(with_text) != all(with_text):
            raise ValueError("either each prompt option has a text span, or none has one")
        if any(with_marker) != all(with_marker):
            raise ValueError("either each prompt option has a marker span, or none has one")
        if any(with_marker) and not any(with_text):
            raise ValueError("a prompt option with a marker span must have a text span")

    def _check_keys(self) -> None:
        """Make sure that the key style, the markers, the references and the enumerators agree.

        Raises:
            ValueError: If the options have markers without a key style or a key style without markers,
                a reference has no key style or names no option, or the enumerator of the key style is
                not allowable.
        """
        has_markers = self.options[0].marker is not None
        if has_markers != (self.key_style is not None):
            raise ValueError("prompt options have a key style if, and only if, each option has a marker span")
        if self.key_style is None:
            if self.key_references:
                raise ValueError("prompt options without a key style cannot have key references")
            return
        for reference in self.key_references:
            if reference.option_position >= len(self.options):
                raise ValueError(
                    f"a key reference names the option position {reference.option_position}, which is absent"
                )
        if self.key_style.enumerator not in self.allowable_enumerators:
            raise ValueError(
                f"the enumerator {self.key_style.enumerator.value!r} of the key style is not one of the "
                f"allowable enumerators {sorted(self.allowable_enumerators)}"
            )

    def _check_order_of_spans(self) -> None:
        """Make sure that the spans follow the order of the options and do not overlap.

        Raises:
            ValueError: If a span of an option starts before the end of the span before it, or two spans
                overlap.
        """
        option_spans = [span for option in self.options for span in (option.marker, option.text) if span is not None]
        for before, after in itertools.pairwise(option_spans):
            if after.start < before.end:
                raise ValueError(
                    "the spans of the prompt options must follow the order of the options, and the marker of an "
                    f"option must stand before its text. The span {after.start} to {after.end} does not"
                )
        spans = sorted([*option_spans, *(reference.span for reference in self.key_references)], key=lambda s: s.start)
        for before, after in itertools.pairwise(spans):
            if after.start < before.end:
                raise ValueError(
                    f"the prompt spans {before.start} to {before.end} and {after.start} to {after.end} overlap"
                )

    @property
    def labels(self) -> tuple[str, ...]:
        """The label of each option, in the order of the options."""
        return tuple(option.label for option in self.options)

    @property
    def spans(self) -> tuple[PromptSpan, ...]:
        """Every span of the options and of the key references, in no fixed order."""
        option_spans = [span for option in self.options for span in (option.marker, option.text) if span is not None]
        return (*option_spans, *(reference.span for reference in self.key_references))

    def check_spans_in_prompt(self, prompt: str | None) -> None:
        """Make sure that the spans fit the prompt that they describe.

        Args:
            prompt: The text of the prompt, or ``None`` if the task has no prompt.

        Raises:
            TimeFValidationError: If the options have spans but the task has no prompt, a span ends after
                the prompt, or a marker or a reference in the prompt is not written in the key style.
        """
        if not self.spans:
            return
        if prompt is None:
            raise TimeFValidationError("the prompt options have spans, but the task has no prompt")
        for span in self.spans:
            if span.end > len(prompt):
                raise TimeFValidationError(
                    f"the prompt span {span.start} to {span.end} ends after the prompt, which has {len(prompt)} characters"
                )
        if self.key_style is None:
            return
        for position, option in enumerate(self.options):
            marker = self.key_style.get_marker_of_option_at(position)
            if option.marker is not None and option.marker.read_from(prompt) != marker:
                raise TimeFValidationError(
                    f"the prompt has {option.marker.read_from(prompt)!r} at the marker of the option at position "
                    f"{position}, but the key style gives {marker!r}"
                )
        for reference in self.key_references:
            expected = self.key_style.get_reference_to_option_at(reference.option_position)
            if reference.span.read_from(prompt) != expected:
                raise TimeFValidationError(
                    f"the prompt has {reference.span.read_from(prompt)!r} at a key reference to the option at "
                    f"position {reference.option_position}, but the key style gives {expected!r}"
                )
