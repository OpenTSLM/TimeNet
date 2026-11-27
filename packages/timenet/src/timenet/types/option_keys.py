"""The keys of the options in a classification prompt.

A prompt can count its options with keys, for example ``A``, ``B`` and ``C``. A key is one symbol of a
:class:`ClassificationEnumerator`. An :class:`OptionKeyStyle` says how a prompt writes the keys.
"""

from enum import StrEnum, unique
from string import ascii_lowercase, ascii_uppercase
from typing import Annotated

from pydantic import AfterValidator, StrictStr
from roman_numerals import MAX, InvalidRomanNumeralError, OutOfRangeError, RomanNumeral

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel


_LARGEST_ROMAN_NUMBER = MAX
"""The largest number that the standard Roman numerals can write."""
_COMMON_MARKER_AFFIXES = (("", ":"), ("", ")"), ("", "."), ("(", ")"))
"""The common prefix and suffix pairs of a marker: ``A:``, ``A)``, ``A.`` and ``(A)``."""


@unique
class ClassificationEnumerator(StrEnum):
    """A series of symbols that counts the options of a classification prompt.

    The first option has position 0.
    """

    LATIN_UPPERCASE = "latin_uppercase"
    """``A``, ``B``, ``C``. The series has 26 symbols."""
    LATIN_LOWERCASE = "latin_lowercase"
    """``a``, ``b``, ``c``. The series has 26 symbols."""
    NUMERIC = "numeric"
    """``1``, ``2``, ``3``. The series has no end."""
    ROMAN_NUMERIC_LOWERCASE = "roman_numeric_lowercase"
    """``i``, ``ii``, ``iii``. The series has 3999 symbols."""
    ROMAN_NUMERIC_UPPERCASE = "roman_numeric_uppercase"
    """``I``, ``II``, ``III``. The series has 3999 symbols."""

    def has_symbols_for_option_count(self, option_count: int) -> bool:
        """Return whether the series has a symbol for each of ``option_count`` options."""
        if self in {ClassificationEnumerator.LATIN_UPPERCASE, ClassificationEnumerator.LATIN_LOWERCASE}:
            return option_count <= len(ascii_uppercase)
        if self is ClassificationEnumerator.NUMERIC:
            return True
        return option_count <= _LARGEST_ROMAN_NUMBER

    def get_symbol_of_option_at(self, position: int) -> str:
        """Return the symbol of the option at a position.

        Args:
            position: The position of the option. The first option has position 0.

        Returns:
            The symbol, for example ``C`` for position 2 of :attr:`LATIN_UPPERCASE`.

        Raises:
            TimeFValidationError: If the position is negative or the series has no symbol for it.
        """
        if position < 0 or not self.has_symbols_for_option_count(position + 1):
            raise TimeFValidationError(
                f"the enumerator {self.value!r} has no symbol for the option position {position}"
            )
        if self is ClassificationEnumerator.LATIN_UPPERCASE:
            return ascii_uppercase[position]
        if self is ClassificationEnumerator.LATIN_LOWERCASE:
            return ascii_lowercase[position]
        if self is ClassificationEnumerator.NUMERIC:
            return str(position + 1)
        numeral = RomanNumeral(position + 1)
        if self is ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE:
            return numeral.to_lowercase()
        return numeral.to_uppercase()

    def get_position_of_option_with_symbol(self, symbol: str) -> int:
        """Return the position of the option that a symbol counts.

        The symbol must be written as :meth:`get_symbol_of_option_at` writes it. For example, ``01`` and
        ``IIII`` are not symbols.

        Args:
            symbol: The symbol, for example ``C``.

        Returns:
            The position of the option. The first option has position 0.

        Raises:
            TimeFValidationError: If the text is not a symbol of the series.
        """
        number: int | None = None
        if self is ClassificationEnumerator.LATIN_UPPERCASE:
            number = ascii_uppercase.find(symbol) + 1 if len(symbol) == 1 else None
        elif self is ClassificationEnumerator.LATIN_LOWERCASE:
            number = ascii_lowercase.find(symbol) + 1 if len(symbol) == 1 else None
        elif self is ClassificationEnumerator.NUMERIC:
            number = int(symbol) if symbol.isascii() and symbol.isdigit() and not symbol.startswith("0") else None
        else:
            # The parser reads both cases, so the case of the series is checked here.
            is_lowercase = self is ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE
            if symbol == (symbol.lower() if is_lowercase else symbol.upper()):
                try:
                    number = int(RomanNumeral.from_string(symbol))
                except (InvalidRomanNumeralError, OutOfRangeError):
                    number = None
        if not number:
            raise TimeFValidationError(f"{symbol!r} is not a symbol of the enumerator {self.value!r}")
        return number - 1


ALL_CLASSIFICATION_ENUMERATORS = frozenset(ClassificationEnumerator)
"""Every classification enumerator."""


def _check_key_affix(affix: str) -> str:
    """Make sure that a prefix or suffix of a key can be told apart from a symbol.

    Returns:
        The prefix or suffix.

    Raises:
        ValueError: If the text holds a letter, a digit, or a line break.
    """
    if any(character.isalnum() or character in "\r\n" for character in affix):
        raise ValueError(f"a key prefix or suffix must not hold a letter, a digit, or a line break, got {affix!r}")
    return affix


KeyAffix = Annotated[StrictStr, AfterValidator(_check_key_affix)]
"""The text before or after the symbol of a key, for example ``(`` or ``:``. It can be empty."""


class OptionKeyStyle(TimeFModel):
    """How a classification prompt writes the keys of its options.

    A prompt writes a key in two forms. The marker stands before an option in the list of options, for
    example ``A:`` or ``(a)``. A reference names an option in other text and in the answer, for example
    ``A`` in ``Answer with A, B or C.``

    The source ``(a) up (b) down. Only answer (a) or (b).`` has this style::

        OptionKeyStyle(
            enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
            marker_prefix="(",
            marker_suffix=")",
            reference_prefix="(",
            reference_suffix=")",
        )
    """

    enumerator: ClassificationEnumerator
    """The series of symbols that counts the options."""
    marker_prefix: KeyAffix = ""
    """The text before the symbol in a marker."""
    marker_suffix: KeyAffix = ""
    """The text after the symbol in a marker, for example ``:``."""
    reference_prefix: KeyAffix = ""
    """The text before the symbol in a reference."""
    reference_suffix: KeyAffix = ""
    """The text after the symbol in a reference."""

    def get_marker_of_option_at(self, position: int) -> str:
        """Return the marker of the option at a position, for example ``C:``.

        Args:
            position: The position of the option. The first option has position 0.

        Returns:
            The marker.

        Raises:
            TimeFValidationError: If the enumerator has no symbol for the position.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by get_symbol_of_option_at)
        return f"{self.marker_prefix}{self.enumerator.get_symbol_of_option_at(position)}{self.marker_suffix}"

    def get_reference_to_option_at(self, position: int) -> str:
        """Return the reference to the option at a position, for example ``C``.

        Args:
            position: The position of the option. The first option has position 0.

        Returns:
            The reference.

        Raises:
            TimeFValidationError: If the enumerator has no symbol for the position.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by get_symbol_of_option_at)
        return f"{self.reference_prefix}{self.enumerator.get_symbol_of_option_at(position)}{self.reference_suffix}"

    def get_position_of_option_with_reference(self, reference: str) -> int:
        """Return the position of the option that a reference names.

        Args:
            reference: The reference, written as :meth:`get_reference_to_option_at` writes it.

        Returns:
            The position of the option. The first option has position 0.

        Raises:
            TimeFValidationError: If the text is not a reference in this style.
        """
        symbol_end = len(reference) - len(self.reference_suffix)
        has_affixes = reference.startswith(self.reference_prefix) and reference.endswith(self.reference_suffix)
        if not has_affixes or symbol_end < len(self.reference_prefix):
            raise TimeFValidationError(f"{reference!r} is not a reference to an option in the style {self!r}")
        return self.enumerator.get_position_of_option_with_symbol(reference[len(self.reference_prefix) : symbol_end])

    def remove_marker_from_option_text(self, option_text: str, position: int) -> str:
        """Return the text of an option without the marker that stands before it.

        Use this method for a source that gives each option with its marker, for example
        ``["A: up", "B: down"]``.

        Args:
            option_text: The option with its marker, for example ``B: down``.
            position: The position of the option. The first option has position 0.

        Returns:
            The text after the marker, without the whitespace between them. The example gives ``down``.

        Raises:
            TimeFValidationError: If the text does not start with the marker of the position.
        """
        marker = self.get_marker_of_option_at(position)
        if not option_text.startswith(marker):
            raise TimeFValidationError(f"the option {option_text!r} does not start with the marker {marker!r}")
        return option_text[len(marker) :].lstrip()

    def get_variants_with_common_markers(
        self, enumerators: frozenset[ClassificationEnumerator]
    ) -> tuple["OptionKeyStyle", ...]:
        """Return this style with each of the given enumerators and each common marker form.

        The common marker forms are ``A:``, ``A)``, ``A.`` and ``(A)``. Each variant keeps the prefix and
        suffix of the references, so text that refers to an option stays in the form of the source.

        Args:
            enumerators: The enumerators of the variants.

        Returns:
            The variants, ordered by enumerator and then by marker form. The order is the same on each call.
        """
        return tuple(
            OptionKeyStyle(
                enumerator=enumerator,
                marker_prefix=prefix,
                marker_suffix=suffix,
                reference_prefix=self.reference_prefix,
                reference_suffix=self.reference_suffix,
            )
            for enumerator in sorted(enumerators)
            for prefix, suffix in _COMMON_MARKER_AFFIXES
        )
