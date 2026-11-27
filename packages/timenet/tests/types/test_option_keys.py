import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ALL_CLASSIFICATION_ENUMERATORS, ClassificationEnumerator, OptionKeyStyle


LETTERS_WITH_COLON = OptionKeyStyle(enumerator=ClassificationEnumerator.LATIN_UPPERCASE, marker_suffix=":")
LETTERS_IN_PARENTHESES = OptionKeyStyle(
    enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
    marker_prefix="(",
    marker_suffix=")",
    reference_prefix="(",
    reference_suffix=")",
)


@pytest.mark.parametrize(
    ("enumerator", "position", "symbol"),
    [
        (ClassificationEnumerator.LATIN_UPPERCASE, 0, "A"),
        (ClassificationEnumerator.LATIN_UPPERCASE, 25, "Z"),
        (ClassificationEnumerator.LATIN_LOWERCASE, 3, "d"),
        (ClassificationEnumerator.NUMERIC, 0, "1"),
        (ClassificationEnumerator.NUMERIC, 99, "100"),
        (ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE, 3, "iv"),
        (ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE, 8, "ix"),
        (ClassificationEnumerator.ROMAN_NUMERIC_UPPERCASE, 13, "XIV"),
        (ClassificationEnumerator.ROMAN_NUMERIC_UPPERCASE, 3998, "MMMCMXCIX"),
    ],
)
def test_an_enumerator_writes_and_reads_the_symbol_of_a_position(enumerator, position, symbol):
    assert enumerator.get_symbol_of_option_at(position) == symbol
    assert enumerator.get_position_of_option_with_symbol(symbol) == position


@pytest.mark.parametrize(
    ("enumerator", "text"),
    [
        (ClassificationEnumerator.LATIN_UPPERCASE, "a"),
        (ClassificationEnumerator.LATIN_UPPERCASE, "AA"),
        (ClassificationEnumerator.LATIN_LOWERCASE, ""),
        (ClassificationEnumerator.NUMERIC, "0"),
        (ClassificationEnumerator.NUMERIC, "01"),
        (ClassificationEnumerator.NUMERIC, "²"),
        (ClassificationEnumerator.ROMAN_NUMERIC_UPPERCASE, "IIII"),
        (ClassificationEnumerator.ROMAN_NUMERIC_UPPERCASE, "iv"),
        (ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE, "Iv"),
        (ClassificationEnumerator.ROMAN_NUMERIC_LOWERCASE, "vx"),
    ],
)
def test_an_enumerator_rejects_text_that_is_not_one_of_its_symbols(enumerator, text):
    with pytest.raises(TimeFValidationError, match="is not a symbol of the enumerator"):
        enumerator.get_position_of_option_with_symbol(text)


def test_an_enumerator_has_no_symbol_past_the_end_of_its_series():
    assert ClassificationEnumerator.LATIN_LOWERCASE.has_symbols_for_option_count(26)
    assert not ClassificationEnumerator.LATIN_LOWERCASE.has_symbols_for_option_count(27)
    with pytest.raises(TimeFValidationError, match="has no symbol for the option position 26"):
        ClassificationEnumerator.LATIN_LOWERCASE.get_symbol_of_option_at(26)
    with pytest.raises(TimeFValidationError, match="has no symbol for the option position -1"):
        ClassificationEnumerator.NUMERIC.get_symbol_of_option_at(-1)


def test_a_key_style_writes_a_marker_and_a_reference_in_their_own_forms():
    assert LETTERS_WITH_COLON.get_marker_of_option_at(2) == "C:"
    assert LETTERS_WITH_COLON.get_reference_to_option_at(2) == "C"
    assert LETTERS_IN_PARENTHESES.get_marker_of_option_at(1) == "(b)"
    assert LETTERS_IN_PARENTHESES.get_reference_to_option_at(1) == "(b)"


def test_a_key_style_reads_only_a_reference_in_its_own_form():
    assert LETTERS_IN_PARENTHESES.get_position_of_option_with_reference("(c)") == 2
    for text in ("c", "(c", ")", "(C)"):
        with pytest.raises(TimeFValidationError):
            LETTERS_IN_PARENTHESES.get_position_of_option_with_reference(text)


def test_a_key_style_removes_only_the_marker_of_the_given_position():
    assert LETTERS_WITH_COLON.remove_marker_from_option_text("B:  down-up", 1) == "down-up"
    with pytest.raises(TimeFValidationError, match="does not start with the marker 'A:'"):
        LETTERS_WITH_COLON.remove_marker_from_option_text("B: down-up", 0)


@pytest.mark.parametrize("affix", ["a", "1", "\n", ") or"])
def test_a_key_style_rejects_an_affix_that_could_be_read_as_a_symbol(affix):
    with pytest.raises(ValueError, match="must not hold a letter, a digit, or a line break"):
        OptionKeyStyle(enumerator=ClassificationEnumerator.NUMERIC, marker_suffix=affix)


def test_the_variants_of_a_key_style_keep_the_form_of_its_references():
    variants = LETTERS_IN_PARENTHESES.get_variants_with_common_markers(ALL_CLASSIFICATION_ENUMERATORS)
    markers = {variant.get_marker_of_option_at(0) for variant in variants}
    assert len(variants) == len(markers) == 5 * 4
    assert {"A:", "a)", "1.", "(i)", "(I)"} <= markers
    assert {variant.get_reference_to_option_at(0) for variant in variants} == {"(A)", "(a)", "(1)", "(i)", "(I)"}
