"""The ``meta.csv`` decoders. No file is opened here; every input is a literal."""

from fractions import Fraction

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.leochen085.slip.tables import corpora, parse_period_us


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("65 Hz", Fraction(1_000_000, 65)),
        ("30 Hz", Fraction(1_000_000, 30)),
        ("0.001 sec", Fraction(1_000)),
        ("0.002 sec", Fraction(2_000)),
        ("0.004 sec", Fraction(4_000)),
        ("0.008 sec", Fraction(8_000)),
        ("0.02 sec", Fraction(20_000)),
        ("0.033 sec", Fraction(33_000)),
        ("4 sec", Fraction(4_000_000)),
        ("30 min", Fraction(1_800_000_000)),
        ("3h", Fraction(10_800_000_000)),
        ("3 h", Fraction(10_800_000_000)),
        ("Hourly", Fraction(3_600_000_000)),
        ("Daily", Fraction(86_400_000_000)),
    ],
)
def test_every_form_the_release_uses(freq: str, expected: Fraction) -> None:
    # All 14 spellings the 37 rows of the pinned meta.csv hold, plus "-" below.
    assert parse_period_us(freq) == expected


def test_the_two_spellings_of_three_hours_agree() -> None:
    # ERA5 Pressure says "3h" and ERA5 Surface says "3 h" for the same rate.
    assert parse_period_us("3h") == parse_period_us("3 h")


@pytest.mark.parametrize("freq", ["-", "", "  "])
def test_no_stated_rate_gives_none(freq: str) -> None:
    assert parse_period_us(freq) is None


@pytest.mark.parametrize("freq", ["every so often", "12 furlongs", "0 Hz", "0 sec", "0.0 min"])
def test_an_unknown_form_raises_rather_than_guessing(freq: str) -> None:
    # A zero is refused whichever unit carries it. A zero period would otherwise read as "no rate",
    # which is what the table writes as "-", and a whole corpus would lose its time in silence.
    with pytest.raises(TimeFFormatError):
        parse_period_us(freq)


@pytest.mark.parametrize("freq", ["2000000 Hz", "1000001 Hz", "0.0000005 sec"])
def test_a_period_finer_than_a_microsecond_raises_here_naming_the_cell(freq: str) -> None:
    # RegularAxis rejects it too, but its message knows nothing about meta.csv. Raised here so the
    # build failure names the cell that caused it.
    with pytest.raises(TimeFFormatError, match="one microsecond"):
        parse_period_us(freq)
    # The bound is on the computed period, so it holds for a rate and for a period alike. One
    # microsecond is the finest the format carries, written either way, and both still parse.
    assert parse_period_us("1000000 Hz") == Fraction(1)
    assert parse_period_us("0.000001 sec") == Fraction(1)


@pytest.mark.parametrize("freq", ["99999999999 h", "9999999999999 sec", "9223372036855 sec"])
def test_a_period_past_the_int64_ceiling_raises_here_too(freq: str) -> None:
    # The quantity pattern bounds neither end, so a period can overshoot as well as undershoot.
    with pytest.raises(TimeFFormatError, match="no coarser"):
        parse_period_us(freq)
    # One second below that last cell is the coarsest whole second the format carries, and it parses.
    assert parse_period_us("9223372036854 sec") == Fraction(9_223_372_036_854_000_000)


def test_a_period_inside_the_range_whose_numerator_is_not_raises_here_too() -> None:
    # The axis bounds the fraction's numerator, not its value, so the range check above does not
    # cover this: 0.0000000000003 Hz is a period of 3.3e18 us, inside the range, over a numerator of
    # 1e19, which is not.
    with pytest.raises(TimeFFormatError, match="numerator"):
        parse_period_us("0.0000000000003 Hz")


def test_a_row_that_states_no_freq_raises_rather_than_failing_on_none() -> None:
    # csv.DictReader gives None for a row that ends early, which would otherwise be an AttributeError.
    with pytest.raises(TimeFFormatError, match="no Freq"):
        parse_period_us(None)


@pytest.mark.parametrize("missing", ["Dataset", "Domain", "Source"])
def test_a_table_missing_a_column_this_reads_raises(missing: str) -> None:
    # A header without the column gives None for every row, which would otherwise be a KeyError.
    row = {"Dataset": "A", "Domain": "Health", "Source": "https://example.test/a", "Freq": "-"}
    del row[missing]
    with pytest.raises(TimeFFormatError, match=missing):
        corpora([row])


def test_corpora_keys_on_the_name_a_shard_will_hold() -> None:
    rows = [
        {"Dataset": "BIDMC32HR", "Domain": "Health", "Source": "https://example.test/a", "Freq": "-"},
        {"Dataset": "Wind Farms", "Domain": "Energy", "Source": "https://example.test/b", "Freq": "4 sec"},
    ]
    found = corpora(rows)
    assert set(found) == {"BIDMC32HR", "Wind Farms"}
    assert found["BIDMC32HR"].period_us is None
    assert found["Wind Farms"].period_us == Fraction(4_000_000)
    assert found["Wind Farms"].domain == "Energy"


def test_a_corpus_named_twice_raises() -> None:
    row = {"Dataset": "Twice", "Domain": "Health", "Source": "", "Freq": "-"}
    with pytest.raises(TimeFFormatError, match="twice"):
        corpora([row, row])
