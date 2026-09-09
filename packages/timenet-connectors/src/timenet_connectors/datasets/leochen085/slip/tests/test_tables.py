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
