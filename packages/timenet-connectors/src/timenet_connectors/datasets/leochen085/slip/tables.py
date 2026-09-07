"""``meta.csv`` rows turned into facts. No I/O at all.

``meta.csv`` describes the corpora SLIP drew from, one row each. A shard's ``dataset`` column names
one of them, so this table is how a row recovers its sampling rate and where it was published.
"""

from __future__ import annotations

from fractions import Fraction
import re

from timenet.errors import TimeFValidationError


_US_PER_S = 1_000_000

# Freq is prose, not a number. Every form the release actually uses, measured over its 37 rows.
_WORD_PERIODS_US: dict[str, int] = {
    "hourly": 3600 * _US_PER_S,
    "daily": 86400 * _US_PER_S,
}
_UNIT_SECONDS: dict[str, int] = {"sec": 1, "min": 60, "h": 3600}
_QUANTITY = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>hz|sec|min|h)$")

NO_RATE = "-"
"""What ``Freq`` holds for a corpus whose rate the release does not state."""


class SourceCorpus:
    """One row of ``meta.csv``: what the release says about a corpus it drew from."""

    def __init__(self, name: str, domain: str, source_url: str, period_us: Fraction | None) -> None:
        self.name = name  # the value a shard's `dataset` column holds
        self.domain = domain  # `meta.csv`'s Domain, which the `category` column repeats
        self.source_url = source_url  # where that corpus was published
        self.period_us = period_us  # None when the release states no rate

    def __repr__(self) -> str:
        return f"SourceCorpus(name={self.name!r}, period_us={self.period_us!r})"


def parse_period_us(freq: str) -> Fraction | None:
    """Turn one ``meta.csv`` ``Freq`` cell into a sampling period in microseconds.

    The column mixes a frequency (``65 Hz``), a period (``0.001 sec``, ``4 sec``), a calendar word
    (``Hourly``, ``Daily``), a period with a unit suffix (``30 min``, ``3h``, ``3 h``) and the null
    marker ``-``. All of those appear in the release.

    Args:
        freq: The cell, as the file states it.

    Returns:
        The period in microseconds, or ``None`` where the release states no rate.

    Raises:
        TimeFValidationError: If the cell is a form the release has not used before, since a new
            form means the table changed and guessing at it would invent a rate.
    """
    text = " ".join(freq.split()).lower()
    if text in {"", NO_RATE}:
        return None
    if text in _WORD_PERIODS_US:
        return Fraction(_WORD_PERIODS_US[text])
    match = _QUANTITY.match(text)
    if match is None:
        raise TimeFValidationError(
            f"meta.csv Freq holds {freq!r}, which is not a rate, a period, 'Hourly', 'Daily' or '-'"
        )
    value, unit = Fraction(match["value"]), match["unit"]
    if unit == "hz":
        if value <= 0:
            raise TimeFValidationError(f"meta.csv Freq holds {freq!r}; a rate in Hz must be above zero")
        return Fraction(_US_PER_S) / value
    return value * _UNIT_SECONDS[unit] * _US_PER_S


def corpora(rows: list[dict[str, str]]) -> dict[str, SourceCorpus]:
    """Turn the rows of ``meta.csv`` into one :class:`SourceCorpus` per corpus, keyed by its name.

    Args:
        rows: The parsed rows, header names as the file writes them.

    Returns:
        One entry per row, keyed by the ``Dataset`` value a shard's ``dataset`` column will hold.

    Raises:
        TimeFValidationError: If two rows name the same corpus, since a shard row would then have
            two rates and no way to choose.
    """
    found: dict[str, SourceCorpus] = {}
    for row in rows:
        name = row["Dataset"].strip()
        if name in found:
            raise TimeFValidationError(f"meta.csv names the corpus {name!r} twice; a row cannot have two rates")
        found[name] = SourceCorpus(
            name=name,
            domain=row["Domain"].strip(),
            source_url=row["Source"].strip(),
            period_us=parse_period_us(row["Freq"]),
        )
    return found
