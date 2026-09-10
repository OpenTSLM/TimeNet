"""``meta.csv`` rows turned into facts. No I/O at all.

``meta.csv`` describes the corpora SLIP drew from, one row each. A shard's ``dataset`` column names
one of them, so this table is how a row recovers its sampling rate and where it was published.
"""

from __future__ import annotations

from fractions import Fraction
import re

from timenet.errors import TimeFFormatError


_US_PER_S = 1_000_000
# RegularAxis holds the period as an int64 numerator of microseconds, so this is the coarsest period
# TimeF addresses. No meta.csv cell comes near it; the check is here so that a Freq at either end of
# the range fails with the cell named.
_MAX_PERIOD_US = 2**63 - 1

# Freq is prose, not a number. Every form the release actually uses, measured over every row.
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


def parse_period_us(freq: str | None) -> Fraction | None:
    """Turn one ``meta.csv`` ``Freq`` cell into a sampling period in microseconds.

    The column mixes a frequency (``65 Hz``), a period (``0.001 sec``, ``4 sec``), a calendar word
    (``Hourly``, ``Daily``), a period with a unit suffix (``30 min``, ``3h``, ``3 h``) and the null
    marker ``-``. All of those appear in the release.

    Args:
        freq: The cell, as the file states it. A row that states no cell at all gives ``None``.

    Returns:
        The period in microseconds, or ``None`` where the release states no rate.

    Raises:
        TimeFFormatError: If the cell is a form the release has not used before, since a new form
            means the table changed and guessing at it would invent a rate. Also if it states a
            period TimeF cannot address: zero, one finer than a microsecond, or one coarser than
            :data:`_MAX_PERIOD_US` of them. Those bounds are the format's, not SLIP's. All are
            raised here so the message names the cell, rather than reaching the axis, whose message
            knows nothing about ``meta.csv``.
    """
    if freq is None:
        raise TimeFFormatError("meta.csv states no Freq for a row; every row of this release states one")
    text = " ".join(freq.split()).lower()
    if text in {"", NO_RATE}:
        return None
    if text in _WORD_PERIODS_US:
        return Fraction(_WORD_PERIODS_US[text])
    match = _QUANTITY.match(text)
    if match is None:
        raise TimeFFormatError(f"meta.csv Freq holds {freq!r}, which is not a rate, a period, 'Hourly', 'Daily' or '-'")
    value, unit = Fraction(match["value"]), match["unit"]
    # Guarded for every unit and not only for Hz. A zero period reads as falsy, so an unguarded one
    # would reach the caller and pass for a corpus that states no rate at all.
    if value <= 0:
        raise TimeFFormatError(f"meta.csv Freq holds {freq!r}; a rate or a period must be above zero")
    period_us = Fraction(_US_PER_S) / value if unit == "hz" else value * _UNIT_SECONDS[unit] * _US_PER_S
    # Both ends, because `_QUANTITY` bounds neither. Caught here, where the cell can be named,
    # rather than in RegularAxis, whose message knows nothing about meta.csv.
    if not 1 <= period_us <= _MAX_PERIOD_US:
        raise TimeFFormatError(
            f"meta.csv Freq holds {freq!r}, a period of {float(period_us):g} us; TimeF addresses "
            f"no finer than one microsecond and no coarser than {_MAX_PERIOD_US} of them"
        )
    # Not the same test as the one above, and not a duplicate of it: RegularAxis bounds the
    # fraction's numerator rather than its value, and a rate written with about thirteen decimal
    # places separates the two — 0.0000000000003 Hz is a period of 3.3e18 us, inside the range,
    # whose numerator is 1e19 and is not. Keep both.
    if period_us.numerator > _MAX_PERIOD_US:
        raise TimeFFormatError(
            f"meta.csv Freq holds {freq!r}, a period of {period_us.numerator}/{period_us.denominator} us; "
            "TimeF stores it as a pair of int64 columns and that numerator does not fit one"
        )
    return period_us


def _cell(row: dict[str, str], column: str) -> str:
    """Give one cell of a ``meta.csv`` row, stripped.

    ``csv.DictReader`` states ``None`` for a column the header does not name and for a row that ends
    before it, so a table missing a column would otherwise reach the caller as a bare ``KeyError`` or
    ``AttributeError`` rather than as a corrupt file.

    Args:
        row: The parsed row.
        column: The header name to read.

    Returns:
        The cell's text, without surrounding space.

    Raises:
        TimeFFormatError: If the row states nothing under that column.
    """
    value = row.get(column)
    if value is None:
        raise TimeFFormatError(f"meta.csv states no {column!r} for a row; this connector reads it from every row")
    return value.strip()


def corpora(rows: list[dict[str, str]]) -> dict[str, SourceCorpus]:
    """Turn the rows of ``meta.csv`` into one :class:`SourceCorpus` per corpus, keyed by its name.

    Args:
        rows: The parsed rows, header names as the file writes them.

    Returns:
        One entry per row, keyed by the ``Dataset`` value a shard's ``dataset`` column will hold.

    Raises:
        TimeFFormatError: If the table states none of the four columns this reads, or a row states
            none of one of them; if two rows name the same corpus, since a shard row would then have
            two rates and no way to choose; or if :func:`parse_period_us` cannot read a ``Freq``.
    """
    found: dict[str, SourceCorpus] = {}
    for row in rows:
        name = _cell(row, "Dataset")
        if name in found:
            raise TimeFFormatError(f"meta.csv names the corpus {name!r} twice; a row cannot have two rates")
        found[name] = SourceCorpus(
            name=name,
            domain=_cell(row, "Domain"),
            source_url=_cell(row, "Source"),
            period_us=parse_period_us(row.get("Freq")),
        )
    return found
