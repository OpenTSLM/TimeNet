"""Which Binance archives the ``binance/spot-klines`` dataset reads, and where they live.

The connector and ``generate_checksums.py`` share these definitions, so the pinned digests always
cover exactly the archives that a build downloads.
"""

from collections.abc import Iterator


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
"""The root of Binance's public monthly spot kline archives."""
INTERVAL = "1h"
"""The candle interval of every archive: one hour."""
PAIRS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")
"""The trading pairs in the dataset, all quoted in USDT."""
FIRST_MONTH = (2023, 1)
"""The first month in the dataset, as ``(year, month)``."""
LAST_MONTH = (2025, 12)
"""The last month in the dataset, as ``(year, month)``."""
_DECEMBER = 12


def months() -> Iterator[tuple[int, int]]:
    """Yield every ``(year, month)`` from :data:`FIRST_MONTH` to :data:`LAST_MONTH`, inclusive.

    Yields:
        The months in calendar order.
    """
    year, month = FIRST_MONTH
    while (year, month) <= LAST_MONTH:
        yield year, month
        year, month = (year + 1, 1) if month == _DECEMBER else (year, month + 1)


def archive_name(pair: str, year: int, month: int) -> str:
    """Return the file name of one monthly archive, such as ``BTCUSDT-1h-2024-01.zip``.

    Args:
        pair: The trading pair, such as ``BTCUSDT``.
        year: The calendar year.
        month: The calendar month, 1 to 12.

    Returns:
        The archive's file name.
    """
    return f"{pair}-{INTERVAL}-{year:04d}-{month:02d}.zip"


def archive_url(pair: str, year: int, month: int) -> str:
    """Return the download URL of one monthly archive.

    Args:
        pair: The trading pair, such as ``BTCUSDT``.
        year: The calendar year.
        month: The calendar month, 1 to 12.

    Returns:
        The archive's URL on data.binance.vision.
    """
    return f"{BASE_URL}/{pair}/{INTERVAL}/{archive_name(pair, year, month)}"
