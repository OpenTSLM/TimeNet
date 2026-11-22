"""Connector for ``binance/spot-klines``: hourly Binance spot candles with daily market-event tasks.

The source is Binance's public archive of monthly spot kline files on data.binance.vision. Each
archive holds one CSV of hourly candles for one trading pair and one month. Every archive is pinned
by the SHA-256 digest in ``checksums.json``, which ``generate_checksums.py`` writes.

The dataset has one Record per trading pair. Its signals are the pair's whole hourly series, from
the first month to the last, on a one-hour regular axis whose origin is the first candle's open
time. An hour without a candle is a null, never a filled value.

Every day gets four tasks, scoped to the seven-day window that ends with the day: how far the price
moved, how volatile it was, how much traded, and a plain-English explanation that derives from the
three. :mod:`.events` holds the rules. Tasks split by the date of the described day; a window that
crosses a split boundary gets no tasks, so no split's inputs overlap another's.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    MissingReason,
    MissingValues,
    Split,
    Task,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet.types.wire import ValueDtype
from timenet_connectors.datasets.binance.spot_klines.events import (
    HOURS_PER_DAY,
    MOVE,
    VOLATILITY,
    VOLUME,
    WINDOW_HOURS,
    DayEvent,
    explain,
    measure_day,
    rationale,
)
from timenet_connectors.datasets.binance.spot_klines.layout import (
    INTERVAL,
    PAIRS,
    archive_name,
    archive_url,
    months,
)
from timenet_connectors.download import Artifact, download_files


_CHECKSUMS = Path(__file__).with_name("checksums.json")
_QUOTE_ASSET = "USDT"
_US_PER_HOUR = 3_600_000_000
_SECONDS_PER_HOUR = 3600
# Binance's spot archives store open times in milliseconds before 2025 and in microseconds from
# 2025 on. Any millisecond time up to the year 5138 is below this bound, and every microsecond time
# after 1973 is above it.
_MICROSECOND_TIMES_FROM = 10**14
# Archives at most this many concurrent downloads, to stay well inside Binance's rate limits.
_MAX_CONCURRENCY = 4

VALIDATION_FROM = date(2025, 1, 1)
"""The first described day of the validation split. Earlier days are training days."""
TEST_FROM = date(2025, 7, 1)
"""The first described day of the test split."""

_GAPS = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="Binance published no candle for this hour, for example during an exchange outage.",
)


def _spec(spec_type: str, name: str, dtype: ValueDtype = "float64") -> TimeSeriesSpec:
    """Build the spec of one candle field.

    Args:
        spec_type: The spec's type tag.
        name: The spec's display name.
        dtype: The value dtype. Prices and volumes keep the source's decimal precision as float64.

    Returns:
        The spec, which allows a missing hour.
    """
    return TimeSeriesSpec(
        spec_type=spec_type,
        name=name,
        unit_value=ureg.dimensionless,
        dtype=dtype,
        missing=_GAPS,
    )


# One spec per CSV column that the dataset keeps, keyed by the signal name.
_SPECS = {
    "open": _spec("open_price", "Open price (quote asset)"),
    "high": _spec("high_price", "High price (quote asset)"),
    "low": _spec("low_price", "Low price (quote asset)"),
    "close": _spec("close_price", "Close price (quote asset)"),
    "volume": _spec("base_volume", "Volume (base asset)"),
    "quote_volume": _spec("quote_volume", "Volume (quote asset)"),
    "trades": _spec("trade_count", "Number of trades", dtype="uint32"),
}
# The CSV column of each kept field. Binance's spot kline CSVs have no header row.
_COLUMNS = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5, "quote_volume": 7, "trades": 8}
_OPEN_TIME_COLUMN = 0


@dataclass(frozen=True)
class KlineArchives:
    """The downloaded monthly archives of every pair, in month order."""

    archives: Mapping[str, Sequence[Path]]
    """The archive paths of each pair, keyed by the pair, such as ``BTCUSDT``."""


@dataclass(frozen=True)
class Candles:
    """One pair's hourly candles on a gap-aware grid."""

    first_open_us: int
    """The open time of the first candle, in Unix microseconds."""
    values: Mapping[str, np.ndarray]
    """One float64 array per kept field, one value per hour, NaN where an hour has no candle."""

    @property
    def missing(self) -> np.ndarray:
        """A Boolean mask of the hours without a candle."""
        return np.isnan(self.values["close"])


def _read_rows(path: Path) -> list[list[str]]:
    """Read the rows of the one CSV inside a monthly archive.

    Args:
        path: The archive.

    Returns:
        The CSV rows, as lists of fields.

    Raises:
        TimeFValidationError: If the archive does not hold exactly one CSV.
    """
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise TimeFValidationError(f"{path.name} must hold exactly one CSV, found {names}")
        text = io.TextIOWrapper(archive.open(names[0]), encoding="utf-8")
        return [line.strip().split(",") for line in text if line.strip()]


def _open_time_us(field: str, path: Path) -> int:
    """Parse one open time, in milliseconds or microseconds, into Unix microseconds.

    Args:
        field: The open-time field of a CSV row.
        path: The archive, for the error message.

    Returns:
        The open time in Unix microseconds.

    Raises:
        TimeFValidationError: If the field is not an integer.
    """
    if not field.isdigit():
        raise TimeFValidationError(f"{path.name} has a non-numeric open time {field!r}")
    value = int(field)
    return value if value >= _MICROSECOND_TIMES_FROM else value * 1000


def read_candles(paths: Sequence[Path]) -> Candles:
    """Read a pair's archives into one hourly grid.

    Args:
        paths: The pair's archives, in month order.

    Returns:
        The candles, with NaN in every field of an hour that has no candle.

    Raises:
        TimeFValidationError: If there are no candles, a candle does not open on the hour, or the
            open times are not strictly increasing.
    """
    times: list[int] = []
    rows: list[list[str]] = []
    for path in paths:
        for row in _read_rows(path):
            open_us = _open_time_us(row[_OPEN_TIME_COLUMN], path)
            if open_us % _US_PER_HOUR:
                raise TimeFValidationError(f"{path.name} has a candle that does not open on the hour: {open_us}")
            if times and open_us <= times[-1]:
                raise TimeFValidationError(f"{path.name} has open times out of order or repeated at {open_us}")
            times.append(open_us)
            rows.append(row)
    if not times:
        raise TimeFValidationError("the archives hold no candles")
    hours = np.array([(time - times[0]) // _US_PER_HOUR for time in times], dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for field, column in _COLUMNS.items():
        grid = np.full(int(hours[-1]) + 1, np.nan, dtype=np.float64)
        grid[hours] = np.array([float(row[column]) for row in rows], dtype=np.float64)
        values[field] = grid
    return Candles(first_open_us=times[0], values=values)


def split_of(day: date) -> Split:
    """Return the split of a described day.

    Args:
        day: The described day.

    Returns:
        The day's split.
    """
    if day >= TEST_FROM:
        return Split.TEST
    if day >= VALIDATION_FROM:
        return Split.VALIDATION
    return Split.TRAIN


def _signal(field: str, candles: Candles, axis: RegularAxis) -> Signal:
    """Build one field's signal, with a null for every hour without a candle.

    Args:
        field: The field name, a key of :data:`_SPECS`.
        candles: The pair's candles.
        axis: The pair's hourly axis.

    Returns:
        The signal.
    """
    spec = _SPECS[field]
    grid = candles.values[field]
    missing = candles.missing
    if not missing.any():
        return Signal.from_values(grid.astype(spec.dtype), spec=spec, name=field, time_axis=axis)
    as_int = spec.dtype == "uint32"
    values = [
        None if gap else (int(value) if as_int else float(value)) for value, gap in zip(grid, missing, strict=True)
    ]
    return Signal.from_values(values, spec=spec, name=field, time_axis=axis)


def _day_tasks(record: Record, base_asset: str, event: DayEvent, scope: TimeInterval, day: date) -> list[Task]:
    """Build the four tasks of one described day.

    Args:
        record: The pair's record.
        base_asset: The traded asset, such as ``BTC``.
        event: The day's measurements.
        scope: The seven-day window that ends with the day.
        day: The described day.

    Returns:
        The price-move, volatility and volume classifications, then the explanation that derives
        from them.
    """
    split = split_of(day)
    metadata: JsonMapping = {
        "date": day.isoformat(),
        "day_return": round(event.day_return, 6),
        "move_z": round(event.move_z, 4),
        "volatility_ratio": round(event.volatility_ratio, 4),
        "volume_ratio": round(event.volume_ratio, 4),
    }
    move = ClassificationTask(
        prompt="How did the price move over the last 24 hours, compared with the six days before?",
        targets=event.move,
        label_set=MOVE,
        inputs=[record],
        scope=scope,
        split=split,
        metadata=metadata,
    )
    volatility = ClassificationTask(
        prompt="How large were the hourly swings over the last 24 hours, compared with the six days before?",
        targets=event.volatility,
        label_set=VOLATILITY,
        inputs=[record],
        scope=scope,
        split=split,
        metadata=metadata,
    )
    volume = ClassificationTask(
        prompt="How much traded over the last 24 hours, compared with the daily average of the six days before?",
        targets=event.volume,
        label_set=VOLUME,
        inputs=[record],
        scope=scope,
        split=split,
        metadata=metadata,
    )
    explanation = AnswerTask(
        prompt=(
            f"These are seven days of hourly {base_asset}/{_QUOTE_ASSET} candles. Explain what happened "
            "in the last 24 hours, compared with the six days before."
        ),
        targets=explain(event, base_asset),
        rationale=rationale(event, base_asset),
        from_tasks=(move, volatility, volume),
        inputs=[record],
        scope=scope,
        split=split,
        metadata=metadata,
    )
    return [move, volatility, volume, explanation]


class BinanceSpotKlinesConnector(BaseConnector[KlineArchives]):
    """Connector for hourly Binance spot candles of major USDT pairs (``binance/spot-klines``)."""

    async def download_async(self, cache_dir: Path) -> KlineArchives:  # ruff: ignore[no-self-use]
        """Download every pinned monthly archive, verifying each against its SHA-256 digest.

        Args:
            cache_dir: The directory that keeps the archives between builds.

        Returns:
            The archive paths of each pair, in month order.

        Raises:
            TimeFValidationError: If ``checksums.json`` has no digest for an archive.
        """
        checksums: dict[str, str] = json.loads(_CHECKSUMS.read_text())
        artifacts: list[Artifact] = []
        archives: dict[str, list[Path]] = {}
        for pair in PAIRS:
            for year, month in months():
                name = archive_name(pair, year, month)
                if name not in checksums:
                    raise TimeFValidationError(f"checksums.json has no digest for {name}; run generate_checksums.py")
                dest = cache_dir / pair / name
                artifacts.append(Artifact(archive_url(pair, year, month), dest, sha256=checksums[name]))
                archives.setdefault(pair, []).append(dest)
        await download_files(artifacts, max_concurrency=_MAX_CONCURRENCY)
        return KlineArchives(archives=archives)

    def convert(self, source: KlineArchives, /) -> TimeFDataset:
        """Build one record per pair and four tasks per measurable day.

        The connector holds each pair's values in memory, because the task labels derive from them.
        The whole dataset is a few megabytes.

        Args:
            source: The archives from :meth:`download_async`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        axis = RegularAxis.from_rate_hz(Fraction(1, _SECONDS_PER_HOUR))
        for pair, paths in source.archives.items():
            candles = read_candles(paths)
            base_asset = pair.removesuffix(_QUOTE_ASSET)
            signals = [_signal(field, candles, axis) for field in _SPECS]
            start = datetime.fromtimestamp(candles.first_open_us / 1_000_000, tz=UTC)
            record = dataset.add_record(
                Record(
                    start_time=TimeOrigin(start),
                    sources=[Source("Binance spot market", signals=signals)],
                    annotations=[
                        Annotation("pair", value=pair),
                        Annotation("base_asset", value=base_asset),
                        Annotation("quote_asset", value=_QUOTE_ASSET),
                        Annotation("interval", value=INTERVAL),
                    ],
                    metadata={"pair": pair, "exchange": "binance"},
                )
            )
            dataset.add_tasks(self._tasks(record, base_asset, candles, start))
        return dataset

    @staticmethod
    def _tasks(record: Record, base_asset: str, candles: Candles, start: datetime) -> list[Task]:
        """Build the tasks of every measurable day of one pair.

        Args:
            record: The pair's record.
            base_asset: The traded asset, such as ``BTC``.
            candles: The pair's candles.
            start: The open time of the pair's first candle.

        Returns:
            Four tasks per day whose window is complete and stays inside one split.
        """
        values = candles.values
        n_hours = len(values["close"])
        # The first window starts at the first midnight (UTC) on or after the first candle.
        window_start = (HOURS_PER_DAY - start.hour) % HOURS_PER_DAY
        tasks: list[Task] = []
        while window_start + WINDOW_HOURS <= n_hours:
            first_day = (start + timedelta(hours=window_start)).date()
            day = first_day + timedelta(days=WINDOW_HOURS // HOURS_PER_DAY - 1)
            event = measure_day(values["open"], values["close"], values["volume"], window_start)
            if event is not None and split_of(first_day) == split_of(day):
                scope = TimeInterval.seconds(
                    window_start * _SECONDS_PER_HOUR, (window_start + WINDOW_HOURS) * _SECONDS_PER_HOUR
                )
                tasks.extend(_day_tasks(record, base_asset, event, scope, day))
            window_start += HOURS_PER_DAY
        return tasks


CONNECTOR = BinanceSpotKlinesConnector
