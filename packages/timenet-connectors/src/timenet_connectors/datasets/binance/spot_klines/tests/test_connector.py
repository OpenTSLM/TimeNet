from datetime import UTC, date, datetime
import math
import zipfile

import numpy as np
import pytest

from timenet.dataset import RegularAxis, TimeFDataset
from timenet.engine import store_dataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import AnswerTask, ClassificationTask, Split, TimeInterval, read_value
from timenet_connectors.datasets.binance.spot_klines import BinanceSpotKlinesConnector
from timenet_connectors.datasets.binance.spot_klines.connector import KlineArchives, read_candles, split_of
from timenet_connectors.datasets.binance.spot_klines.events import measure_day


# Synthetic candles shaped exactly like Binance's spot kline CSVs. They are generated here, not taken
# from Binance's archive, so the repo ships no source bytes. Ten days from 2023-01-01 00:00 UTC.
_START_MS = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
_HOUR_MS = 3_600_000
_HOURS = 240
_EVENT_DAY = 9
"""The last day of the fixture: a sharp rally with a volatility spike and a volume surge."""


def _log_returns() -> list[float]:
    # Ordinary hours alternate +0.1% and -0.1%, so every ordinary day ends where it started. The event
    # day alternates +2% and 0%, so it gains 24% in log terms with ten times the usual hourly swings.
    returns = [0.001 if hour % 2 == 0 else -0.001 for hour in range(_HOURS)]
    for hour in range(_EVENT_DAY * 24, _HOURS):
        returns[hour] = 0.02 if hour % 2 == 0 else 0.0
    return returns


def _rows(hours: range, *, skip: frozenset[int] = frozenset(), microseconds: bool = False) -> list[str]:
    rows = []
    price = 100.0
    for hour, log_return in enumerate(_log_returns()):
        open_ = price
        price = open_ * math.exp(log_return)
        if hour not in hours or hour in skip:
            continue
        volume = 10.0 if hour >= _EVENT_DAY * 24 else 1.0
        open_time = _START_MS + hour * _HOUR_MS
        close_time = open_time + _HOUR_MS - 1
        if microseconds:
            open_time, close_time = open_time * 1000, close_time * 1000 + 999
        fields = [open_time, open_, max(open_, price), min(open_, price), price, volume, close_time]
        fields += [volume * price, 7, volume / 2, volume * price / 2, 0]
        rows.append(",".join(str(field) for field in fields))
    return rows


def _archive(tmp_path, name: str, rows: list[str]):
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}.csv", "\n".join(rows) + "\n")
    return path


def _archives(tmp_path, *, skip: frozenset[int] = frozenset()) -> KlineArchives:
    # The first archive stores open times in milliseconds and the second in microseconds, like
    # Binance's archives before and after January 2025.
    first = _archive(tmp_path, "TESTUSDT-1h-a", _rows(range(120), skip=skip))
    second = _archive(tmp_path, "TESTUSDT-1h-b", _rows(range(120, _HOURS), skip=skip, microseconds=True))
    return KlineArchives(archives={"TESTUSDT": [first, second]})


def _convert(tmp_path, **kwargs) -> TimeFDataset:
    return BinanceSpotKlinesConnector().convert(_archives(tmp_path, **kwargs))


def _labels(dataset: TimeFDataset, label_set_name: str) -> list[str]:
    return [
        str(read_value(task.targets[0]))
        for task in dataset.tasks
        if isinstance(task, ClassificationTask)
        and task.targets is not None
        and task.label_set.annotations[0].value == label_set_name
    ]


def test_metadata():
    card = BinanceSpotKlinesConnector().metadata()
    assert card.dataset_id == "binance/spot-klines"
    assert card.license_url is not None


def test_one_record_per_pair_on_an_hourly_grid(tmp_path):
    dataset = _convert(tmp_path)
    (record,) = dataset.records
    assert record.metadata == {"pair": "TESTUSDT", "exchange": "binance"}
    assert record.start_time.timestamp == _START_MS * 1000
    assert {signal.name for signal in record.signals} == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
    }
    close = next(signal for signal in record.signals if signal.name == "close")
    assert close.n_values == _HOURS
    assert isinstance(close.time_axis, RegularAxis)
    assert close.time_axis.time_offset_us(1) == _HOUR_MS * 1000
    assert {annotation.key: annotation.value for annotation in record.annotations} == {
        "pair": "TESTUSDT",
        "base_asset": "TEST",
        "quote_asset": "USDT",
        "interval": "1h",
    }


def test_four_tasks_for_every_complete_window(tmp_path):
    dataset = _convert(tmp_path)
    # Windows start at midnight on days 0 to 3, and each describes its seventh day, days 6 to 9.
    assert len(dataset.tasks) == 16
    answers = [task for task in dataset.tasks if isinstance(task, AnswerTask)]
    assert [task.metadata["date"] for task in answers] == ["2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"]
    assert all(task.split == Split.TRAIN for task in dataset.tasks)
    assert isinstance(answers[0].scope, TimeInterval)
    assert (answers[0].scope.start_us, answers[0].scope.end_us) == (0, 168 * _HOUR_MS * 1000)
    assert all(len(task.from_tasks) == 3 for task in answers)


def test_labels_find_the_event_day(tmp_path):
    dataset = _convert(tmp_path)
    assert _labels(dataset, "price_move") == ["flat", "flat", "flat", "sharp_rally"]
    assert _labels(dataset, "volatility") == ["normal", "normal", "normal", "spike"]
    assert _labels(dataset, "volume") == ["normal", "normal", "normal", "surge"]
    last = next(task for task in reversed(dataset.tasks) if isinstance(task, AnswerTask))
    assert last.targets is not None
    explanation = str(read_value(last.targets[0]))
    assert explanation.startswith("TEST rose 27.12% over the last 24 hours")
    assert "a sharp rally" in explanation
    assert "a volume surge" in explanation


def test_a_missing_hour_stays_null_and_its_windows_get_no_tasks(tmp_path):
    dataset = _convert(tmp_path, skip=frozenset({30}))
    close = next(signal for signal in dataset.records[0].signals if signal.name == "close")
    assert close.to_arrow().is_null().to_pylist().index(True) == 30
    assert close.to_arrow().null_count == 1
    # Hour 30 falls in day 1, so the windows that start on days 0 and 1 are incomplete.
    answers = [task for task in dataset.tasks if isinstance(task, AnswerTask)]
    assert [task.metadata["date"] for task in answers] == ["2023-01-09", "2023-01-10"]


def test_rejects_a_candle_off_the_hour(tmp_path):
    rows = _rows(range(3))
    rows[1] = rows[1].replace(str(_START_MS + _HOUR_MS), str(_START_MS + _HOUR_MS + 1), 1)
    with pytest.raises(TimeFValidationError, match="does not open on the hour"):
        read_candles([_archive(tmp_path, "bad", rows)])


def test_rejects_repeated_open_times(tmp_path):
    rows = _rows(range(3))
    with pytest.raises(TimeFValidationError, match="out of order or repeated"):
        read_candles([_archive(tmp_path, "repeat", [*rows, rows[-1]])])


def test_rejects_an_archive_without_one_csv(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "no candles")
    with pytest.raises(TimeFValidationError, match="exactly one CSV"):
        read_candles([path])


def test_measure_day_skips_a_window_without_movement():
    flat = np.full(168, 100.0)
    assert measure_day(flat, flat, np.ones(168), 0) is None


def test_split_boundaries():
    assert split_of(date(2024, 12, 31)) == Split.TRAIN
    assert split_of(date(2025, 1, 1)) == Split.VALIDATION
    assert split_of(date(2025, 6, 30)) == Split.VALIDATION
    assert split_of(date(2025, 7, 1)) == Split.TEST


def test_round_trip_through_timef(tmp_path):
    dataset = _convert(tmp_path)
    root = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(open_local_version(root)) as reader:
        loaded = reader.read()
    assert len(loaded.records) == 1
    assert len(loaded.tasks) == 16
    close = next(signal for signal in loaded.records[0].signals if signal.name == "close")
    assert close.to_numpy()[0] == pytest.approx(100.0 * math.exp(0.001))
