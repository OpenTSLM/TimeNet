from collections import Counter
import csv
import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import AnswerTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.opentslm.m4_caption.connector import (
    M4_CAPTION_SHA256,
    SPLITS_PATH,
    M4CaptionConnector,
    M4CaptionReader,
)
from timenet_connectors.splitting import FrozenSplits


_SERIES_IDS = {
    "Daily": ["D1", "D2"],
    "Hourly": ["H1"],
    "Monthly": ["M1", "M2", "M3"],
    "Quarterly": ["Q1", "Q2"],
    "Weekly": ["W1"],
    "Yearly": ["Y1"],
}
"""Ten series in the order that OpenTSLM's loader reads the frequencies."""

_SPLITS = {"source_sha256": M4_CAPTION_SHA256, "validation": ["W1"], "test": ["M3"]}
"""A split file for the ten series. Every series that it does not name is in the train split."""
_EXPECTED_SPLIT = {"M3": Split.TEST, "W1": Split.VALIDATION}

_FLOAT32_VALUES = [1089.199951171875, 1078.9100341796875, 8000.0, 195.92799377441406]
"""Values as the release writes them: M4 values rounded to float32, printed with every digit."""


_LONGEST_SERIES = 9919
"""The number of values of the longest series in the release, a daily one. Its CSV cell has more
characters than the default cell limit of the csv module."""


def _values(m4_id: str) -> list[float]:
    # A distinct length and offset for each series, so a mix-up of two series changes the values. The sum
    # is in float32, so each value stays a float32 value, like in the release.
    shift = np.float32(sum(map(ord, m4_id)))
    values = np.array(_FLOAT32_VALUES, dtype=np.float32) + shift
    if m4_id == "D1":
        return np.resize(values, _LONGEST_SERIES).tolist()
    return values.tolist()[: 2 + len(m4_id)]


def _caption(m4_id: str) -> str:
    return f"Caption of {m4_id}.\n\nA second paragraph."


def _write_csv(path: Path, columns: tuple[str, str], rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _expected_split(m4_id: str) -> Split:
    return _EXPECTED_SPLIT.get(m4_id, Split.TRAIN)


def _write_release(
    directory: Path,
    *,
    series_rows: dict[str, list[tuple[str, str]]] | None = None,
    captions: dict[str, str] | None = None,
    splits: dict[str, str | list[str]] = _SPLITS,
) -> M4CaptionReader:
    # Like the archive, the files sit in a folder of their own, beside the files that the loader skips.
    root = directory / "M4TimeSeriesCaptionDataset"
    root.mkdir()
    for frequency, m4_ids in _SERIES_IDS.items():
        default_rows = [(f"series-{m4_id}", json.dumps(_values(m4_id))) for m4_id in m4_ids]
        caption_rows = [(f"series-{m4_id}", (captions or {}).get(m4_id, _caption(m4_id))) for m4_id in m4_ids]
        _write_csv(
            root / f"m4_series_{frequency}.csv", ("id", "series"), (series_rows or {}).get(frequency, default_rows)
        )
        _write_csv(root / f"m4_captions_{frequency}.csv", ("id", "caption"), caption_rows)
    _write_csv(root / "m4_captions.csv", ("id", "caption"), [("series-W1", "A caption that OpenTSLM never used.")])
    splits_path = directory / "splits.json.xz"
    splits_path.write_bytes(lzma.compress(json.dumps(splits).encode()))
    return M4CaptionReader(directory=root, splits_path=splits_path)


def _convert(directory: Path, **changes) -> TimeFDataset:
    return M4CaptionConnector().convert([_write_release(directory, **changes)])


def test_each_series_becomes_a_record_and_a_caption_task(tmp_path):
    tasks = _convert(tmp_path).tasks_of(AnswerTask)
    expected = [(frequency, m4_id) for frequency, m4_ids in _SERIES_IDS.items() for m4_id in m4_ids]
    assert len(tasks) == len(expected)
    for task, (frequency, m4_id) in zip(tasks, expected, strict=True):
        (record,) = task.inputs
        (signal,) = record.signals
        # The release values are float32 values, so float32 keeps every digit.
        assert signal.to_numpy().dtype == np.float32
        np.testing.assert_array_equal(signal.to_numpy(), np.array(_values(m4_id), dtype=np.float32))
        # The release has no timestamps, and a month or a year has no fixed length.
        assert signal.time_axis == OrdinalAxis()
        assert signal.spec.unit_value is None
        assert record.metadata == {"m4_id": m4_id}
        (annotation,) = record.annotations
        assert (annotation.key, read_value(annotation.value)) == ("frequency", frequency)
        assert task.targets is not None
        assert [read_value(target) for target in task.targets] == [_caption(m4_id)]
        # The prompt of OpenTSLM's loader, the same for every series.
        assert read_value(task.prompt) == (
            "Please generate a detailed caption for this time-series, describing it as accurately as possible."
        )
        assert task.split == _expected_split(m4_id)


def test_a_series_whose_caption_is_an_error_message_is_left_out(tmp_path):
    # Two of the three error messages of the caption API that the release has instead of a caption.
    captions = {
        "D2": "Error generating caption: Connection error.",
        "Q1": "Error generating caption: Error code: 404",
    }
    tasks = _convert(tmp_path, captions=captions).tasks_of(AnswerTask)
    assert [task.inputs[0].metadata["m4_id"] for task in tasks] == ["D1", "H1", "M1", "M2", "M3", "Q2", "W1", "Y1"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_sha256": "0" * 64}, "the split was made from the source 0+, not from the pinned"),
        ({"test": ["M3", "W1"]}, r"\['W1'\] are named more than once"),
        ({"test": ["M3", "M9"]}, r"the split names \['M9'\] that the source does not have"),
    ],
    ids=["another release", "a series in two splits", "a series that the release does not have"],
)
def test_a_split_file_that_does_not_fit_the_release_raises(tmp_path, changes, message):
    with pytest.raises(TimeNetConnectorError, match=message):
        _convert(tmp_path, splits=_SPLITS | changes)


_DATASET_SERIES_COUNTS = {"D": 3974, "H": 414, "M": 47943, "Q": 23604, "W": 359, "Y": 23000}
"""The number of series of each frequency in the dataset, by the letter that starts their M4 ids. The
dataset leaves out 253 daily, 57 monthly and 396 quarterly series, whose caption is an error message."""


def test_the_frozen_split_is_of_the_pinned_release_and_has_a_tenth_of_each_frequency_in_validation_and_test():
    splits = FrozenSplits[str].read_from_file(SPLITS_PATH)
    assert splits.source_sha256 == M4_CAPTION_SHA256
    for m4_ids in (splits.validation, splits.test):
        counts = Counter(m4_id[0] for m4_id in m4_ids)
        for letter, dataset_count in _DATASET_SERIES_COUNTS.items():
            # A tenth of the series of each frequency, rounded down or up.
            assert dataset_count // 10 <= counts[letter] <= -(-dataset_count // 10)


@pytest.mark.parametrize(
    ("series_rows", "message"),
    [
        ({"Daily": [("series-D1", "[1.0, null]"), ("series-D2", "[1.0]")]}, "line 2: .*Input should be a valid number"),
        ({"Daily": [("series-D1", "[1.0, true]"), ("series-D2", "[1.0]")]}, "line 2: .*Input should be a valid number"),
        ({"Daily": [("series-D1", "[]"), ("series-D2", "[1.0]")]}, "line 2: .*List should have at least 1 item"),
        ({"Daily": [("series-D2", "[1.0]"), ("series-D1", "[1.0]")]}, "series id 'series-D2' does not match"),
        ({"Daily": [("series-M9", "[1.0]"), ("series-D2", "[1.0]")]}, "is not a Daily M4 series id"),
        ({"Daily": [("series-D1", "[1.0]")]}, "has 1 series rows but 2 caption rows"),
    ],
    ids=[
        "a null value",
        "a Boolean value",
        "an empty series",
        "series and captions out of step",
        "an id of another frequency",
        "a series row too few",
    ],
)
def test_a_malformed_series_file_raises_with_its_location(tmp_path, series_rows, message):
    # The CSV reader rejects a cell that does not hold a list of numbers itself, and names the line of the file.
    with pytest.raises(TimeNetConnectorError, match=rf"(?s)m4_series_Daily\.csv.*{message}"):
        _convert(tmp_path, series_rows=series_rows)


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
