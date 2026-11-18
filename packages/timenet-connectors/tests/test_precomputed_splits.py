import json
import lzma
from pathlib import Path

from pydantic import NonNegativeInt
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types.splits import Split
from timenet_connectors.splitting import PrecomputedSplits


_SOURCE_SHA256 = "a" * 64


def _write_split_file(path: Path, **splits: object) -> Path:
    content = json.dumps({"source_sha256": _SOURCE_SHA256} | splits).encode()
    path.write_bytes(lzma.compress(content) if path.suffix == ".xz" else content)
    return path


def test_rows_by_position_get_their_split_and_the_rest_are_train(tmp_path):
    path = _write_split_file(tmp_path / "splits.json", validation=[1], test=[3])
    splits = PrecomputedSplits[NonNegativeInt].read_from_file(path)
    assert splits.assign_splits(range(5), source_sha256=_SOURCE_SHA256) == [
        Split.TRAIN,
        Split.VALIDATION,
        Split.TRAIN,
        Split.TEST,
        Split.TRAIN,
    ]


def test_series_by_id_get_their_split_from_an_xz_file(tmp_path):
    path = _write_split_file(tmp_path / "splits.json.xz", validation=["W1"], test=["M3"])
    splits = PrecomputedSplits[str].read_from_file(path)
    assert splits.assign_splits(["M3", "D1", "W1"], source_sha256=_SOURCE_SHA256) == [
        Split.TEST,
        Split.TRAIN,
        Split.VALIDATION,
    ]


def test_a_split_made_from_another_source_raises(tmp_path):
    splits = PrecomputedSplits[str].read_from_file(_write_split_file(tmp_path / "splits.json", validation=[], test=[]))
    with pytest.raises(TimeNetConnectorError, match=f"made from the source {_SOURCE_SHA256}, not from the pinned 0+"):
        splits.assign_splits(["M1"], source_sha256="0" * 64)


def test_a_split_that_names_a_key_the_source_does_not_have_raises(tmp_path):
    splits = PrecomputedSplits[NonNegativeInt].read_from_file(
        _write_split_file(tmp_path / "splits.json", validation=[1], test=[5])
    )
    with pytest.raises(TimeNetConnectorError, match=r"the split names \[5\] that the source does not have"):
        splits.assign_splits(range(5), source_sha256=_SOURCE_SHA256)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.dumps({"source_sha256": _SOURCE_SHA256, "validation": [1], "test": [1]}), r"\[1\] are named more"),
        (json.dumps({"source_sha256": _SOURCE_SHA256, "validation": [-1], "test": []}), "greater than or equal to 0"),
        ("not JSON", "Invalid JSON"),
    ],
    ids=["a key in two splits", "a negative position", "a file that is not JSON"],
)
def test_a_malformed_split_file_raises_with_its_name(tmp_path, content, message):
    path = tmp_path / "splits.json"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=rf"(?s)splits\.json does not hold a valid split: .*{message}"):
        PrecomputedSplits[NonNegativeInt].read_from_file(path)


def test_an_xz_file_that_is_not_compressed_raises(tmp_path):
    path = tmp_path / "splits.json.xz"
    path.write_text("{}")
    with pytest.raises(TimeNetConnectorError, match=r"splits\.json\.xz does not hold a valid split"):
        PrecomputedSplits[str].read_from_file(path)
