import json
import lzma
from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types.splits import Split
from timenet_connectors.splitting import PrecomputedSplitter


_SOURCE_SHA256 = "a" * 64


def _split_file_content(**splits: object) -> str:
    return json.dumps({"source_sha256": _SOURCE_SHA256} | splits)


def _write_split_file(path: Path, **splits: object) -> Path:
    content = _split_file_content(**splits).encode()
    path.write_bytes(lzma.compress(content) if path.suffix == ".xz" else content)
    return path


def test_samples_by_position_get_the_split_that_the_file_names(tmp_path):
    path = _write_split_file(tmp_path / "splits.json", train_keys=[0, 2, 4], validation_keys=[1], test_keys=[3])
    splitter = PrecomputedSplitter[str].from_file(path, source_sha256=_SOURCE_SHA256)
    assert dict(splitter.split(["a", "b", "c", "d", "e"])) == {
        Split.TRAIN: ("a", "c", "e"),
        Split.VALIDATION: ("b",),
        Split.TEST: ("d",),
    }


def test_samples_by_key_get_their_split_from_an_xz_file(tmp_path):
    path = _write_split_file(tmp_path / "splits.json.xz", train_keys=["D1"], validation_keys=["W1"], test_keys=["M3"])
    splitter = PrecomputedSplitter[tuple[str, int]].from_file(
        path, source_sha256=_SOURCE_SHA256, key=lambda sample: sample[0]
    )
    assert dict(splitter.split([("M3", 0), ("D1", 1), ("W1", 2)])) == {
        Split.TRAIN: (("D1", 1),),
        Split.VALIDATION: (("W1", 2),),
        Split.TEST: (("M3", 0),),
    }


def test_a_split_made_from_another_source_raises(tmp_path):
    path = _write_split_file(tmp_path / "splits.json", train_keys=[], validation_keys=[], test_keys=[])
    with pytest.raises(TimeNetConnectorError, match=f"made from the source {_SOURCE_SHA256}, not from the pinned 0+"):
        PrecomputedSplitter[str].from_file(path, source_sha256="0" * 64)


def test_a_split_that_names_a_key_the_source_does_not_have_raises(tmp_path):
    path = _write_split_file(tmp_path / "splits.json", train_keys=[0, 2, 3], validation_keys=[1], test_keys=[5])
    splitter = PrecomputedSplitter[str].from_file(path, source_sha256=_SOURCE_SHA256)
    with pytest.raises(TimeNetConnectorError, match=r"the split names \[5\] that the source does not have"):
        splitter.split(["a", "b", "c", "d"])


def test_a_sample_that_no_split_names_raises(tmp_path):
    path = _write_split_file(tmp_path / "splits.json", train_keys=[0, 2], validation_keys=[1], test_keys=[])
    splitter = PrecomputedSplitter[str].from_file(path, source_sha256=_SOURCE_SHA256)
    with pytest.raises(TimeNetConnectorError, match=r"the split does not name \[3\], which the source has"):
        splitter.split(["a", "b", "c", "d"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_split_file_content(train_keys=[1], validation_keys=[1], test_keys=[]), r"\[1\] are named more"),
        (_split_file_content(train_keys=[0], validation_keys=[-1], test_keys=[]), "greater than or equal to 0"),
        (_split_file_content(validation_keys=[], test_keys=[]), r"train_keys\n  Field required"),
        ("not JSON", "Invalid JSON"),
    ],
    ids=["a key in two splits", "a negative position", "a file without train keys", "a file that is not JSON"],
)
def test_a_malformed_split_file_raises_with_its_name(tmp_path, content, message):
    path = tmp_path / "splits.json"
    path.write_text(content)
    with pytest.raises(TimeNetConnectorError, match=rf"(?s)splits\.json does not hold a valid split: .*{message}"):
        PrecomputedSplitter[str].from_file(path, source_sha256=_SOURCE_SHA256)


def test_an_xz_file_that_is_not_compressed_raises(tmp_path):
    path = tmp_path / "splits.json.xz"
    path.write_text("{}")
    with pytest.raises(TimeNetConnectorError, match=r"splits\.json\.xz does not hold a valid split"):
        PrecomputedSplitter[str].from_file(path, source_sha256=_SOURCE_SHA256)
