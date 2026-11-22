from collections import Counter
import csv
from datetime import datetime, timedelta
import gzip
import os
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split, TemporalLocalizationTask, TimeInterval
from timenet.types.text import read_value
from timenet_connectors.datasets.icsdataset.hai_loop_attacks.connector import (
    EXAMPLE_COLUMNS,
    EXAMPLES_COMMIT,
    EXAMPLES_DIR_ENV,
    EXAMPLES_FILENAME,
    LOOP_STATE,
    SOURCE_FILES,
    WINDOW_SECONDS,
    DataFile,
    HaiLoopAttacksConnector,
    HaiRelease,
    SourceFile,
    check_examples_table,
    examples_url,
    local_examples_table,
    read_examples,
)


_N_ROWS = 700
"""Rows of each synthetic data file: two windows fit, with room to spare."""
_START = {"20.07": datetime(2019, 10, 29, 11, 0, 0), "22.04": datetime(2021, 7, 10, 0, 0, 1)}
# Two versions cover both raw formats: 20.07 writes ";" and "time", 22.04 writes "," and
# "timestamp" with CRLF line ends. The column names differ between versions, like in HAI.
_FILES = (
    SourceFile("20.07", "test1", "test1.csv.gz", "0" * 64),
    SourceFile("22.04", "test1", "test1.csv", "0" * 64),
)
_COLUMNS = {
    "20.07": ("P1_B2016", "P1_PIT01", "P1_PCV01D", "P1_PCV02D", "P2_SIT01", "P1_FT01"),
    "22.04": ("P1_B2016", "P1_PIT01", "P1_PCV01D", "P1_PCV02D", "P2_SIT01", "P1_FT01"),
}
_FORMAT = {"20.07": (";", "time", "\n"), "22.04": (",", "timestamp", "\r\n")}


def _value(column: int, row: int) -> float:
    # Values with a few decimals, like HAI prints them. Each column and row gets its own value.
    return round(column * 100 + row * 0.25 + 0.003, 4)


def _write_data_file(
    directory: Path, source: SourceFile, *, n_rows: int = _N_ROWS, skip_row: int | None = None
) -> Path:
    delimiter, time_column, line_end = _FORMAT[source.version]
    columns = _COLUMNS[source.version]
    lines = [delimiter.join((time_column, *columns))]
    for row in range(n_rows):
        if row == skip_row:
            continue
        time = _START[source.version] + timedelta(seconds=row)
        cells = [f"{time:%Y-%m-%d %H:%M:%S}", *(repr(_value(column, row)) for column in range(len(columns)))]
        lines.append(delimiter.join(cells))
    text = line_end.join(lines) + line_end
    path = directory / source.path
    path.parent.mkdir(parents=True, exist_ok=True)
    if source.filename.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8", newline="")
    return path


def _example(  # ruff: ignore[too-many-arguments] - one argument per table column the tests vary
    version: str,
    *,
    offset: int,
    loop: str,
    split: str,
    label: str = "normal",
    attack: tuple[int, int] | None = None,
) -> dict[str, str]:
    start = _START[version] + timedelta(seconds=offset)
    window_id = f"{version}/test1/{start:%Y%m%dT%H%M%S}"
    if loop == "P1-PC":
        tags, roles = "P1_B2016 P1_PIT01 P1_PCV01D P1_PCV02D", "target measurement command command"
    else:
        # The 20.07 turbine speed loop has one series, like in HAI.
        tags, roles = "P2_SIT01", "measurement"
    # attack gives the window's attack. Only the attacked loop gets its seconds, like in the table.
    attack_start, attack_end = attack if attack and label == "attack" else (-1, -1)
    return {
        "example_id": f"{window_id}/{loop}",
        "window_id": window_id,
        "version": version,
        "file": "test1",
        "split": split,
        "window_kind": "attack" if attack else "normal",
        "anchor": "A101" if attack else "",
        "scored": str(attack is not None),
        "start": f"{start:%Y-%m-%d %H:%M:%S}",
        "loop": loop,
        "label": label,
        "attack_ids": "A101" if label == "attack" else "",
        "loop_uncertain": "False",
        "attack_start_s": str(attack_start),
        "attack_end_s": str(attack_end),
        "no_visible_change": "False",
        "reacting": "False",
        "train_use": "True",
        "tags": tags,
        "roles": roles,
        "source_tags": tags,
        "answer": f"Synthetic text for {loop}. Answer: {label}",
    }


_EXAMPLES = [
    # An attack whose flags outlast the window: its interval is cut at the window end.
    _example("20.07", offset=100, loop="P1-PC", split="train", label="attack", attack=(50, 400)),
    _example("20.07", offset=100, loop="P2-SC", split="train", attack=(50, 400)),
    _example("22.04", offset=10, loop="P1-PC", split="validation", label="attack", attack=(30, 99)),
    _example("22.04", offset=350, loop="P1-PC", split="test"),
]


def _write_release(directory: Path, examples: list[dict[str, str]] = _EXAMPLES, **file_options) -> HaiRelease:
    files = tuple(DataFile(source, _write_data_file(directory, source, **file_options)) for source in _FILES)
    table = directory / "examples.csv.gz"
    with gzip.open(table, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(examples)
    return HaiRelease(files=files, examples_path=table)


def test_each_file_becomes_a_record_with_one_signal_per_loop_series(tmp_path):
    dataset = HaiLoopAttacksConnector().convert(_write_release(tmp_path))
    assert len(dataset.records) == 2
    for record, source in zip(dataset.records, _FILES, strict=True):
        assert record.metadata["version"] == source.version
        assert record.metadata["file"] == "test1"
        assert record.metadata["start_time"] == f"{_START[source.version]:%Y-%m-%d %H:%M:%S}"
        assert record.metadata["n_seconds"] == _N_ROWS
        names = {signal.name for signal in record.signals}
        # Only the series that the version's examples use, never the other columns.
        expected = {"P1_B2016", "P1_PIT01", "P1_PCV01D", "P1_PCV02D"}
        assert names == (expected | {"P2_SIT01"} if source.version == "20.07" else expected)
        columns = _COLUMNS[source.version]
        for signal in record.signals:
            column = columns.index(signal.name)
            # Exact equality: float64 keeps every digit of the source text.
            np.testing.assert_array_equal(signal.to_numpy(), [_value(column, row) for row in range(_N_ROWS)])


def test_each_example_becomes_a_classification_task_over_its_window(tmp_path):
    dataset = HaiLoopAttacksConnector().convert(_write_release(tmp_path))
    tasks = dataset.tasks_of(ClassificationTask)
    assert [task.split for task in tasks] == [Split.TRAIN, Split.TRAIN, Split.VALIDATION, Split.TEST]
    for task, example in zip(tasks, _EXAMPLES, strict=True):
        (record,) = task.inputs
        assert record.metadata["version"] == example["version"]
        assert task.label_set == LOOP_STATE
        assert task.targets is not None
        assert read_value(task.targets[0]) == example["label"]
        assert read_value(task.rationale) == example["answer"]
        offset = int((datetime.fromisoformat(example["start"]) - _START[example["version"]]).total_seconds())
        scope = task.scope
        assert isinstance(scope, TimeInterval)
        assert (scope.start_us, scope.end_us) == (offset * 1_000_000, (offset + WINDOW_SECONDS) * 1_000_000)
        assert scope.time_series is not None
        assert [signal.name for signal in scope.time_series] == example["tags"].split()
        assert task.metadata["example_id"] == example["example_id"]
        assert task.metadata["roles"] == example["roles"].split()
        assert task.metadata["loop"] == example["loop"]
        assert task.metadata["scored"] == (example["scored"] == "True")
        assert task.metadata["train_use"] is True
        assert task.metadata["reacting"] is False


def test_each_attack_example_gets_a_localization_task_cut_at_the_window_end(tmp_path):
    dataset = HaiLoopAttacksConnector().convert(_write_release(tmp_path))
    attacks = [task for task in dataset.tasks_of(ClassificationTask) if task.metadata["attack_ids"]]
    locates = dataset.tasks_of(TemporalLocalizationTask)
    assert len(locates) == 2
    # 20.07: window at 100 s, attack from second 50 to 400 of the window: cut at second 300.
    # 22.04: window at 10 s, attack from second 30 to 99, inclusive: ends one second later.
    expected = [(150, 400, True), (40, 110, False)]
    for locate, classify, (start, end, cut) in zip(locates, attacks, expected, strict=True):
        assert locate.targets is not None
        (interval,) = locate.targets
        assert isinstance(interval, TimeInterval)
        assert (interval.start_us, interval.end_us) == (start * 1_000_000, end * 1_000_000)
        assert locate.split == classify.split
        assert locate.scope == classify.scope
        assert locate.from_tasks == (classify,)
        assert locate.metadata["cut_at_window_end"] is cut


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = HaiLoopAttacksConnector().convert(_write_release(source))
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())


def test_a_missing_second_raises(tmp_path):
    with pytest.raises(TimeNetConnectorError, match=r"hai-20\.07/test1\.csv\.gz: data rows 400 and 401 are 2 s apart"):
        HaiLoopAttacksConnector().convert(_write_release(tmp_path, skip_row=400))


def test_a_window_past_the_end_of_its_file_raises(tmp_path):
    late = _example("22.04", offset=_N_ROWS - WINDOW_SECONDS + 1, loop="P1-PC", split="test")
    with pytest.raises(TimeNetConnectorError, match="the window leaves its file of 700 rows"):
        HaiLoopAttacksConnector().convert(_write_release(tmp_path, [*_EXAMPLES, late]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label": "anomaly"}, "label 'anomaly' is not one of"),
        ({"loop": "P3-LC"}, "does not match its version, file, start and loop"),
        ({"roles": "target measurement command"}, "must name the same one or more series"),
        ({"roles": "target measurement command valve"}, r"roles \['valve'\] are not among"),
        ({"attack_start_s": "-5"}, "does not start inside the 300 s window"),
        ({"attack_ids": ""}, "names no attack or no attack interval"),
        ({"answer": "Answer: normal"}, "answer does not end with 'Answer: attack'"),
        ({"scored": "yes"}, "scored 'yes' is not True or False"),
        ({"reacting": "True", "train_use": "False"}, "an attack example has reacting set"),
        ({"no_visible_change": "True"}, "train_use is set on a no_visible_change or reacting example"),
    ],
    ids=[
        "an unknown label",
        "an id that does not match the loop",
        "fewer roles than tags",
        "an unknown role",
        "an attack before the window",
        "an attack without an id",
        "an answer for another label",
        "a flag that is not a bool",
        "a reacting attack",
        "a hidden attack in the training sample",
    ],
)
def test_a_malformed_example_raises_with_its_location(tmp_path, change, message):
    examples = [*_EXAMPLES[:2], _EXAMPLES[2] | change]
    with pytest.raises(TimeNetConnectorError, match=rf"examples\.csv\.gz data row 3: .*{message}"):
        HaiLoopAttacksConnector().convert(_write_release(tmp_path, examples))


def test_a_normal_example_with_an_attack_second_raises(tmp_path):
    examples = [_EXAMPLES[3] | {"attack_start_s": "40"}]
    with pytest.raises(TimeNetConnectorError, match="a normal example has attack_start_s 40, not -1"):
        HaiLoopAttacksConnector().convert(_write_release(tmp_path, examples))


def test_a_normal_example_with_no_visible_change_raises(tmp_path):
    examples = [_EXAMPLES[3] | {"no_visible_change": "True", "train_use": "False"}]
    with pytest.raises(TimeNetConnectorError, match="a normal example has no_visible_change set"):
        HaiLoopAttacksConnector().convert(_write_release(tmp_path, examples))


def test_the_table_url_is_pinned_by_commit():
    assert examples_url() == (
        "https://raw.githubusercontent.com/vid-t/hai-loop-attacks-data/"
        f"0e569b70a270908a86feba8bdc3b4d4f3bce0f4c/{EXAMPLES_FILENAME}"
    )
    # A full 40-character commit, not a branch or a short hash.
    assert len(EXAMPLES_COMMIT) == 40
    assert int(EXAMPLES_COMMIT, 16) >= 0
    url = examples_url("example-org", "example-repo", "0123abc")
    assert url == f"https://raw.githubusercontent.com/example-org/example-repo/0123abc/{EXAMPLES_FILENAME}"


@pytest.mark.parametrize(
    ("owner", "repo", "commit"),
    [
        ("<owner>", "example-repo", "0123abc"),
        ("example-org", "<repo>", "0123abc"),
        ("example-org", "example-repo", "<commit>"),
    ],
)
def test_the_table_url_refuses_placeholders(owner, repo, commit):
    with pytest.raises(TimeNetConnectorError, match=f"set {EXAMPLES_DIR_ENV}"):
        examples_url(owner, repo, commit)


def test_without_the_variable_no_local_table_is_used(monkeypatch):
    monkeypatch.delenv(EXAMPLES_DIR_ENV, raising=False)
    assert local_examples_table() is None


def test_a_local_folder_without_the_table_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(EXAMPLES_DIR_ENV, str(tmp_path))
    with pytest.raises(TimeNetConnectorError, match="does not exist"):
        local_examples_table()


def test_another_table_fails_the_pin(tmp_path, monkeypatch):
    with gzip.open(tmp_path / EXAMPLES_FILENAME, "wt", encoding="utf-8") as stream:
        stream.write(",".join(EXAMPLE_COLUMNS) + "\n")
    with pytest.raises(TimeNetConnectorError, match="not the pinned"):
        check_examples_table(tmp_path / EXAMPLES_FILENAME)
    monkeypatch.setenv(EXAMPLES_DIR_ENV, str(tmp_path))
    with pytest.raises(TimeNetConnectorError, match="not the pinned"):
        local_examples_table()


@pytest.mark.skipif(
    not os.environ.get(EXAMPLES_DIR_ENV), reason=f"{EXAMPLES_DIR_ENV} names no local copy of the examples table"
)
def test_the_pinned_table_holds_the_published_counts():
    table = local_examples_table()
    assert table is not None
    examples = read_examples(table)
    counts = Counter((example.split, example.label) for example in examples)
    assert counts == {
        (Split.TRAIN, "attack"): 822,
        (Split.TRAIN, "normal"): 17814,
        (Split.VALIDATION, "attack"): 144,
        (Split.VALIDATION, "normal"): 948,
        (Split.TEST, "attack"): 492,
        (Split.TEST, "normal"): 3912,
    }
    assert sum(example.no_visible_change for example in examples) == 98
    assert sum(example.reacting for example in examples) == 323
    assert sum(example.train_use for example in examples) == 3880
    # Every example names one of the pinned data files.
    pinned = {(source.version, source.stem) for source in SOURCE_FILES}
    assert {(example.version, example.file) for example in examples} <= pinned
