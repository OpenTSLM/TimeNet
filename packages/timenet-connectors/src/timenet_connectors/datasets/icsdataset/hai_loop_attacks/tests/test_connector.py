from collections.abc import Generator
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta
import gzip
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.engine import open_parents, store_dataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, ObjectKind, Split, TemporalLocalizationTask, TimeInterval, ureg
from timenet.types.reference_keys import ParentAlias
from timenet.types.text import read_value
from timenet_connectors.datasets.icsdataset.hai.connector import DataFile, HaiConnector, HaiRelease, LabelFile
from timenet_connectors.datasets.icsdataset.hai.files import SourceFile
from timenet_connectors.datasets.icsdataset.hai_loop_attacks.connector import (
    HaiLoopAttacksConnector,
    check_examples_table,
)
from timenet_connectors.datasets.icsdataset.hai_loop_attacks.examples import (
    EXAMPLE_COLUMNS,
    LOOP_STATE,
    WINDOW_SECONDS,
    read_examples,
)


_N_ROWS = 700
"""Rows of each synthetic data file: two windows fit, with room to spare."""
_START = {"20.07": datetime(2019, 10, 29, 11, 0, 0), "23.05": datetime(2022, 8, 12, 16, 0, 1)}
# The parent's files, with the real column names of their version. 20.07 names the water loop's series
# P3_LH, P3_LL and P3_LT01, which are P3_LH01, P3_LL01 and P3_LIT01 in 22.04. 23.05 names the boiler
# pressure target x1001_15_ASSIGN_OUT, which is P1_B2016 in 22.04. Each file has a column that no loop
# uses, so the tests see that the child imports only the loop series.
_FILES = (SourceFile("20.07", "test1.csv.gz", "0" * 64), SourceFile("23.05", "hai-test1.csv", "0" * 64))
_COLUMNS = {
    "20.07": (
        "P1_FT01",
        "P3_LCP01D",
        "P3_LCV01D",
        "P3_LH",
        "P3_LL",
        "P3_LT01",
        "attack",
        "attack_P1",
        "attack_P2",
        "attack_P3",
    ),
    "23.05": ("P1_FT01", "P1_PCV01D", "P1_PCV02D", "P1_PIT01", "x1001_15_ASSIGN_OUT"),
}
_LOOPS = {
    "P3-LC": {
        "tags": "P3_LH01 P3_LL01 P3_LIT01 P3_LCP01D P3_LCV01D",
        "source_tags": "P3_LH P3_LL P3_LT01 P3_LCP01D P3_LCV01D",
        "roles": "target target measurement command command",
        "matched_from": "name name manual same_name same_name",
    },
    "P1-PC": {
        "tags": "P1_B2016 P1_PIT01 P1_PCV01D P1_PCV02D",
        "source_tags": "x1001_15_ASSIGN_OUT P1_PIT01 P1_PCV01D P1_PCV02D",
        "roles": "target measurement command command",
        "matched_from": "manual same_name same_name same_name",
    },
}
_VERSION_LOOP = {"20.07": "P3-LC", "23.05": "P1-PC"}
_CLASSIFICATION_KEYS = {
    "example_id",
    "window_id",
    "version",
    "file",
    "loop",
    "tags",
    "source_tags",
    "roles",
    "tag_matched_from",
}


def _value(column: str, row: int) -> float:
    # Each column and row gets its own value, with a few decimals like HAI prints them.
    return round(sum(map(ord, column)) % 97 + row * 0.25 + 0.003, 4)


def _write_parent_files(directory: Path) -> HaiRelease:
    files = []
    for source in _FILES:
        delimiter, time_column, line_end = (
            (";", "time", "\n") if source.version == "20.07" else (",", "timestamp", "\r\n")
        )
        columns = _COLUMNS[source.version]
        lines = [delimiter.join((time_column, *columns))]
        for row in range(_N_ROWS):
            time = _START[source.version] + timedelta(seconds=row)
            cells = ["0" if c.startswith("attack") else repr(_value(c, row)) for c in columns]
            lines.append(delimiter.join([f"{time:%Y-%m-%d %H:%M:%S}", *cells]))
        path = directory / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = line_end.join(lines) + line_end
        if source.filename.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                stream.write(text)
        else:
            path.write_text(text, encoding="utf-8", newline="")
        files.append(DataFile(source, path))
    labels = directory / "hai-23.05" / "label-test1.csv"
    stamps = (f"{_START['23.05'] + timedelta(seconds=row):%Y-%m-%d %H:%M:%S},0" for row in range(_N_ROWS))
    labels.write_text("\r\n".join(["timestamp,label", *stamps]) + "\r\n", encoding="utf-8", newline="")
    label = LabelFile(SourceFile("23.05", "label-test1.csv", "0" * 64), labels)
    return HaiRelease(files=tuple(files), labels={"hai-test1": label})


def _example(  # ruff: ignore[too-many-arguments] - one argument per table column the tests vary
    version: str,
    *,
    offset: int,
    split: str,
    runs: str = "",
    scored: bool = False,
    hidden: bool = False,
    reacting: bool = False,
) -> dict[str, str]:
    start = _START[version] + timedelta(seconds=offset)
    loop = _VERSION_LOOP[version]
    window_id = f"{version}/{_FILES[list(_START).index(version)].stem}/{start:%Y%m%dT%H%M%S}"
    label = "attack" if runs else "normal"
    pairs = [part.split("-") for part in runs.split()]
    return {
        "example_id": f"{window_id}/{loop}",
        "window_id": window_id,
        "version": version,
        "file": window_id.split("/")[1],
        "split": split,
        "window_kind": "attack" if runs else "normal",
        "anchor": "A101" if runs else "",
        "scored": str(scored),
        "start": f"{start:%Y-%m-%d %H:%M:%S}",
        "loop": loop,
        "label": label,
        "attack_ids": "A101" if runs else "",
        "loop_uncertain": "False",
        "attack_start_s": pairs[0][0] if runs else "-1",
        "attack_end_s": pairs[-1][1] if runs else "-1",
        "attack_runs": runs,
        "no_visible_change": str(hidden),
        "reacting": str(reacting),
        "train_use": str(not hidden and not reacting),
        **_LOOPS[loop],
        "answer": "" if hidden else f"Synthetic text for {loop}. Answer: {label}",
    }


_EXAMPLES = [
    # The flag pulses: two runs, and the second outlasts the window, so its interval is cut at the end.
    _example("20.07", offset=100, split="train", runs="50-60 70-400", scored=True),
    _example("20.07", offset=350, split="train", reacting=True),
    # No series shows a change, so the example has no rationale.
    _example("23.05", offset=10, split="validation", runs="30-99", hidden=True),
    _example("23.05", offset=350, split="test"),
]


def _write_table(path: Path, examples: list[dict[str, str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(examples)
    return path


@contextmanager
def _composed(tmp_path: Path, examples: list[dict[str, str]] = _EXAMPLES) -> Generator[TimeFDataset]:
    """Build and store the HAI parent from the synthetic files, then compose the child inside the parent
    context, as the engine does."""
    root = tmp_path / "registry"
    store_dataset(HaiConnector().convert(_write_parent_files(tmp_path / "hai")), root)
    connector = HaiLoopAttacksConnector()
    table = _write_table(tmp_path / "examples.csv.gz", examples)
    with open_parents(connector.metadata(), LocalRegistry(root)) as context:
        yield connector.compose({"examples": table}, context)


def test_each_file_is_a_copy_of_its_parent_record_with_only_the_loop_series(tmp_path):
    with _composed(tmp_path) as dataset:
        assert [record.metadata["path"] for record in dataset.records] == [one.path for one in _FILES]
        copies = {
            child_id
            for (alias, kind, _), child_id in dataset._import_state.import_map.items()
            if alias == ParentAlias("hai") and kind is ObjectKind.RECORD
        }
        assert copies == {record.id for record in dataset.records}
        old, new = ({signal.name: signal for signal in record.signals} for record in dataset.records)
        assert set(old) == set(_LOOPS["P3-LC"]["source_tags"].split())
        assert set(new) == set(_LOOPS["P1-PC"]["source_tags"].split())
        # The columns keep their name in the version, and their metadata gives the 22.04 tag.
        assert old["P3_LT01"].metadata == {
            "manual_point": "P3_LIT01",
            "tag": "P3_LIT01",
            "loop": "P3-LC",
            "role": "measurement",
            "tag_matched_from": "manual",
        }
        assert old["P3_LH"].metadata["tag"] == "P3_LH01"
        assert old["P3_LH"].metadata["tag_matched_from"] == "name"
        assert new["x1001_15_ASSIGN_OUT"].metadata["tag"] == "P1_B2016"
        assert new["x1001_15_ASSIGN_OUT"].metadata["role"] == "target"
        # The specs and the values come from the parent.
        assert old["P3_LT01"].spec.unit_value == ureg.percent
        assert new["x1001_15_ASSIGN_OUT"].spec.unit_value == ureg.bar
        np.testing.assert_array_equal(old["P3_LT01"].to_numpy(), [_value("P3_LT01", row) for row in range(_N_ROWS)])


def test_each_example_is_a_classification_task_that_gives_nothing_away(tmp_path):
    with _composed(tmp_path) as dataset:
        tasks = dataset.tasks_of(ClassificationTask)
        assert [task.split for task in tasks] == [Split.TRAIN, Split.TRAIN, Split.VALIDATION, Split.TEST]
        for task, example in zip(tasks, _EXAMPLES, strict=True):
            assert task.label_set == LOOP_STATE
            assert task.targets is not None
            assert read_value(task.targets[0]) == example["label"]
            offset = int((datetime.fromisoformat(example["start"]) - _START[example["version"]]).total_seconds())
            scope = task.scope
            assert isinstance(scope, TimeInterval)
            assert (scope.start_us, scope.end_us) == (offset * 1_000_000, (offset + WINDOW_SECONDS) * 1_000_000)
            assert scope.time_series is not None
            assert [signal.name for signal in scope.time_series] == example["source_tags"].split()
            assert [signal.metadata["tag"] for signal in scope.time_series] == example["tags"].split()
            # Only what names the example and its series: no attack, window kind or label-only flag.
            assert set(task.metadata) == _CLASSIFICATION_KEYS
            assert task.metadata["tag_matched_from"] == example["matched_from"].split()
        assert read_value(tasks[0].rationale) == _EXAMPLES[0]["answer"]
        # No series of the hidden attack moves, so it has no rationale.
        assert tasks[2].rationale is None


def test_each_attack_example_gets_one_interval_per_flagged_run(tmp_path):
    with _composed(tmp_path) as dataset:
        attacks = [task for task in dataset.tasks_of(ClassificationTask) if task.targets == ("attack",)]
        locates = dataset.tasks_of(TemporalLocalizationTask)
        # 20.07: window at 100 s, runs 50-60 and 70-400 of the window: the second is cut at second 300.
        # 23.05: window at 10 s, one run 30-99, inclusive: it ends one second later.
        expected = [([(150, 161), (170, 400)], True), ([(40, 110)], False)]
        assert len(locates) == len(expected)
        for locate, classify, (spans, cut) in zip(locates, attacks, expected, strict=True):
            assert locate.targets is not None
            intervals = [span for span in locate.targets if isinstance(span, TimeInterval)]
            assert [(span.start_us // 1_000_000, span.end_us // 1_000_000) for span in intervals] == spans
            assert len(intervals) == len(locate.targets)
            assert locate.split == classify.split
            assert locate.scope == classify.scope
            assert locate.from_tasks == (classify,)
            assert locate.metadata["cut_at_window_end"] is cut
            assert locate.metadata["attack_ids"] == ["A101"]
        assert locates[0].metadata["scored"] is True
        assert locates[1].metadata["no_visible_change"] is True


def test_the_dataset_lists_the_examples_of_the_protocol(tmp_path):
    with _composed(tmp_path) as dataset:
        lists = {annotation.key: read_value(annotation.value) for annotation in dataset.annotations}
        assert lists["training_sample"] == [_EXAMPLES[0]["example_id"], _EXAMPLES[3]["example_id"]]
        assert lists["scored_windows"] == [_EXAMPLES[0]["window_id"]]
        assert lists["reacting_examples"] == [_EXAMPLES[1]["example_id"]]


def test_the_dataset_round_trips_through_storage(tmp_path):
    with _composed(tmp_path) as dataset:
        version_dir = store_dataset(dataset, tmp_path / "registry")
        with TimeFReader(open_local_version(version_dir)) as reader:
            reader.verify()
            assert_datasets_equal(dataset, reader.read())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"loop": "P3-LC"}, "does not match its version, file, start and loop"),
        ({"attack_runs": ""}, "names no attack or no flagged run"),
    ],
    ids=["an id that does not match the loop", "an attack without a flagged run"],
)
def test_a_malformed_example_raises_with_its_location(tmp_path, change, message):
    examples = [*_EXAMPLES[:2], _EXAMPLES[2] | change]
    with pytest.raises(TimeNetConnectorError, match=rf"examples\.csv\.gz data row 3: .*{message}"):
        read_examples(_write_table(tmp_path / "examples.csv.gz", examples))


def test_a_window_past_the_end_of_its_file_raises(tmp_path):
    late = _example("23.05", offset=_N_ROWS - WINDOW_SECONDS + 1, split="test")
    with (
        pytest.raises(TimeNetConnectorError, match="the window leaves its file of 700 rows"),
        _composed(tmp_path, [*_EXAMPLES, late]),
    ):
        pass


def test_another_table_fails_the_pin(tmp_path):
    path = _write_table(tmp_path / "examples.csv.gz", _EXAMPLES)
    with pytest.raises(TimeNetDownloadError, match="not the pinned"):
        check_examples_table(path)
