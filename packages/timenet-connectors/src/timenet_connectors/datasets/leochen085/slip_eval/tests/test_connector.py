"""The evaluation connector, against synthetic folders written into ``tmp_path``."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.errors import TimeFFormatError
from timenet.types import ClassificationTask, Task
from timenet_connectors.datasets.leochen085.slip_eval.connector import (
    SlipEvalConnector,
    SlipEvalSource,
    _iter_tasks,
    _iter_windows,
    fingerprint,
)
from timenet_connectors.datasets.leochen085.slip_eval.tasks import label_id


def _schema_of(task: Task) -> str:
    # iter_tasks gives Task; only ClassificationTask carries target_schema.
    assert isinstance(task, ClassificationTask)
    assert task.target_schema is not None
    return task.target_schema


Row = tuple[list[list[float]], str, str | None]  # the window, its label, its participant or None


def _write(root: Path, folder: str, split: str, rows: list[Row], part: int = 0) -> None:
    # Invented, not a slice of the release. A reader can tell by the window length: the shortest
    # folder the release ships holds 200 values per window and the longest 3,000, where every window
    # written here holds two.
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[object]] = {
        "X": [window for window, _, _ in rows],
        "label": [label for _, label, _ in rows],
        "text_label": [label for _, label, _ in rows],
        "prompt": [f"The subject is $label. ({folder})"] * len(rows),
    }
    if any(participant is not None for _, _, participant in rows):
        columns["participant_id"] = [participant for _, _, participant in rows]
    pq.write_table(pa.table(columns), directory / f"{split}-{part:05d}-of-00001.parquet")


def _wisdm(root: Path) -> None:
    _write(
        root,
        "wisdm",
        "train",
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "sitting", "1633"),
            ([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], "teeth", "1638"),
        ],
    )
    _write(root, "wisdm", "test", [([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]], "sitting", "1633")])


def test_one_record_per_window_named_for_its_folder_split_and_row(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    records = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)]).records
    assert [r.record_id for r in records] == [
        "slip-eval-wisdm-train-000000",
        "slip-eval-wisdm-train-000001",
        "slip-eval-wisdm-test-000000",
    ]


def test_a_participant_becomes_a_subject(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    records = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)]).records
    # Three folders number their subjects independently, so a bare id would merge two people.
    assert records[0].subject_ids == ("wisdm-1633",)


def test_a_folder_with_no_participant_column_carries_no_subject(tmp_path: Path) -> None:
    _write(tmp_path, "Beijing_AQI", "train", [([[1.0, 2.0]] * 7, "Good", None)])
    records = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)]).records
    assert records[0].subject_ids == ()


def test_the_prompt_is_kept_verbatim_however_wrong(tmp_path: Path) -> None:
    _write(tmp_path, "AsphaltObstacles", "train", [([[1.0, 2.0]], "speed_bump", None)])
    tasks = list(_iter_tasks(tmp_path, {("AsphaltObstacles", "train", 0): "r"}))
    assert tasks[0].prompt == "The subject is $label. (AsphaltObstacles)"


def test_the_answer_is_a_reference_and_not_an_inline_copy(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    task = next(iter(dataset.iter_tasks()))
    assert task.target is None
    assert task.target_annotation_ids == (label_id("wisdm", "sitting"),)


def test_the_same_label_is_registered_once_and_shared(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    tasks = list(dataset.iter_tasks())
    sitting = [t for t in tasks if t.target_annotation_ids == (label_id("wisdm", "sitting"),)]
    assert len(sitting) == 2  # one in train, one in test, both pointing at the one annotation


def test_the_folder_and_the_split_ride_on_the_task(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    task = next(iter(dataset.iter_tasks()))
    assert task.input_annotation_ids == ("benchmark-wisdm", "split-train")


def test_the_target_schema_is_the_id_of_the_annotation_holding_the_class_set(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    task = next(iter(dataset.iter_tasks()))
    assert _schema_of(task) == "vocabulary-wisdm"
    vocabulary = next(a for a in dataset.registered_annotations if a.id == "vocabulary-wisdm")
    assert vocabulary.value == ["sitting", "teeth"]


def test_a_label_id_is_the_same_in_every_build(tmp_path: Path) -> None:
    # The builtin hash is salted per interpreter; a digest over the text is not.
    assert label_id("wisdm", "sitting") == "wisdm-label-f87506342d69d829"


def test_a_split_of_several_files_numbers_rows_across_the_split(tmp_path: Path) -> None:
    _write(tmp_path, "sleepEDF", "train", [([[1.0], [2.0]], "W", None), ([[3.0], [4.0]], "N1", None)], part=0)
    _write(tmp_path, "sleepEDF", "train", [([[5.0], [6.0]], "N2", None)], part=1)
    windows = list(_iter_windows(tmp_path, "sleepEDF"))
    assert [w.index for w in windows] == [0, 1, 2]
    assert [w.row_in_file for w in windows] == [0, 1, 0]


def test_the_three_ppg_folders_share_one_record_per_distinct_window(tmp_path: Path) -> None:
    a, b = [[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]
    _write(tmp_path, "PPG_CVA", "train", [(a, "stroke", None), (b, "no stroke", None)])
    _write(tmp_path, "PPG_DM", "train", [(a, "diabetes", None), (b, "no diabetes", None)])
    _write(tmp_path, "PPG_HTN", "train", [(a, "normal", None), (b, "stage 1", None)])
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    assert [r.record_id for r in dataset.records] == [
        "slip-eval-PPG_CVA-train-000000",
        "slip-eval-PPG_CVA-train-000001",
    ]
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == 6
    first = dataset.records[0].record_id
    on_first = [_schema_of(t) for t in tasks if t.record_ids == (first,)]
    assert sorted(on_first) == ["vocabulary-PPG_CVA", "vocabulary-PPG_DM", "vocabulary-PPG_HTN"]


def test_a_window_duplicated_inside_one_folder_keeps_both_tasks(tmp_path: Path) -> None:
    # Seven windows appear twice in each real PPG folder, one pair across its own split.
    window = [[1.0, 2.0]]
    _write(tmp_path, "PPG_CVA", "train", [(window, "stroke", None), (window, "stroke", None)])
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    assert [r.record_id for r in dataset.records] == ["slip-eval-PPG_CVA-train-000000"]
    tasks = list(dataset.iter_tasks())
    # Both rows keep a task on the one record: the release ships the window twice.
    assert len(tasks) == 2
    assert {t.record_ids for t in tasks} == {("slip-eval-PPG_CVA-train-000000",)}


def test_the_task_stream_gives_the_same_tasks_when_read_again(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    dataset = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])
    once = [(t.record_ids, t.target_annotation_ids) for t in dataset.iter_tasks()]
    again = [(t.record_ids, t.target_annotation_ids) for t in dataset.iter_tasks()]
    assert once == again
    assert len(once) == 3


def test_the_fingerprint_is_a_value_over_arrays_and_needs_no_file() -> None:
    a = [np.array([1.0, 2.0]), np.array([3.0])]
    b = [np.array([1.0, 2.0]), np.array([3.0])]
    c = [np.array([1.0, 2.0]), np.array([4.0])]
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)
    # Signal order is part of the window, so two orderings are two windows.
    assert fingerprint(a) != fingerprint(list(reversed(a)))


def test_a_window_whose_signals_differ_in_length_raises(tmp_path: Path) -> None:
    directory = tmp_path / "wisdm"
    directory.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "X": [[[1.0, 2.0], [3.0], [4.0, 5.0]]],
                "label": ["sitting"],
                "text_label": ["sitting"],
                "prompt": ["The subject is $label."],
            }
        ),
        directory / "train-00000-of-00001.parquet",
    )
    with pytest.raises(TimeFFormatError, match="same span"):
        list(_iter_windows(tmp_path, "wisdm"))


def test_a_folder_shipping_an_unexpected_signal_count_raises(tmp_path: Path) -> None:
    # wisdm's card names three signals; a release that ships four has changed shape.
    _write(tmp_path, "wisdm", "train", [([[1.0], [2.0], [3.0], [4.0]], "sitting", None)])
    with pytest.raises(TimeFFormatError, match="the card names 3"):
        SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)])


def test_the_values_read_back_in_the_dtype_the_spec_declares(tmp_path: Path) -> None:
    # The folders store double. A spec left at the default float32 makes every loader raise, and
    # only when the writer calls it — long after convert has returned.
    _wisdm(tmp_path)
    records = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)]).records
    series = records[0].time_series[0]
    assert series.spec.dtype == "float64"
    assert series.to_numpy().tolist() == [1.0, 2.0]


def test_every_signal_of_every_record_loads(tmp_path: Path) -> None:
    _wisdm(tmp_path)
    _write(tmp_path, "Beijing_AQI", "train", [([[1.0, 2.0]] * 7, "Good", None)])
    records = SlipEvalConnector().convert([SlipEvalSource(root=tmp_path)]).records
    read = [len(s.to_numpy()) for r in records for s in r.time_series]
    assert read == [2, 2, 2, 2, 2, 2, 2, 2, 2, *([2] * 7)]
