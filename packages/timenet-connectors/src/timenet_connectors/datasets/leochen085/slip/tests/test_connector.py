"""Check that SLIP corpus rows and collection windows keep their captions, labels, facts, values, and splits."""

from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.types import ClassificationTask, InputModality
from timenet_connectors.datasets.leochen085.slip.connector import (
    _EVALUATION_RATES_HZ,
    COLLECTIONS,
    SlipConnector,
    SlipSource,
)


# The pinned release's meta.csv, unchanged, so the test reads the same corpus rows a build does.
META_CSV = Path(__file__).parent / "fixtures" / "meta.csv"


def test_every_evaluation_collection_has_a_stated_rate():
    assert set(_EVALUATION_RATES_HZ) == set(COLLECTIONS)


def _write_corpus_shard(path: Path) -> Path:
    path.parent.mkdir(exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "category": ["IoT", "Health"],
                "dataset": ["Capture24", "PigCVP"],
                "caption0": ["Steady.", "Rising."],
                "caption1": ["Nearly flat.", "Rising."],
                "caption2": ["No large changes.", "Climbs."],
                "caption3": ["Little variation.", "Goes up."],
                "time_series": pa.array(
                    [[[1.0, 2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], type=pa.list_(pa.list_(pa.float32()))
                ),
            }
        ),
        path,
    )
    return path


def _write_window_shard(path: Path, *, label: pa.Array, text_label: str, participant: pa.Array | None) -> Path:
    columns: dict[str, object] = {
        "X": pa.array([[[1.0, 2.0], [3.0, 4.0]]], type=pa.list_(pa.list_(pa.float64()))),
        "label": label,
        "text_label": [text_label],
        "prompt": ["The subject is $label."],
    }
    if participant is not None:
        columns["participant_id"] = participant
    path.parent.mkdir(exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def _facts(record: Record) -> dict[str, object]:
    return {annotation.key: annotation.value for annotation in record.annotations if "caption" not in annotation.key}


def _values(dataset: TimeFDataset, key: str) -> set[object]:
    return {
        annotation.value for record in dataset.records for annotation in record.annotations if annotation.key == key
    }


def test_corpus_rows_become_training_records_with_corpus_facts_and_caption_tasks(tmp_path):
    shard = _write_corpus_shard(tmp_path / "data" / "train-00000-of-00001.parquet")

    dataset = SlipConnector().convert([SlipSource(corpus=(shard,), collections=(), meta_csv=META_CSV)])
    tasks = dataset.get_train()
    first, second = dataset.records

    assert [signal.id for signal in second.signals] == ["slip-data-train-000001-s00", "slip-data-train-000001-s01"]
    assert all(isinstance(signal.time_axis, OrdinalAxis) for record in dataset.records for signal in record.signals)
    assert first.signals[0].to_arrow().to_pylist() == [1.0, 2.0, 3.0]
    assert second.signals[1].to_arrow().to_pylist() == [6.0, 7.0]
    # Capture24 states a rate and no source; PigCVP states a source and no rate.
    assert _facts(first) == {"source_dataset": "Capture24", "domain": "IoT", "source_rate": "30 Hz"}
    assert _facts(second) == {
        "source_dataset": "PigCVP",
        "domain": "Health",
        "source_url": "https://arxiv.org/abs/1810.07758",
    }
    # The second row repeats caption0 in caption1, so it gets three tasks rather than four.
    assert [task.id for task in tasks] == [
        *(f"slip-data-train-000000-caption{index}-task" for index in range(4)),
        *(f"slip-data-train-000001-caption{index}-task" for index in (0, 2, 3)),
    ]
    # Every corpus task is in the train split, and the stream yields them again on each read.
    assert [task.id for task in dataset.get_all()] == [task.id for task in tasks]
    assert dataset.get_test() == ()
    assert all(task.input_modalities == frozenset({InputModality.TIME_SERIES}) for task in tasks)


def test_windows_become_records_of_their_subset_and_the_release_derives_one_schema(tmp_path):
    corpus = _write_corpus_shard(tmp_path / "data" / "train-00000-of-00001.parquet")
    # The release types these columns differently per collection: a label is an int, a float, or the
    # text label itself, and a participant id is text or a number.
    windows = (
        _write_window_shard(
            tmp_path / "sleepEDF" / "train-00000-of-00002.parquet",
            label=pa.array([2], pa.int64()),
            text_label="deep slow wave sleep",
            participant=pa.array(["SC4001E0"]),
        ),
        _write_window_shard(
            tmp_path / "sleepEDF" / "train-00001-of-00002.parquet",
            label=pa.array([0], pa.int64()),
            text_label="wakefulness with full awareness",
            participant=pa.array(["SC4012E0"]),
        ),
        _write_window_shard(
            tmp_path / "studentlife" / "test-00000-of-00001.parquet",
            label=pa.array([1.0], pa.float64()),
            text_label="medium stress",
            participant=pa.array(["4_3_28_2"]),
        ),
        _write_window_shard(
            tmp_path / "wisdm" / "test-00000-of-00001.parquet",
            label=pa.array(["walking"]),
            text_label="walking",
            participant=pa.array([1600], pa.int64()),
        ),
        _write_window_shard(
            tmp_path / "PPG_DM" / "train-00000-of-00001.parquet",
            label=pa.array([0], pa.int64()),
            text_label="normal",
            participant=None,
        ),
    )

    source = SlipSource(corpus=(corpus,), collections=tuple(sorted(windows)), meta_csv=META_CSV)
    dataset = SlipConnector().convert([source])
    schema = dataset.derive_schema()
    classified = [task for task in dataset.get_all() if isinstance(task, ClassificationTask)]

    # A subset's rows number on across its shards.
    assert [record.id for record in dataset.records] == [
        "slip-data-train-000000",
        "slip-data-train-000001",
        "slip-PPG_DM-train-000000",
        "slip-sleepEDF-train-000000",
        "slip-sleepEDF-train-000001",
        "slip-studentlife-test-000000",
        "slip-wisdm-test-000000",
    ]
    # A train subset joins the corpus captions in the train split; a test subset is the test split.
    assert [task.id for task in dataset.get_train()] == [
        *(task.id for task in dataset.get_all() if not isinstance(task, ClassificationTask)),
        "slip-PPG_DM-train-000000-classification",
        "slip-sleepEDF-train-000000-classification",
        "slip-sleepEDF-train-000001-classification",
    ]
    assert [task.id for task in dataset.get_test()] == [
        "slip-studentlife-test-000000-classification",
        "slip-wisdm-test-000000-classification",
    ]
    assert len(classified) == len(windows)
    assert {task.targets for task in classified} == {
        ("deep slow wave sleep",),
        ("wakefulness with full awareness",),
        ("medium stress",),
        ("walking",),
        ("normal",),
    }
    assert all(task.prompt is None and task.input_modalities == {InputModality.TIME_SERIES} for task in classified)
    # One annotation key has one type across the corpus and every collection.
    assert {descriptor.key: descriptor.value_type for descriptor in schema.annotations}["participant_id"] == "str"
    assert _values(dataset, "participant_id") == {"SC4001E0", "SC4012E0", "4_3_28_2", "1600"}
    assert _values(dataset, "source_dataset") == {"Capture24", "PigCVP", "PPG_DM", "sleepEDF", "studentlife", "wisdm"}
    assert _values(dataset, "label") == {2, 1, 0}
    assert _values(dataset, "label_template") == {"The subject is $label."}
    assert {descriptor.key for descriptor in schema.annotations}.isdisjoint({"text_label", "split", "component"})
    assert dataset.records[2].signals[0].time_axis == RegularAxis.from_rate_hz(65)
    assert dataset.records[3].signals[0].time_axis == RegularAxis.from_rate_hz(100)
    assert dataset.records[5].signals[0].time_axis == RegularAxis.from_rate_hz(Fraction(1, 60))
    assert dataset.records[6].signals[0].time_axis == RegularAxis.from_rate_hz(30)
    assert dataset.records[2].signals[1].to_arrow().to_pylist() == [3.0, 4.0]
