"""Check the SLIP evaluation split, label, and sensor values."""

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.types import InputModality
from timenet_connectors.datasets.leochen085.slip_eval.connector import EvalSource, SlipEvalConnector


def test_eval_train_and_test_rows_keep_native_labels(tmp_path):
    folder = tmp_path / "uci_har"
    folder.mkdir()
    shards = []
    for split in ("train", "test"):
        shard = folder / f"{split}-00000-of-00001.parquet"
        pq.write_table(
            pa.table(
                {
                    "X": pa.array([[[1.0, 2.0], [3.0, 4.0]]], type=pa.list_(pa.list_(pa.float64()))),
                    "label": pa.array([2], type=pa.int64()),
                    "text_label": ["walking"],
                    "prompt": ["Identify the activity."],
                    "participant_id": ["person-1"],
                }
            ),
            shard,
        )
        shards.append(shard)

    dataset = SlipEvalConnector().convert([EvalSource(shards=tuple(shards), root=tmp_path)])

    assert len(dataset.records) == len(dataset.tasks) == 2
    assert {task.targets for task in dataset.tasks} == {(2,)}
    assert all(
        task.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES}) for task in dataset.tasks
    )
    assert {
        annotation.value for record in dataset.records for annotation in record.annotations if annotation.key == "split"
    } == {"train", "test"}
    assert dataset.records[0].signals[0].to_arrow().to_pylist() == [1.0, 2.0]
