"""Check that SLIP training captions remain attached to their source window."""

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.types import InputModality
from timenet_connectors.datasets.leochen085.slip_train.connector import SlipSource, SlipTrainConnector


def test_one_training_row_produces_four_stable_caption_tasks(tmp_path):
    shard = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(
        pa.table(
            {
                "category": ["Health"],
                "dataset": ["demo"],
                "caption0": ["Steady."],
                "caption1": ["Nearly flat."],
                "caption2": ["No large changes."],
                "caption3": ["Little variation."],
                "time_series": pa.array([[[1.0, 2.0, 3.0]]], type=pa.list_(pa.list_(pa.float32()))),
            }
        ),
        shard,
    )
    meta = tmp_path / "meta.csv"
    meta.write_text("Dataset,Domain,Source,Freq\ndemo,Health,https://example.com,1 Hz\n", encoding="utf-8")

    dataset = SlipTrainConnector().convert([SlipSource(shards=(shard,), meta_csv=meta)])
    tasks = list(dataset.iter_tasks())

    assert len(dataset.records) == 1
    assert [task.id for task in tasks] == [f"slip-shard-00000-row-000000-caption{index}-task" for index in range(4)]
    assert [task.id for task in dataset.iter_tasks()] == [task.id for task in tasks]
    assert all(task.input_modalities == frozenset({InputModality.TIME_SERIES}) for task in tasks)
    assert dataset.records[0].signals[0].to_arrow().to_pylist() == [1.0, 2.0, 3.0]
