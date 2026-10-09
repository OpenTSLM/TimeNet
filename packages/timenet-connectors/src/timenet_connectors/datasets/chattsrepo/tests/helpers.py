"""Reusable fixture assertions for the ChatTS configuration connectors."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.dataset import TimeFDataset
from timenet.types import AnswerTask
from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


ROWS = [
    {
        "input": "Compare <ts><ts/> and <ts><ts/>.",
        "timeseries": [[1.0, 2.0, 3.0], [3.0, 2.0]],
        "output": "The first rises while the second falls.",
    }
]


def assert_configuration_conversion(connector: ChatTSConnector, configuration: str, tmp_path: Path) -> None:
    """Assert the shared conversion contract for one ChatTS configuration."""
    parquet_path = tmp_path / f"{configuration}.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), parquet_path)
    dataset = connector.convert([{"path": parquet_path, "__chatts_configuration": configuration}])
    assert isinstance(dataset, TimeFDataset)
    assert dataset.has_task_stream
    assert dataset.tasks == ()
    assert len(dataset.records) == 1
    (record,) = dataset.records
    assert record.metadata["chatts_configuration"] == configuration
    assert record.signals[0].spec.unit_value is None
    assert record.signals[0].to_arrow().type == pa.float64()
    assert [signal.to_numpy().tolist() for signal in record.signals] == [[1.0, 2.0, 3.0], [3.0, 2.0]]
    (task,) = tuple(dataset.iter_tasks())
    assert isinstance(task, AnswerTask)
    assert task.prompt == ROWS[0]["input"]
    assert task.targets == (ROWS[0]["output"],)
    assert task.metadata["chatts_configuration"] == configuration
    assert task.split is connector.TASK_SPLIT
    assert record.id == f"{configuration}-0"
    assert task.id == f"{configuration}-0-answer"
    assert task.inputs == (record,)
    assert task.targets is not None
