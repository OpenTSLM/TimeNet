"""Tests for ChatTS aggregation and configuration-specific downloads."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet_connectors.datasets.chattsrepo.align_256 import ChatTSAlign256Connector
from timenet_connectors.datasets.chattsrepo.chatts_training_dataset import ChatTSTrainingDatasetConnector


_ROWS = [{"input": "q", "timeseries": [[1.0, 2.0]], "output": "a"}]


def test_complete_connector_keeps_each_configuration_metadata():
    """Build the complete dataset from rows tagged with their source configuration."""
    connector = ChatTSTrainingDatasetConnector()
    dataset = connector.convert(
        [{**_ROWS[0], "__chatts_configuration": configuration} for configuration in connector.HF_CONFIGS]
    )
    assert len(dataset.records) == len(connector.HF_CONFIGS)
    assert {task.metadata["chatts_configuration"] for task in dataset.tasks} == {
        "align_256",
        "align_random",
        "sft",
        "ift",
        "dev",
    }


def test_configuration_download_uses_only_its_parquet_prefix(monkeypatch, tmp_path):
    """Select only the requested Hub configuration rather than all ChatTS rows."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    parquet_path = tmp_path / "align_256" / "train" / "0000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(_ROWS), parquet_path)
    seen: dict[str, object] = {}

    def fake_list(repo_id, repo_type=None, revision=None):
        seen["revision"] = revision
        return ["align_256/train/0000.parquet", "sft/train/0000.parquet"]

    def fake_download(repo_id, filename, repo_type=None, revision=None, cache_dir=None):
        seen["filename"] = filename
        return str(parquet_path)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert ChatTSAlign256Connector().download(tmp_path / "cache") == [
        {**_ROWS[0], "__chatts_configuration": "align_256"}
    ]
    assert seen == {"revision": "refs/convert/parquet", "filename": "align_256/train/0000.parquet"}
