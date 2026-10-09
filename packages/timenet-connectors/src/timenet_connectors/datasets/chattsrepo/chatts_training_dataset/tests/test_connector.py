"""Tests for ChatTS composition and configuration-specific downloads."""

from collections.abc import Iterable, Iterator
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.composition import BuildContext, ParentDatasetView
from timenet.dataset import Record, TimeFDataset
from timenet.types import Task
from timenet_connectors.datasets.chattsrepo.align_256 import ChatTSAlign256Connector
from timenet_connectors.datasets.chattsrepo.align_random import ChatTSAlignRandomConnector
from timenet_connectors.datasets.chattsrepo.chatts_training_dataset import ChatTSTrainingDatasetConnector
from timenet_connectors.datasets.chattsrepo.dev import ChatTSDevConnector
from timenet_connectors.datasets.chattsrepo.ift import ChatTSIftConnector
from timenet_connectors.datasets.chattsrepo.sft import ChatTSSftConnector


_ROWS = [{"input": "q", "timeseries": [[1.0, 2.0]], "output": "a"}]


class _Parent:
    """Minimal parent view used to exercise connector composition without storage I/O."""

    def __init__(self, dataset: TimeFDataset) -> None:
        """Store the converted configuration dataset."""
        self.dataset = dataset

    def iter_records(self, record_ids: Iterable[str] | None = None) -> Iterator[Record]:
        """Yield records, optionally selected by id."""
        selected = None if record_ids is None else set(record_ids)
        yield from (record for record in self.dataset.records if selected is None or record.id in selected)

    def iter_tasks(self, records: Iterable[Record] | None = None) -> Iterator[Task]:
        """Yield every task belonging to the configuration."""
        yield from self.dataset.tasks


def test_complete_connector_imports_each_configuration():
    """Compose the complete dataset from each configuration's records and tasks."""
    connector = ChatTSTrainingDatasetConnector()
    parents = {
        parent.metadata().dataset_id: _Parent(parent.convert(_ROWS))
        for parent in (
            ChatTSAlign256Connector(),
            ChatTSAlignRandomConnector(),
            ChatTSSftConnector(),
            ChatTSIftConnector(),
            ChatTSDevConnector(),
        )
    }
    context = BuildContext(
        connector.metadata(),
        cast("dict[str, ParentDatasetView]", parents),
    )
    dataset = connector.compose([], context)

    assert len(dataset.records) == len(parents)
    assert {task.metadata["chatts_configuration"] for task in dataset.tasks} == {
        "align_256",
        "align_random",
        "sft",
        "ift",
        "dev",
    }
    assert {record_import.parent_dataset_id for record_import in dataset.record_imports.values()} == set(parents)


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
