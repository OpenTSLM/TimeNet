from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.testing import assert_datasets_equal
from timenet.types import (
    AnnotationType,
    ClassificationTask,
    LabelingTask,
    QATask,
    View,
)
from timenet_connectors.datasets.timenet.hello_world import HelloWorldConnector


def _convert() -> TimeFDataset:
    connector = HelloWorldConnector()
    return connector.convert(connector.download(Path("cache")))


def test_is_a_connector():
    assert isinstance(HelloWorldConnector(), BaseConnector)


def test_metadata():
    assert HelloWorldConnector().metadata().dataset_id == "timenet/hello-world"


def test_download_is_deterministic_and_offline():
    connector = HelloWorldConnector()
    assert connector.download(Path("cache-a")) == connector.download(Path("cache-b"))


def test_convert_is_deterministic():
    assert_datasets_equal(_convert(), _convert())


def test_schema_covers_every_feature():
    schema = _convert().derive_schema()
    assert len(schema.time_series_specs) == 2  # sine + cosine
    assert len(schema.data_sources) == 1
    annotation_types = {a.annotation_type for a in schema.annotations}
    assert annotation_types == {AnnotationType.STATIC, AnnotationType.POINT, AnnotationType.INTERVAL}
    assert set(schema.tasks) == {ClassificationTask, QATask, LabelingTask}


def test_has_window_sample():
    views = {s.view for s in _convert().samples}
    assert View.WINDOW in views


def test_shares_a_series_across_samples_by_id():
    dataset = _convert()
    series_by_id: dict[str, int] = {}
    for sample in dataset.samples:
        for ts in sample.time_series:
            series_by_id[ts.time_series_id] = series_by_id.get(ts.time_series_id, 0) + 1
    assert any(count >= 2 for count in series_by_id.values()), "expected a series shared across samples"


def test_shares_an_annotation_across_samples_by_id():
    dataset = _convert()
    ann_ids: dict[str, int] = {}
    for sample in dataset.samples:
        for ann in sample.annotations:
            ann_ids[ann.id] = ann_ids.get(ann.id, 0) + 1
    assert any(count >= 2 for count in ann_ids.values()), "expected an annotation shared across samples"


def test_has_a_task_chain():
    dataset = _convert()
    assert any(task.from_tasks for task in dataset.tasks), "expected a task derived via from_tasks"
