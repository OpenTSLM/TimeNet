"""Check that the plots dataset composes on its ``datadog/arfbench`` parent.

The dataset imports the series tasks. It adds a chart record per chart and a chart task per question. It stores
only the charts. It reads the series through the Parquet release of the parent. A question without its charts
raises an error."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from timenet.dataset import TimeFDataset
from timenet.engine import open_parents, store_dataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import ClassificationTask, InputModality, Split, read_values
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.datadog.arfbench.connector import ARFBenchConnector
from timenet_connectors.datasets.datadog.arfbench.tests.release import (
    CORRELATION_ROW,
    ROWS,
    facts,
    write_qa_csv,
    write_release,
)
from timenet_connectors.datasets.datadog.arfbench_plots.connector import CHART_DIR, ARFBenchPlotsConnector


def _write_chart_dir(root: Path) -> Path:
    chart_dir = root / CHART_DIR
    chart_dir.mkdir()
    for metric in ("35997_0", "35997_1"):
        Image.new("RGBA", (3, 2), (10, 20, 30, 42)).save(chart_dir / f"{metric}.png")
    Image.new("RGB", (6, 2), (255, 255, 255)).save(chart_dir / "35997_0-35997_1.png")
    return chart_dir


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """A local registry that holds the ``datadog/arfbench`` parent, built from the test release."""
    root = tmp_path / "registry"
    release = tmp_path / "series"
    release.mkdir()
    parent = ARFBenchConnector()
    store_dataset(parent.convert(write_release(release)), root, values_backend=parent.values_backend)
    return root


@pytest.fixture
def charts(tmp_path: Path) -> list[HubFileRef]:
    """The question table and the three charts of the test release."""
    root = tmp_path / "plots"
    root.mkdir()
    return [HubFileRef(write_qa_csv(root, ROWS), "questions"), HubFileRef(_write_chart_dir(root), "charts")]


@contextmanager
def _composed(registry: Path, refs: list[HubFileRef]) -> Generator[TimeFDataset]:
    """Compose the plots dataset inside its parent context, as the engine does."""
    connector = ARFBenchPlotsConnector()
    with open_parents(connector.metadata(), LocalRegistry(registry)) as context:
        yield connector.compose(refs, context)


def test_the_connector_declares_the_series_dataset_as_its_parent():
    connector = ARFBenchPlotsConnector()

    assert connector.metadata().dataset_id == "datadog/arfbench-plots"
    assert connector.values_backend == "zarr"
    assert dict(connector.SOURCE_FILES) == {"questions": "arfbench-qa.csv", "charts": "arfbench-images/*.png"}
    assert {str(alias): str(ref) for alias, ref in connector.metadata().parents.items()} == {
        "series": "datadog/arfbench@1.0.0"
    }


def test_the_child_imports_the_series_tasks_and_adds_a_chart_task_per_question(registry, charts):
    with _composed(registry, charts) as dataset:
        records = [record.metadata["source_record_id"] for record in dataset.records]
        tasks = dataset.get_all()

    # The import brings the metric records in the order that the tasks first cite them. The charts come next.
    assert records == [
        "arfbench-35997_1",
        "arfbench-35997_0",
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    # The series tasks come first, then the chart tasks. The row index of the question pairs the two tasks.
    assert [task.metadata["question_index"] for task in tasks] == [0, 1, 0, 1]
    assert [read_values(task.targets) for task in tasks] == [("Yes",), ("No",)] * 2
    series_correlation, chart_correlation = tasks[1], tasks[3]
    assert series_correlation.input_modalities == frozenset({InputModality.TEXT, InputModality.TIME_SERIES})
    assert chart_correlation.input_modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})
    # Both tasks of a question pick among its options, from the same label set.
    assert isinstance(series_correlation, ClassificationTask) and isinstance(chart_correlation, ClassificationTask)
    assert series_correlation.label_set == chart_correlation.label_set
    assert series_correlation.label_set.name == "question_1"
    # The chart task reads the combined chart first, then the chart of each metric.
    assert [record.metadata["source_record_id"] for record in chart_correlation.inputs] == [
        "arfbench-chart-35997_0-35997_1",
        "arfbench-chart-35997_0",
        "arfbench-chart-35997_1",
    ]
    assert [record.metadata["source_record_id"] for record in tasks[2].inputs] == ["arfbench-chart-35997_1"]
    assert all(task.split is Split.TEST for task in tasks)
    assert facts(chart_correlation) == facts(series_correlation)


def test_the_child_stores_only_the_charts_and_reads_the_series_from_the_parent(registry, charts):
    connector = ARFBenchPlotsConnector()
    with _composed(registry, charts) as dataset:
        dataset.derive_schema()
        version_dir = store_dataset(dataset, registry, values_backend=connector.values_backend)
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert reader.values_backend == "zarr"
        back = {record.metadata["source_record_id"]: record for record in reader.iter_records()}
        tasks = reader.read().get_all()
    # The series values stay in the Parquet release of the parent. The child names it as a payload source.
    assert [str(source) for source in open_local_version(version_dir).manifest.payload_sources] == [
        "datadog/arfbench@1.0.0"
    ]
    # The child stores its control plane and the two chart shapes. It stores no series values of its own.
    assert sorted(path.relative_to(version_dir).parts[0] for path in version_dir.iterdir()) == [
        "control.duckdb",
        "manifest.json",
        "time_series.zarr",
    ]
    assert sorted(path.name for path in (version_dir / "time_series.zarr").iterdir() if path.is_dir()) == [
        "rgb_image_2x6",
        "rgba_image_2x3",
    ]
    assert back["arfbench-35997_0"].sources[0].signals[1].to_arrow().to_pylist() == [1.5, None, 2.5]
    chart = back["arfbench-chart-35997_0"].signals[0]
    assert chart.spec.spec_type == "rgba_image_2x3"
    np.testing.assert_array_equal(chart.to_numpy()[0, 0, 0], [10, 20, 30, 42])
    assert [task.metadata["question_index"] for task in tasks] == [0, 1, 0, 1]


def test_a_pair_without_a_combined_chart_raises(registry, charts):
    questions = next(ref.path for ref in charts if ref.part == "questions")
    write_qa_csv(questions.parent, [*ROWS, {**CORRELATION_ROW, "Unnamed: 0": "2", "query_group": "35997_1, 35997_0"}])
    with (
        pytest.raises(TimeNetConnectorError, match=r"question 2: no chart for 35997_1-35997_0"),
        _composed(registry, charts),
    ):
        pass
