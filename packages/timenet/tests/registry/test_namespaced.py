from timenet.dataset import TimeFDataset, TimeSeries
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import assert_datasets_equal, make_dataset, sine_loader
from timenet.types import (
    ClassificationTask,
    DatasetMetadata,
    License,
    TimeSeriesSpec,
    Version,
    View,
    ureg,
)
from timenet.writer import TimeFWriter


def _namespaced_dataset() -> TimeFDataset:
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="ChengsenWang/TSQA",
            dataset_version=Version(1, 0, 0),
            name="TSQA",
            description="A namespaced dataset.",
            license=License.APACHE_2_0,
        )
    )
    spec = TimeSeriesSpec(
        spec_type="tsqa_series",
        name="Series",
        unit_sampling_rate=ureg.hertz,
        unit_timestamp=ureg.second,
        unit_value=ureg.dimensionless,
    )
    series = TimeSeries(
        spec=spec,
        channel="v",
        sampling_rate_hz=1.0,
        loader=sine_loader(n=8, sampling_rate_hz=1.0),
        time_series_id="ns-ts-0",
        t_start_s=0.0,
        t_end_s=8.0,
    )
    sample = dataset.add_sample(time_series=(series,), view=View.FULL, sample_id="ns-sample-0")
    dataset.add_task(sample, ClassificationTask(label="x", id="ns-task-0"))
    return dataset


def _write(root, dataset):
    dataset.derive_schema()
    with TimeFWriter(root, dataset) as writer:
        writer.write()


def test_list_datasets_finds_multiple_namespaced(tmp_path):
    _write(tmp_path, make_dataset())  # namespaced id: timenet/hello-world
    _write(tmp_path, _namespaced_dataset())  # namespaced id: ChengsenWang/TSQA
    ids = {m.dataset_id for m in LocalRegistry(tmp_path).list_datasets()}
    assert ids == {"timenet/hello-world", "ChengsenWang/TSQA"}


def test_get_manifest_and_open_file_namespaced(tmp_path):
    _write(tmp_path, _namespaced_dataset())
    registry = LocalRegistry(tmp_path)
    manifest = registry.get_manifest("ChengsenWang/TSQA")
    assert manifest.dataset_id == "ChengsenWang/TSQA"
    with registry.open_file("ChengsenWang/TSQA", "1.0.0", "manifest.json") as handle:
        assert b"TSQA" in handle.read()


def test_namespaced_round_trip(tmp_path):
    original = _namespaced_dataset()
    _write(tmp_path, _namespaced_dataset())
    version_dir = tmp_path / "ChengsenWang" / "TSQA" / "1.0.0"
    assert version_dir.exists()
    with TimeFReader(version_dir) as reader:
        assert_datasets_equal(original, reader.read())


def test_cache_dir_is_ignored_by_list(tmp_path):
    _write(tmp_path, _namespaced_dataset())
    (tmp_path / ".cache" / "ChengsenWang").mkdir(parents=True)
    ids = {m.dataset_id for m in LocalRegistry(tmp_path).list_datasets()}
    assert ids == {"ChengsenWang/TSQA"}
