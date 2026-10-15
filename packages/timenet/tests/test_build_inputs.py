from pathlib import Path

import pyarrow as pa
import pytest

from timenet.connectors import BaseConnector, BuildContext
from timenet.connectors.context import build_scope
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.engine import run_pipeline
from timenet.errors import TimeNetBuildError
from timenet.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import DatasetRef, Version


class DerivedConnector(BaseConnector[None]):
    def __init__(self, metadata):
        super().__init__()
        self.card = metadata
        self.raw_downloads = 0
        self.input_reader = None

    def metadata(self):
        return self.card

    def download(self, cache_dir: Path) -> list[None]:
        self.raw_downloads += 1
        return []

    def convert(self, raw_refs: list[None], *, context: BuildContext) -> TimeFDataset:
        parent = context.inputs["base"]
        self.input_reader = parent
        child = TimeFDataset(metadata=self.card)
        imported = child.copy_from(parent, namespace="base", include_tasks=True)
        original = next(iter(imported.signals.values()))

        def transformed():
            assert parent._control is not None, "input reader closed before writing the computed transform"
            values = original.values.read()
            return pa.array([value * 2 for value in values.to_pylist()], type=values.type)

        signal = Signal.from_loader(
            loader=transformed,
            n_values=original.n_values,
            spec=original.spec,
            time_axis=original.time_axis,
            name="doubled",
        )
        child.add_record(record=Record(record_id="transformed", sources=(Source(name="transform", signals=(signal,)),)))
        return child


def test_stored_pinned_input_reuses_payloads_and_lives_through_write(tmp_path):
    parent = make_dataset()
    parent.write(path=tmp_path)
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    connector = DerivedConnector(
        parent.metadata.model_copy(update={"dataset_id": "timenet/derived", "inputs": {"base": reference}})
    )
    path = run_pipeline(connector, tmp_path)
    assert connector.raw_downloads == 1
    assert connector.input_reader is not None and connector.input_reader._control is None
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        transformed = next(reader.iter_records(["transformed"])).signals[0].values.read()
        manifest = DatasetVersion.open_local(path).manifest
    original = parent.records[0].signals[0].values.read()
    assert transformed.to_pylist() == [value * 2 for value in original.to_pylist()]
    assert manifest.payload_sources[0].dataset == reference
    assert manifest.metadata.inputs == {"base": reference}


def test_missing_input_pin_fails_before_running_the_child(tmp_path, monkeypatch):
    parent = make_dataset()
    requested = DatasetRef(dataset_id=parent.metadata.dataset_id, version=Version(9, 0, 0))
    connector = DerivedConnector(
        parent.metadata.model_copy(update={"dataset_id": "timenet/derived", "inputs": {"base": requested}})
    )

    class OtherReleaseBuilder:
        def declared_version(self, dataset_id):
            return "1.0.0"

        def build(self, *args, **kwargs):
            pytest.fail("an incompatible connector was built")

    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("timenet.engine.inputs.find_builder", lambda dataset_id: OtherReleaseBuilder())
    with pytest.raises(TimeNetBuildError, match="declares release"):
        run_pipeline(connector, tmp_path)
    assert connector.raw_downloads == 0


def test_exact_build_path_detects_cycles_across_process_contexts(tmp_path, monkeypatch):
    parent = make_dataset()
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    connector = DerivedConnector(
        parent.metadata.model_copy(update={"dataset_id": "timenet/derived", "inputs": {"base": reference}})
    )
    with build_scope(reference), pytest.raises(TimeNetBuildError, match="cycle"):
        run_pipeline(connector, tmp_path)


def test_missing_exact_input_builds_once_and_is_reused(tmp_path, monkeypatch):
    parent = make_dataset()
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    builds = []

    class ExactBuilder:
        def declared_version(self, dataset_id):
            return str(parent.metadata.dataset_version)

        def build(self, dataset_id, root, *, force=False):
            builds.append((dataset_id, force))
            return parent.write(path=root)

    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("timenet.engine.inputs.find_builder", lambda dataset_id: ExactBuilder())
    metadata = parent.metadata.model_copy(update={"dataset_id": "timenet/derived", "inputs": {"base": reference}})
    first = run_pipeline(DerivedConnector(metadata), tmp_path)
    second = run_pipeline(DerivedConnector(metadata.model_copy(update={"dataset_version": Version(1, 0, 1)})), tmp_path)
    assert first.exists() and second.exists()
    assert builds == [(parent.metadata.dataset_id, False)]
