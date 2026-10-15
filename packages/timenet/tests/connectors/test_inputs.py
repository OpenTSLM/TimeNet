import pytest

from timenet.connectors import BuildContext
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


def test_parent_building_is_opt_in_and_exact(tmp_path):
    parent = make_dataset()
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    metadata = parent.metadata.model_copy(update={"dataset_id": "test/child", "inputs": {"parent": ref}})
    registry = LocalRegistry(tmp_path)
    calls = []

    def build_parent(requested):
        calls.append(requested)
        with TimeFWriter(tmp_path, parent) as writer:
            writer.write()

    with pytest.raises(TimeNetDatasetNotFoundError, match="--build-parents"):
        BuildContext.open(metadata, registry, parent_builder=build_parent)
    assert calls == []
    with BuildContext.open(metadata, registry, build_parents=True, parent_builder=build_parent) as context:
        assert context.inputs["parent"].metadata.dataset_id == ref.dataset_id
        assert context.locks["parent"].source == ref
    assert calls == [ref]
    with BuildContext.open(metadata, registry, build_parents=True, parent_builder=build_parent):
        pass
    assert calls == [ref]
    cyclic = parent.metadata.model_copy(update={"inputs": {"self": ref}})
    with pytest.raises(TimeFValidationError, match="cycle"):
        BuildContext.open(cyclic, registry, build_parents=True)


def test_parent_fetching_is_opt_in_and_precedes_conversion(tmp_path):
    parent = make_dataset()
    origin = LocalRegistry(tmp_path / "origin")
    origin.store(parent)
    target = LocalRegistry(tmp_path / "target")
    ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    metadata = parent.metadata.model_copy(update={"dataset_id": "test/child", "inputs": {"parent": ref}})
    with pytest.raises(TimeNetDatasetNotFoundError, match="--fetch-parents"):
        BuildContext.open(metadata, target)
    with BuildContext.open(metadata, target, fetch_parents=str(origin.root)) as context:
        version = target.open_version(ref.dataset_id, str(ref.version))
        assert all(
            (target.root / ref.dataset_id / str(ref.version) / part.path).is_file()
            for part in version.manifest.files.all_files()
        )
        assert (
            context.inputs["parent"]
            .read()
            .records[0]
            .signals[0]
            .to_arrow()
            .equals(parent.records[0].signals[0].to_arrow())
        )
