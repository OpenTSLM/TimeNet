import json
import shutil

from _fake_registry import build_service_fake
import pytest

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetRegistryError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry
from timenet.registry.version import DatasetVersion
from timenet.testing import make_dataset
from timenet.types import Annotation, DatasetRef


@pytest.mark.parametrize("backend", ["local", "remote"])
def test_download_fetches_flat_physical_owners_and_ignores_unused_inputs(tmp_path, backend):
    registry_root = tmp_path / "registry"
    parent = make_dataset()
    parent_path = parent.write(path=registry_root)
    intermediate = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/intermediate"}))
    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
        intermediate.copy_from(reader, namespace="base")
        intermediate.annotate(Annotation(id="manual", key="manual", value="maintenance manual" * 10000))
        intermediate_path = intermediate.write(path=registry_root)
    owner_manifest = intermediate_path / "manifest.json"
    owner_manifest.write_text(json.dumps(json.loads(owner_manifest.read_text()), separators=(",", ":")))
    leaf = TimeFDataset(
        metadata=parent.metadata.model_copy(
            update={
                "dataset_id": "timenet/context-only",
                "inputs": {
                    "unused": DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
                },
            }
        )
    )
    with TimeFReader(DatasetVersion.open_local(intermediate_path)) as reader:
        leaf.annotate(reader.annotations[0])
        leaf_path = leaf.write(path=registry_root)
    manifest = DatasetVersion.open_local(leaf_path).manifest
    assert [source.dataset.dataset_id for source in manifest.payload_sources] == ["timenet/intermediate"]
    offline = tmp_path / "offline"
    target = offline / leaf.metadata.dataset_id / "1.0.0"
    transport, _ = build_service_fake(registry_root)
    registry = (
        LocalRegistry(registry_root)
        if backend == "local"
        else RemoteRegistry("http://api.local", transport=transport, cache_dir=offline)
    )
    registry.download_version(leaf.metadata.dataset_id, "1.0.0", target)
    assert not (offline / parent.metadata.dataset_id).exists()
    assert (offline / intermediate.metadata.dataset_id / "1.0.0/manifest.json").read_bytes() == (
        intermediate_path / "manifest.json"
    ).read_bytes()
    shutil.rmtree(registry_root)
    with TimeFReader(DatasetVersion.open_local(target)) as reader:
        reader.verify()
        assert reader.annotations[0].read_value() == "maintenance manual" * 10000


def test_download_rejects_corrupt_owned_payloads_before_committing_the_child(tmp_path):
    parent = make_dataset()
    parent.annotate(Annotation(key="manual", value="maintenance" * 10000))
    source = tmp_path / "source"
    parent_path = parent.write(path=source)
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/child"}))
    with TimeFReader(DatasetVersion.open_local(parent_path)) as reader:
        child.copy_from(reader, namespace="base")
        child.write(path=source)
    artifact = parent_path / "payloads/str.parquet"
    artifact.write_bytes(artifact.read_bytes()[:-1] + b"!")
    target = tmp_path / "offline" / child.metadata.dataset_id / "1.0.0"
    with pytest.raises(TimeFFormatError, match="checksum mismatch"):
        LocalRegistry(source).download_version(child.metadata.dataset_id, "1.0.0", target)
    assert not (target / "manifest.json").exists()


def test_publish_requires_hosted_owners_or_an_explicit_standalone_copy(tmp_path):
    parent = make_dataset()
    path = parent.write(path=tmp_path / "source")
    child = TimeFDataset(metadata=parent.metadata.model_copy(update={"dataset_id": "timenet/published"}))
    service = tmp_path / "service"
    service.mkdir()
    transport, requests = build_service_fake(service)
    remote = RemoteRegistry("http://api.local", transport=transport, cache_dir=tmp_path / "cache")
    with TimeFReader(DatasetVersion.open_local(path)) as reader:
        imported = child.copy_from(reader, namespace="base", include_tasks=True)
        with pytest.raises(TimeNetRegistryError, match="absent from the destination"):
            remote.store(child)
        assert all(request.method != "POST" for request in requests)
        remote.store(child, payload_policy="copy")
    shutil.rmtree(tmp_path / "source")
    with TimeFReader(remote.open_version(child.metadata.dataset_id, "1.0.0")) as reader:
        reader.verify()
        actual = next(reader.iter_records([imported.records["record-0"].id])).signals[0].values.read()
    assert actual.equals(parent.records[0].signals[0].values.read())
    assert remote.get_manifest(child.metadata.dataset_id, "1.0.0").payload_sources == ()
