import pytest

from timenet.composition import BuildContext
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, DatasetRef
from timenet.writer import TimeFWriter


def _write_parent(tmp_path) -> TimeFDataset:
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return dataset


def _child_metadata(parent: TimeFDataset) -> DatasetMetadata:
    return DatasetMetadata.model_validate(
        {
            **parent.metadata.model_dump(),
            "dataset_id": "test/child",
            "parents": [f"{parent.metadata.dataset_id}@{parent.metadata.dataset_version}"],
        }
    )


def test_build_context_imports_records_and_locks_the_parent(tmp_path):
    parent = _write_parent(tmp_path)
    parent_ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)

    with BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context:
        dataset = TimeFDataset(metadata=context.metadata)
        record = next(context.parent("timenet/hello-world").iter_records())
        (imported,) = context.parent("timenet/hello-world").import_records(dataset, [record.id])
        assert imported.id == f"{parent_ref}::{record.id}"
        with pytest.raises(TimeFValidationError, match="not qualified"):
            dataset.import_record(record, parent="timenet/hello-world")
        assert dataset.records == (imported,)
        context.verify_imports(dataset)
        assert dataset.owned_records == ()
        assert dataset.tasks == ()
        lock = context.dependency_lock()

    assert [dependency.dataset for dependency in lock] == [parent_ref]
    assert lock[0].manifest_checksum.startswith("sha256:")


def test_build_context_rejects_an_import_the_parent_does_not_hold(tmp_path):
    parent = _write_parent(tmp_path)
    foreign = make_dataset().records[1]
    foreign.record_id = f"{parent.metadata.dataset_id}@{parent.metadata.dataset_version}::foreign"
    with BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context:
        dataset = TimeFDataset(metadata=context.metadata)
        context.parent("timenet/hello-world").import_records(dataset, ["record-0"])
        dataset.import_record(foreign, parent="timenet/hello-world")
        with pytest.raises(TimeFValidationError, match=r"does not hold imported record\(s\) \['foreign'\]"):
            context.verify_imports(dataset)


def test_build_context_rejects_an_undeclared_parent_dataset_id(tmp_path):
    parent = _write_parent(tmp_path)
    with (
        BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context,
        pytest.raises(TimeFValidationError, match="no parent"),
    ):
        context.parent("other")


def test_build_context_addresses_same_named_parents_by_full_dataset_id(tmp_path):
    registry = LocalRegistry(tmp_path)
    parent_ids = ("first/recordings", "second/recordings")
    for dataset_id in parent_ids:
        original = make_dataset()
        dataset = TimeFDataset(
            metadata=DatasetMetadata.model_validate({**original.metadata.model_dump(), "dataset_id": dataset_id})
        )
        dataset.add_record(record=original.records[0])
        registry.store(dataset)
    metadata = DatasetMetadata.model_validate(
        {
            **dataset.metadata.model_dump(),
            "dataset_id": "test/child",
            "parents": [f"{name}@1.0.0" for name in parent_ids],
        }
    )
    with BuildContext.open(metadata, registry) as context:
        assert set(context.parents) == set(parent_ids)
        for dataset_id in parent_ids:
            assert context.parent(dataset_id).reference.dataset_id == dataset_id
        with pytest.raises(TimeFValidationError, match="no parent"):
            context.parent("recordings")
