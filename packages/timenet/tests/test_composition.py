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
            "parents": [
                {
                    "alias": "base",
                    "dataset_id": parent.metadata.dataset_id,
                    "version": str(parent.metadata.dataset_version),
                },
            ],
        }
    )


def test_build_context_imports_records_and_locks_the_parent(tmp_path):
    parent = _write_parent(tmp_path)
    parent_ref = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)

    with BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context:
        dataset = TimeFDataset(metadata=context.metadata)
        record = next(context.parent("base").iter_records())
        assert dataset.import_record(record, parent="base") is record
        assert dataset.records == (record,)
        assert dataset.owned_records == ()
        assert dataset.tasks == ()
        lock = context.dependency_lock()

    assert [dependency.dataset for dependency in lock] == [parent_ref]
    assert lock[0].manifest_checksum.startswith("sha256:")


def test_build_context_rejects_an_import_the_parent_does_not_hold(tmp_path):
    parent = _write_parent(tmp_path)
    foreign = make_dataset().records[1]
    foreign.record_id = "foreign"
    with BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context:
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.import_record(next(context.parent("base").iter_records(["record-0"])), parent="base")
        dataset.import_record(foreign, parent="base")
        with pytest.raises(TimeFValidationError, match=r"does not hold imported record\(s\) \['foreign'\]"):
            context.verify_imports(dataset)


def test_build_context_rejects_an_undeclared_parent_alias(tmp_path):
    parent = _write_parent(tmp_path)
    with (
        BuildContext.open(_child_metadata(parent), LocalRegistry(tmp_path)) as context,
        pytest.raises(TimeFValidationError, match="no parent alias"),
    ):
        context.parent("other")
