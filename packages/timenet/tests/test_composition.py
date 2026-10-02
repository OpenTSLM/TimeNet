from dataclasses import replace

import pytest

from timenet.composition import BuildContext, DatasetBuilder
from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import DatasetRef, ParentDataset, Version
from timenet.writer import TimeFWriter


def _write_parent(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    return dataset


def test_build_context_imports_records_and_locks_the_parent(tmp_path):
    parent = _write_parent(tmp_path)
    parent_ref = DatasetRef(parent.metadata.dataset_id, parent.metadata.dataset_version)
    metadata = replace(
        parent.metadata,
        dataset_id="test/child",
        dataset_version=Version(1, 0, 0),
        parents=(ParentDataset("base", parent_ref),),
    )

    with BuildContext.open(metadata, LocalRegistry(tmp_path)) as context:
        builder = context.dataset()
        record = next(context.parent("base").iter_records())
        assert isinstance(builder, DatasetBuilder)
        assert builder.import_record(record, parent="base") is record
        assert builder.tasks == ()
        dependencies = context.dependency_lock()

    assert dependencies.direct[0].dataset == parent_ref
    assert dependencies.lock[0].dataset == parent_ref
    assert dependencies.lock[0].manifest_checksum.startswith("sha256:")


def test_builder_rejects_a_replacement_record_not_yielded_by_parent(tmp_path):
    parent = _write_parent(tmp_path)
    parent_ref = DatasetRef(parent.metadata.dataset_id, parent.metadata.dataset_version)
    metadata = replace(
        parent.metadata,
        dataset_id="test/child",
        parents=(ParentDataset("base", parent_ref),),
    )

    with (
        BuildContext.open(metadata, LocalRegistry(tmp_path)) as context,
        pytest.raises(TimeFValidationError, match="was not yielded"),
    ):
        context.dataset().import_record(Record(record_id=parent.records[0].id), parent="base")
