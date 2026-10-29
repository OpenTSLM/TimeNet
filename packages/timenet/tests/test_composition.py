import pytest

from timenet.composition import BuildContext
from timenet.errors import TimeFValidationError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import Annotation


def test_import_keeps_parent_values_and_isolates_child_annotations(tmp_path):
    parent = make_dataset()
    registry = LocalRegistry(tmp_path)
    parent.records[0].metadata["source_record_id"] = "base-record"
    registry.store(parent)
    metadata = type(parent.metadata).model_validate(
        {
            **parent.metadata.model_dump(),
            "dataset_id": "test/child",
            "parents": {"base": f"{parent.metadata.dataset_id}@{parent.metadata.dataset_version}"},
        }
    )
    with BuildContext.open(metadata, registry) as context:
        source = next(context.parent("base").iter_records())
        assert next(context.parent("base").iter_records([source.id])) is source
        assert next(context.parent("base").iter_records(["base-record"])) is source
        child = context.dataset()
        imported = child.import_record(source, parent="base")
        assert child.import_record(source, parent="base") is imported
        imported.annotate(Annotation(key="child_only", value=True))
        assert all(annotation.key != "child_only" for annotation in source.annotations)
        assert imported.signals[0]._values is source.signals[0]._values
        assert imported is not source
        registry.store(child)
        with pytest.raises(TimeFValidationError, match="was not yielded"):
            child.import_record(parent.records[0], parent="base")
    handle = registry.open_version("test/child")
    assert handle.manifest.payload_sources
    with BuildContext.open(metadata, registry) as context:
        source = next(context.parent("base").iter_records())
        assert imported.signals[0].to_numpy().tolist() == source.signals[0].to_numpy().tolist()
