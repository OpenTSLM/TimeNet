import re

from timenet.dataset import TimeFDataset
from timenet.dataset.describe import describe_text
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, License, Version


def _previews(text: str, record_id: int) -> bool:
    return re.search(rf"^  {record_id}\s", text, re.MULTILINE) is not None


def test_describe_text_has_all_sections():
    dataset = make_dataset()
    dataset.derive_schema()
    text = describe_text(dataset, rows=5)
    assert f"{dataset.metadata.dataset_id} @ {dataset.metadata.dataset_version}" in text
    assert str(dataset.metadata.license) in text
    assert "counts" in text
    # Counts are asserted loosely: make_dataset() gains records and specs further up the stack, and
    # this test is about the sections rendering, not the fixture's exact size.
    assert f"{dataset.tasks[0].task_type}=" in text  # tasks histogram
    spec_type = dataset.records[0].signals[0].spec.spec_type
    assert f"{spec_type}=" in text  # series-by-spec histogram
    assert "specs" in text
    assert spec_type in text
    assert _previews(text, dataset.records[0].id)  # record preview


def test_describe_works_without_derived_schema():
    dataset = make_dataset()  # derive_schema() not called
    assert dataset.schema is None
    text = describe_text(dataset, rows=5)
    assert f"{dataset.metadata.dataset_id} @ {dataset.metadata.dataset_version}" in text
    assert _previews(text, dataset.records[0].id)


def test_describe_rows_limits_preview():
    dataset = make_dataset()
    text = describe_text(dataset, rows=1)
    assert "first 1 of " in text
    assert _previews(text, dataset.records[0].id)
    assert not _previews(text, dataset.records[1].id)


def test_describe_prints_to_stdout(capsys):
    dataset = make_dataset()
    dataset.describe()
    assert f"{dataset.metadata.dataset_id} @ {dataset.metadata.dataset_version}" in capsys.readouterr().out


def test_describe_empty_dataset_does_not_crash():
    empty = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/empty", dataset_version=Version(1, 0, 0), name="E", description="d", license=License.MIT
        )
    )
    text = describe_text(empty, rows=5)
    assert "demo/empty @ 1.0.0" in text
    assert "records      0" in text
