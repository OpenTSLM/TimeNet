import re

from timenet.dataset import TimeFDataset
from timenet.dataset.describe import describe_text
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, License, Version
from timenet.types.ids import entity_id


def _previews(text: str, record_id: int) -> bool:
    return re.search(rf"^  {record_id}\s", text, re.MULTILINE) is not None


def test_describe_text_has_all_sections():
    dataset = make_dataset()
    dataset.derive_schema()
    text = describe_text(dataset, rows=5)
    assert "timenet/hello-world @ 1.0.0" in text
    assert "CC-BY-4.0" in text
    assert "counts" in text
    # Counts are asserted loosely: make_dataset() gains records and specs further up the stack, and
    # this test is about the sections rendering, not the fixture's exact size.
    assert "classification=" in text  # tasks histogram
    assert "sine=" in text  # series-by-spec histogram
    assert "specs" in text
    assert "sine" in text
    assert _previews(text, entity_id(dataset.records[0]))  # record preview


def test_describe_works_without_derived_schema():
    dataset = make_dataset()  # derive_schema() not called
    assert dataset.schema is None
    text = describe_text(dataset, rows=5)
    assert "timenet/hello-world @ 1.0.0" in text
    assert _previews(text, entity_id(dataset.records[0]))


def test_describe_rows_limits_preview():
    dataset = make_dataset()
    text = describe_text(dataset, rows=1)
    assert "first 1 of " in text
    assert _previews(text, entity_id(dataset.records[0]))
    assert not _previews(text, entity_id(dataset.records[1]))


def test_describe_prints_to_stdout(capsys):
    make_dataset().describe()
    assert "timenet/hello-world @ 1.0.0" in capsys.readouterr().out


def test_describe_empty_dataset_does_not_crash():
    empty = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="demo/empty", dataset_version=Version(1, 0, 0), name="E", description="d", license=License.MIT
        )
    )
    text = describe_text(empty, rows=5)
    assert "demo/empty @ 1.0.0" in text
    assert "records      0" in text
