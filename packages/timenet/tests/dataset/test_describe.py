from timenet.dataset import TimeFDataset
from timenet.dataset.describe import describe_text
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, License, Version


def test_describe_text_has_all_sections():
    dataset = make_dataset()
    dataset.derive_schema()
    text = describe_text(dataset, rows=5)
    assert "hello_world @ 1.0.0" in text
    assert "CC-BY-4.0" in text
    assert "counts" in text
    assert "classification=1" in text  # tasks histogram
    assert "sine=3" in text  # series-by-spec histogram
    assert "specs" in text
    for spec_type in ("sine", "cosine"):
        assert spec_type in text
    assert "sample-0" in text  # sample preview


def test_describe_works_without_derived_schema():
    dataset = make_dataset()  # derive_schema() not called
    assert dataset.schema is None
    text = describe_text(dataset, rows=5)
    assert "hello_world @ 1.0.0" in text
    assert "sample-0" in text


def test_describe_rows_limits_preview():
    text = describe_text(make_dataset(), rows=1)
    assert "first 1 of 3" in text
    assert "sample-0" in text
    assert "sample-1" not in text


def test_describe_prints_to_stdout(capsys):
    make_dataset().describe()
    assert "hello_world @ 1.0.0" in capsys.readouterr().out


def test_describe_empty_dataset_does_not_crash():
    empty = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="empty", dataset_version=Version(1, 0, 0), name="E", description="d", license=License.MIT
        )
    )
    text = describe_text(empty, rows=5)
    assert "empty @ 1.0.0" in text
    assert "samples      0" in text
