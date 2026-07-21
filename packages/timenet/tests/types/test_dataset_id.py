import pytest

from timenet.types import DatasetMetadata, License, Version


def _metadata(dataset_id: str) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_version=Version(1, 0, 0),
        name="n",
        description="d",
        license=License.MIT,
    )


@pytest.mark.parametrize("dataset_id", ["hello_world", "ChengsenWang/TSQA", "a.b-c", "org/name-1"])
def test_accepts_flat_and_namespaced_ids(dataset_id):
    assert _metadata(dataset_id).dataset_id == dataset_id


@pytest.mark.parametrize(
    "dataset_id",
    ["", "/leading", "trailing/", "a/b/c", "has space", "bad$char", "a//b"],
)
def test_rejects_bad_ids(dataset_id):
    with pytest.raises(ValueError):
        _metadata(dataset_id)


@pytest.mark.parametrize("dataset_id", ["..", ".", "../evil", "foo/..", "../..", "./x"])
def test_rejects_path_traversal_ids(dataset_id):
    # dataset ids are joined into filesystem paths, so a traversal segment must never validate.
    with pytest.raises(ValueError):
        _metadata(dataset_id)


@pytest.mark.parametrize("dataset_id", [".hidden", "foo/.bar", ".git"])
def test_rejects_leading_dot_segments(dataset_id):
    # A leading-dot id writes to disk but discovery skips hidden dirs, so it must never validate.
    with pytest.raises(ValueError):
        _metadata(dataset_id)
