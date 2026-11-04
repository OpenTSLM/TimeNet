"""The provenance record says where each release in the store comes from, and stays in the store."""

from datetime import UTC, datetime
import json

from _demo_child import _child_of_the_demo
import pytest

import timenet
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry
from timenet.registry._paths import release_dir
from timenet.registry.origin import ORIGIN_FILE, Origin, read_origin, record_origin, without_credentials
from timenet.testing import make_dataset
from timenet.types.references import DatasetRef


DEMO = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)


@pytest.fixture
def store(tmp_path) -> LocalRegistry:
    registry = LocalRegistry(tmp_path / "store", store=True)
    registry.put_dataset(make_dataset())
    return registry


def _read(store: LocalRegistry) -> Origin:
    """Return the record of the demo release, which must have one."""
    origin = read_origin(store, DEMO)
    assert origin is not None
    return origin


def test_a_record_round_trips(store):
    before = datetime.now(UTC).replace(microsecond=0)
    record_origin(store, DEMO, "build", "timenet-connectors 0.9.0")
    origin = _read(store)

    assert (origin.origin, origin.source) == ("build", "timenet-connectors 0.9.0")
    assert origin.at.tzinfo is UTC
    assert before <= origin.at <= datetime.now(UTC)
    # The file holds the three keys, with the time in UTC to the second.
    record = json.loads((release_dir(store.root, DEMO) / ORIGIN_FILE).read_text())
    assert record == {
        "origin": "build",
        "source": "timenet-connectors 0.9.0",
        "at": origin.at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def test_a_new_record_replaces_the_old_one(store):
    record_origin(store, DEMO, "copy", "https://registry.example.org")
    record_origin(store, DEMO, "build", "timenet-connectors 0.9.0")
    assert _read(store).origin == "build"
    # The write leaves no temporary file next to the record.
    assert [path.name for path in release_dir(store.root, DEMO).glob(f"{ORIGIN_FILE}*")] == [ORIGIN_FILE]


def test_a_url_is_recorded_without_user_info_query_or_fragment(store):
    record_origin(store, DEMO, "copy", "https://user:secret@registry.example.org/r?sig=abc#top")
    assert _read(store).source == "https://registry.example.org/r"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("https://user:secret@registry.example.org:8443/r?sig=abc#top", "https://registry.example.org:8443/r"),
        ("s3://bucket/prefix?versionId=1", "s3://bucket/prefix"),
        ("https://registry.example.org", "https://registry.example.org"),
        ("/data/registry", "/data/registry"),
        ("timenet-connectors 0.9.0", "timenet-connectors 0.9.0"),
    ],
)
def test_without_credentials_keeps_only_scheme_host_and_path_of_a_url(location, expected):
    assert without_credentials(location) == expected


def test_a_release_without_a_record_has_none(store):
    assert read_origin(store, DEMO) is None


def test_a_record_needs_the_release_in_the_store(tmp_path):
    empty = LocalRegistry(tmp_path / "store", store=True)
    with pytest.raises(TimeNetDatasetNotFoundError, match=f"the store holds no {DEMO}"):
        record_origin(empty, DEMO, "copy", "https://registry.example.org")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"origin": "other", "source": "x", "at": "2026-11-03T14:02:11Z"}',
        '{"origin": "copy", "source": "x", "at": "2026-11-03T14:02:11"}',
        '{"origin": "copy", "source": "x"}',
        '{"origin": "copy", "source": "x", "at": "2026-11-03T14:02:11Z", "size": 1}',
    ],
    ids=["invalid-json", "other-origin", "naive-time", "missing-key", "extra-key"],
)
def test_a_damaged_record_raises_with_its_fix(store, text):
    path = release_dir(store.root, DEMO) / ORIGIN_FILE
    path.write_text(text)
    with pytest.raises(TimeFFormatError, match="Delete it to count the release as built") as caught:
        read_origin(store, DEMO)
    assert str(path) in str(caught.value)


def test_a_time_with_another_offset_reads_as_utc(store):
    path = release_dir(store.root, DEMO) / ORIGIN_FILE
    path.write_text('{"origin": "copy", "source": "x", "at": "2026-11-03T16:02:11+02:00"}')
    at = _read(store).at
    assert at == datetime(2026, 11, 3, 14, 2, 11, tzinfo=UTC)
    assert at.tzinfo is UTC


def test_publish_never_copies_the_record(tmp_path):
    store = LocalRegistry(tmp_path / "store", store=True)
    _, parent = _child_of_the_demo(store.root)
    child = DatasetRef.parse("test/child@1.0.0")
    record_origin(store, parent, "copy", "https://registry.example.org")
    record_origin(store, child, "build", "timenet-connectors 0.9.0")

    timenet.publish(str(child), dest=tmp_path / "dest", source=store)

    dest = LocalRegistry(tmp_path / "dest")
    assert dest.exists(child.dataset_id, str(child.version))
    assert dest.exists(parent.dataset_id, str(parent.version))
    assert not list(dest.root.rglob(f"{ORIGIN_FILE}*"))
