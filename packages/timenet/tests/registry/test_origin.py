"""The provenance record says where each release in the store comes from, and stays in the store."""

from datetime import UTC, datetime
import json

from _demo_child import _child_of_the_demo
from _origin_record import origin_file
from pydantic import ValidationError
import pytest

import timenet
from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry
from timenet.registry.origin import Origin, OriginKind, without_credentials
from timenet.testing import make_dataset
from timenet.types.references import DatasetRef


DEMO = DatasetRef(dataset_id=make_dataset().metadata.dataset_id, version=make_dataset().metadata.dataset_version)


@pytest.fixture
def store(tmp_path) -> LocalRegistry:
    registry = LocalRegistry(tmp_path / "store")
    registry.put_dataset(make_dataset())
    return registry


def _read(store: LocalRegistry) -> Origin:
    """Return the record of the demo release, which must have one."""
    origin = store.read_origin(DEMO)
    assert origin is not None
    return origin


def test_a_record_round_trips(store):
    before = datetime.now(UTC).replace(microsecond=0)
    store.record_origin(DEMO, OriginKind.BUILD, "timenet-connectors 0.9.0")
    origin = _read(store)

    assert (origin.kind, origin.source) == (OriginKind.BUILD, "timenet-connectors 0.9.0")
    assert origin.at.tzinfo is UTC
    assert before <= origin.at <= datetime.now(UTC)
    # The file holds the three keys, with the time in UTC to the second.
    record = json.loads(origin_file(store, DEMO).read_text())
    assert record == {
        "kind": "build",
        "source": "timenet-connectors 0.9.0",
        "at": origin.at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def test_a_new_record_replaces_the_old_one(store):
    store.record_origin(DEMO, OriginKind.DOWNLOAD, "https://registry.example.org")
    store.record_origin(DEMO, OriginKind.BUILD, "timenet-connectors 0.9.0")
    assert _read(store).kind is OriginKind.BUILD
    # The write leaves no temporary file next to the record.
    path = origin_file(store, DEMO)
    assert list(path.parent.glob(f"{path.name}*")) == [path]


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("https://user:secret@registry.example.org:8443/r?sig=abc#top", "https://registry.example.org:8443/r"),
        ("/data/registry", "/data/registry"),
    ],
)
def test_without_credentials_keeps_only_scheme_host_and_path_of_a_url(location, expected):
    assert without_credentials(location) == expected


def test_a_release_without_a_record_has_none(store):
    assert store.read_origin(DEMO) is None


def test_a_record_needs_the_release_in_the_local_registry(tmp_path):
    empty = LocalRegistry(tmp_path / "store")
    with pytest.raises(TimeNetDatasetNotFoundError, match=f"the local registry holds no {DEMO}"):
        empty.record_origin(DEMO, OriginKind.DOWNLOAD, "https://registry.example.org")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("not json", "Invalid JSON: expected ident at line 1 column 2"),
        (
            '{"kind": "other", "source": "x", "at": "2026-11-03T14:02:11Z"}',
            "kind: Input should be 'download' or 'build'",
        ),
    ],
    ids=["invalid-json", "other-kind"],
)
def test_a_damaged_record_raises_with_its_first_problem(store, text, problem):
    path = origin_file(store, DEMO)
    path.write_text(text)
    with pytest.raises(TimeFFormatError) as caught:
        store.read_origin(DEMO)
    assert str(caught.value) == f"{path} is not a valid provenance record: {problem}."
    assert isinstance(caught.value.__cause__, ValidationError)


def test_a_time_with_another_offset_reads_as_utc(store):
    path = origin_file(store, DEMO)
    path.write_text('{"kind": "download", "source": "x", "at": "2026-11-03T16:02:11+02:00"}')
    at = _read(store).at
    assert at == datetime(2026, 11, 3, 14, 2, 11, tzinfo=UTC)
    assert at.tzinfo is UTC


def test_publish_never_copies_the_record(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    _, parent = _child_of_the_demo(store.root)
    child = DatasetRef.parse("test/child@1.0.0")
    store.record_origin(parent, OriginKind.DOWNLOAD, "https://registry.example.org")
    store.record_origin(child, OriginKind.BUILD, "timenet-connectors 0.9.0")

    timenet.publish(str(child), dest=tmp_path / "dest", source=store)

    dest = LocalRegistry(tmp_path / "dest")
    assert dest.exists(child.dataset_id, str(child.version))
    assert dest.exists(parent.dataset_id, str(parent.version))
    assert not list(dest.root.rglob(f"{origin_file(dest, child).name}*"))
