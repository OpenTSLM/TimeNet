import pytest

from timenet import builders
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetInvalidCardError,
    TimeNetRegistryError,
)
from timenet.registry import LocalRegistry
from timenet.sources import (
    Found,
    InstalledConnectors,
    NotFound,
    NotServed,
    ReleaseSource,
    resolve_release,
)
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, Version
from timenet.types.access import AccessRequirement
from timenet.types.references import DatasetRef


URL = "https://example.org/dua"
V1, V11, V2 = Version(1, 0, 0), Version(1, 1, 0), Version(2, 0, 0)


class _Source:
    """Serves the versions in ``served`` and lists the versions in ``listed`` without serving them."""

    def __init__(self, *served: str, listed: tuple[str, ...] = ()):
        self.served, self.listed = set(served), set(listed)

    def resolve(self, dataset_id, version):
        known = sorted(self.served | self.listed, key=Version.parse)
        wanted = (known[-1] if known else None) if version is None else version
        if wanted in self.served:
            return Found(Version.parse(wanted), self)
        if wanted in self.listed:
            return NotServed(Version.parse(wanted), Access.CREDENTIALED, URL)
        return NotFound()


class _Broken:
    def resolve(self, dataset_id, version):
        raise TimeNetRegistryError("the registry does not answer")


class _Backend:
    def __init__(self, card):
        self._card = card

    def dataset_ids(self):
        return [self._card.dataset_id]

    def card(self, dataset_id):
        return self._card if dataset_id == self._card.dataset_id else None


def _fail(*args, **kwargs):
    raise AssertionError("resolve() must not read every manifest")


def test_an_exact_version_comes_from_the_first_source_that_provides_it():
    second = _Source("1.0.0")
    assert resolve_release([_Source(), second, _Source("1.0.0")], "a/b", "1.0.0") == Found(V1, second)


def test_latest_is_the_highest_version_across_all_sources():
    remote = _Source("1.0.0", "1.1.0")
    assert resolve_release([_Source("1.0.0"), remote], "a/b", None) == Found(V11, remote)


def test_latest_comes_from_the_first_source_that_holds_the_highest_version():
    store = _Source("1.1.0")
    assert resolve_release([store, _Source("1.1.0")], "a/b", None) == Found(V11, store)


def test_a_release_that_is_not_served_falls_through_and_does_not_count_for_latest():
    connectors = _Source("1.0.0")
    assert resolve_release([_Source(listed=("2.0.0",)), connectors], "a/b", None) == Found(V1, connectors)
    assert resolve_release([_Source(listed=("2.0.0",))], "a/b", "2.0.0") == NotServed(V2, Access.CREDENTIALED, URL)


def test_latest_asks_a_source_that_did_not_serve_a_higher_version_for_the_chosen_one():
    hosted = _Source("1.0.0", listed=("2.0.0",))
    assert resolve_release([hosted, _Source("1.0.0")], "a/b", None) == Found(V1, hosted)


def test_no_source_gives_not_found():
    assert resolve_release([_Source(), _Source()], "a/b", None) == NotFound()
    assert resolve_release([_Source("1.0.0")], "a/b", "2.0.0") == NotFound()


def test_an_error_of_a_source_stops_the_resolution():
    with pytest.raises(TimeNetRegistryError):
        resolve_release([_Broken(), _Source("1.0.0")], "a/b", "1.0.0")
    with pytest.raises(TimeNetRegistryError):
        resolve_release([_Source("1.0.0"), _Broken()], "a/b", None)


def test_not_served_names_the_access_page_of_each_card():
    own = AccessRequirement(DatasetRef.parse("a/b@2.0.0"), Access.CREDENTIALED, URL)
    ancestor = AccessRequirement(DatasetRef.parse("a/ancestor@1.0.0"), Access.RESTRICTED, "https://example.org/ask")
    error = NotServed(V2, Access.RESTRICTED, URL, (own, ancestor)).access_error("a/b")
    assert isinstance(error, TimeNetAccessError)
    assert (error.dataset, error.access, error.requirements) == (
        DatasetRef.parse("a/b@2.0.0"),
        Access.RESTRICTED,
        (own, ancestor),
    )


def test_installed_connectors_answer_with_the_declared_version(monkeypatch):
    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card))
    connectors = InstalledConnectors()
    assert connectors.resolve(card.dataset_id, None) == Found(V1, connectors)
    assert connectors.resolve(card.dataset_id, "1.0.0") == Found(V1, connectors)
    assert str(connectors) == "installed connectors"


def test_installed_connectors_answer_not_found_for_another_version(monkeypatch):
    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card))
    assert InstalledConnectors().resolve(card.dataset_id, "2.0.0") == NotFound()
    assert InstalledConnectors().resolve("x/y", None) == NotFound()


def test_installed_connectors_without_a_package_answer_not_found(monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    assert InstalledConnectors().resolve("timenet/hello-world", None) == NotFound()
    assert "no connector package is installed" in str(InstalledConnectors())


def test_a_card_error_of_a_backend_propagates(monkeypatch):
    class Broken(_Backend):
        def card(self, dataset_id):
            raise TimeNetInvalidCardError("the card cannot be read")

    install_backends(monkeypatch, Broken(make_dataset().metadata))
    dataset_id = make_dataset().metadata.dataset_id
    with pytest.raises(TimeNetInvalidCardError):
        InstalledConnectors().resolve(dataset_id, None)
    with pytest.raises(TimeNetInvalidCardError):
        resolve_release([InstalledConnectors()], dataset_id, None)


def test_not_served_names_the_page_when_it_has_no_requirements():
    error = NotServed(V2, Access.CREDENTIALED, URL).access_error("a/b")
    assert error.requirements == (AccessRequirement(DatasetRef.parse("a/b@2.0.0"), Access.CREDENTIALED, URL),)
    assert URL in str(error)


@pytest.mark.parametrize("latest", [None, "", "latest"])
def test_every_latest_sentinel_means_the_latest_version(latest, tmp_path, monkeypatch):
    remote = _Source("1.0.0", "1.1.0")
    assert resolve_release([remote], "a/b", latest) == Found(V11, remote)

    store = LocalRegistry(tmp_path)
    store.put_dataset(make_dataset())
    assert store.resolve("timenet/hello-world", latest) == Found(V1, store)

    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card))
    connectors = InstalledConnectors()
    assert connectors.resolve(card.dataset_id, latest) == Found(V1, connectors)


def test_installed_connectors_are_equal():
    assert InstalledConnectors() == InstalledConnectors()
    assert len({InstalledConnectors(), InstalledConnectors()}) == 1
    assert isinstance(InstalledConnectors(), ReleaseSource)


def test_a_local_registry_answers_not_found_without_reading_every_manifest(tmp_path, monkeypatch):
    registry = LocalRegistry(tmp_path)
    registry.put_dataset(make_dataset())
    monkeypatch.setattr(LocalRegistry, "list_datasets", _fail)
    assert registry.resolve("test/unknown", None) == NotFound()
    assert registry.resolve("test/unknown", "1.0.0") == NotFound()
    assert registry.resolve("timenet/hello-world", "2.0.0") == NotFound()


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_a_local_registry_rejects_an_invalid_dataset_id(tmp_path, version):
    with pytest.raises(TimeFValidationError):
        LocalRegistry(tmp_path).resolve("../evil", version)
