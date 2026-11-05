from pathlib import Path

import pytest

from timenet import builders
from timenet.errors import (
    ConnectorCardWarning,
    TimeFValidationError,
    TimeNetInvalidCardError,
    TimeNetRegistryError,
)
from timenet.registry import LocalRegistry
from timenet.sources import (
    FoundInConnector,
    FoundInRegistry,
    InstalledConnectors,
    NotFound,
    NotServed,
    resolve_release,
)
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, AnswerTask, Domain, Version


V1, V11, V2 = Version(1, 0, 0), Version(1, 1, 0), Version(2, 0, 0)


class _Source(LocalRegistry):
    """A registry that serves the versions in ``served`` and lists the versions in ``listed`` without serving them."""

    def __init__(self, *served: str, listed: tuple[str, ...] = ()):
        super().__init__(Path("unused"))
        self.served, self.listed = set(served), set(listed)

    def resolve(self, dataset_id, version):
        known = sorted(self.served | self.listed, key=Version.parse)
        wanted = (known[-1] if known else None) if version is None else str(version)
        if wanted in self.served:
            return FoundInRegistry(Version.parse(wanted), self)
        if wanted in self.listed:
            return NotServed(Version.parse(wanted), Access.CREDENTIALED, ())
        return NotFound()


class _Broken(_Source):
    def resolve(self, dataset_id, version):
        raise TimeNetRegistryError("the registry does not answer")


class _Backend:
    def __init__(self, card):
        self._card = card

    def dataset_ids(self):
        return [self._card.dataset_id]

    def card(self, dataset_id):
        return self._card if dataset_id == self._card.dataset_id else None


class _Cards:
    """A backend with a connector for each card."""

    def __init__(self, *cards):
        self.cards = {card.dataset_id: card for card in cards}

    def dataset_ids(self):
        return list(self.cards)

    def card(self, dataset_id):
        return self.cards.get(dataset_id)


def _card(dataset_id, name="Hello World"):
    return make_dataset().metadata.model_copy(update={"dataset_id": dataset_id, "name": name})


def _fail(*args, **kwargs):
    raise AssertionError("resolve() must not read every manifest")


def test_an_exact_version_comes_from_the_first_source_that_provides_it():
    second = _Source("1.0.0")
    assert resolve_release([_Source(), second, _Source("1.0.0")], "a/b", "1.0.0") == FoundInRegistry(V1, second)


def test_without_a_version_the_first_source_that_serves_the_dataset_gives_its_newest_version():
    first = _Source("1.0.0")
    sources = [_Source(), first, _Source("1.1.0")]
    assert resolve_release(sources, "a/b", None) == FoundInRegistry(V1, first)
    assert resolve_release(sources, "a/b", "") == FoundInRegistry(V1, first)


def test_without_a_version_a_source_that_does_not_serve_the_dataset_falls_through():
    second = _Source("1.0.0")
    listed = _Source(listed=("2.0.0",))
    assert resolve_release([listed, second, _Source("1.1.0")], "a/b", None) == FoundInRegistry(V1, second)
    # With no source that serves it, the access error of the first listing comes last.
    assert resolve_release([listed, _Source()], "a/b", None) == NotServed(V2, Access.CREDENTIALED, ())


def test_latest_is_the_highest_version_across_all_sources():
    remote = _Source("1.0.0", "1.1.0")
    assert resolve_release([_Source("1.0.0"), remote], "a/b", "latest") == FoundInRegistry(V11, remote)


def test_latest_comes_from_the_first_source_that_holds_the_highest_version():
    store = _Source("1.1.0")
    assert resolve_release([store, _Source("1.1.0")], "a/b", "latest") == FoundInRegistry(V11, store)


def test_a_release_that_is_not_served_falls_through_and_does_not_count_for_latest():
    connectors = _Source("1.0.0")
    assert resolve_release([_Source(listed=("2.0.0",)), connectors], "a/b", "latest") == FoundInRegistry(V1, connectors)
    assert resolve_release([_Source(listed=("2.0.0",))], "a/b", "2.0.0") == NotServed(V2, Access.CREDENTIALED, ())


def test_latest_asks_a_source_that_did_not_serve_a_higher_version_for_the_chosen_one():
    hosted = _Source("1.0.0", listed=("2.0.0",))
    assert resolve_release([hosted, _Source("1.0.0")], "a/b", "latest") == FoundInRegistry(V1, hosted)


def test_no_source_gives_not_found():
    assert resolve_release([_Source(), _Source()], "a/b", None) == NotFound()
    assert resolve_release([_Source("1.0.0")], "a/b", "2.0.0") == NotFound()


def test_an_error_of_a_source_stops_the_resolution():
    with pytest.raises(TimeNetRegistryError):
        resolve_release([_Broken(), _Source("1.0.0")], "a/b", "1.0.0")
    with pytest.raises(TimeNetRegistryError):
        resolve_release([_Source("1.0.0"), _Broken()], "a/b", "latest")


def test_installed_connectors_answer_with_the_builder_of_the_declared_version(monkeypatch):
    card = make_dataset().metadata
    backend = _Backend(card)
    install_backends(monkeypatch, backend)
    latest, exact = (
        InstalledConnectors().resolve(card.dataset_id, None),
        InstalledConnectors().resolve(card.dataset_id, V1),
    )
    assert isinstance(latest, FoundInConnector) and isinstance(exact, FoundInConnector)
    assert (latest.builder.backend, latest.version) == (exact.builder.backend, exact.version) == (backend, V1)


def test_installed_connectors_answer_not_found_for_another_version(monkeypatch):
    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card))
    assert InstalledConnectors().resolve(card.dataset_id, V2) == NotFound()
    assert InstalledConnectors().resolve("x/y", None) == NotFound()


def test_installed_connectors_without_a_package_answer_not_found(monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    assert InstalledConnectors().resolve("timenet/hello-world", None) == NotFound()
    assert "no connector package is installed" in str(InstalledConnectors())


def test_installed_connectors_list_their_cards(monkeypatch):
    first = _Cards(_card("z/last"), _card("a/first", "First"))
    second = _Cards(_card("a/first", "Second"), _card("m/middle"))
    install_backends(monkeypatch, first, second)
    # Sorted by dataset ID. The first entry point wins for an ID that two claim, as it does for a build.
    assert [(card.dataset_id, card.name) for card in InstalledConnectors().list_datasets()] == [
        ("a/first", "First"),
        ("m/middle", "Hello World"),
        ("z/last", "Hello World"),
    ]
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    assert InstalledConnectors().list_datasets() == []


def test_installed_connectors_leave_out_a_card_that_cannot_be_read(monkeypatch):
    class Broken(_Cards):
        def card(self, dataset_id):
            raise TimeNetInvalidCardError("the card cannot be read")

    # The second backend claims the dataset too, but the first entry point wins, as it does for a build.
    install_backends(monkeypatch, Broken(_card("a/broken")), _Cards(_card("a/broken"), _card("b/kept")))
    with pytest.warns(ConnectorCardWarning, match=r"^a/broken: the card of its installed connector cannot be read"):
        cards = InstalledConnectors().list_datasets()
    assert [card.dataset_id for card in cards] == ["b/kept"]


def test_an_error_of_the_dataset_ids_of_a_backend_propagates_from_the_cards(monkeypatch):
    class Broken(_Cards):
        def dataset_ids(self):
            raise TimeNetInvalidCardError("the cards cannot be listed")

    install_backends(monkeypatch, Broken())
    with pytest.raises(TimeNetInvalidCardError, match="cannot be listed"):
        InstalledConnectors().list_datasets()


def test_installed_connectors_search_their_cards(monkeypatch):
    cards = _Cards(_card("b/second"), _card("a/first"))
    install_backends(monkeypatch, cards)
    assert [card.dataset_id for card in InstalledConnectors().search(query="hello")] == ["a/first", "b/second"]
    assert [card.dataset_id for card in InstalledConnectors().search(query="hello", limit=1)] == ["a/first"]


def test_a_search_of_installed_connectors_leaves_out_a_card_that_does_not_match(monkeypatch):
    install_backends(monkeypatch, _Cards(_card("a/first")))
    assert InstalledConnectors().search(domain=Domain.CARDIOLOGY) == []


def test_a_type_filter_matches_no_card_of_installed_connectors(monkeypatch):
    install_backends(monkeypatch, _Cards(_card("a/first")))
    # Both filters read the schema of a manifest, and a card has none.
    assert InstalledConnectors().search(task=AnswerTask) == []
    assert InstalledConnectors().search(time_series_spec="sine") == []


def test_a_local_registry_answers_not_found_without_reading_every_manifest(tmp_path, monkeypatch):
    registry = LocalRegistry(tmp_path)
    registry.put_dataset(make_dataset())
    monkeypatch.setattr(LocalRegistry, "list_datasets", _fail)
    assert registry.resolve("test/unknown", None) == NotFound()
    assert registry.resolve("test/unknown", V1) == NotFound()
    assert registry.resolve("timenet/hello-world", V2) == NotFound()


def test_a_local_registry_rejects_an_invalid_dataset_id(tmp_path):
    with pytest.raises(TimeFValidationError):
        LocalRegistry(tmp_path).resolve("../evil", None)
