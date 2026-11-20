from types import SimpleNamespace

import pytest

from timenet import builders
from timenet.builders import declared, installed_builder
from timenet.errors import TimeNetBuildError, TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.testing import install_backends, make_dataset
from timenet.types.references import DatasetRef


class _Backend:
    def __init__(self, *known, card=None):
        self._known = known
        self._card = card

    def dataset_ids(self):
        return iter(self._known)

    def card(self, dataset_id):
        return self._card

    def build(self, release, store, *, force, pinned_by, options):
        raise AssertionError(f"the lookup does not build {release}")


def test_installed_builder_returns_the_first_backend_whose_dataset_ids_hold_the_id(monkeypatch):
    card = make_dataset().metadata
    wanted = _Backend("a/c", card.dataset_id, card=card)
    install_backends(monkeypatch, _Backend("a/b"), wanted, _Backend(card.dataset_id, card=card))
    found = installed_builder(card.dataset_id)
    assert found is not None
    assert (found.backend, found.card) == (wanted, card)
    assert installed_builder("x/y") is None


def test_installed_builder_reads_no_backend_after_the_first_that_lists_the_id(monkeypatch):
    class _Unread(_Backend):
        def dataset_ids(self):
            raise AssertionError("the lookup reads no backend after the match")

    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card.dataset_id, card=card), _Unread())
    found = installed_builder(card.dataset_id)
    assert found is not None
    assert found.card == card


def test_declared_returns_the_release_of_the_card_and_refuses_another_version(monkeypatch):
    card = make_dataset().metadata
    install_backends(monkeypatch, _Backend(card.dataset_id, card=card))
    release = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    assert declared(card.dataset_id).release == release
    assert declared(card.dataset_id, card.dataset_version).release == release
    with pytest.raises(TimeNetVersionUnavailableError) as raised:
        declared(card.dataset_id, "2.0.0")
    assert (raised.value.release, raised.value.declared) == (
        DatasetRef.parse(f"{card.dataset_id}@2.0.0"),
        card.dataset_version,
    )
    with pytest.raises(TimeNetDatasetNotFoundError, match="no installed connector builds 'x/y'"):
        declared("x/y")


def test_an_installed_builder_names_the_distribution_of_the_backend(monkeypatch):
    install_backends(monkeypatch, _Backend("x/y", card=make_dataset().metadata))
    builder = installed_builder("x/y")
    assert builder is not None
    assert builder.package == "timenet-testing 0"


def test_a_backend_without_a_distribution_gives_no_installed_builder(monkeypatch):
    backend = _Backend("a/b", card=make_dataset().metadata)
    entry = SimpleNamespace(dist=None, load=lambda: lambda: backend)
    monkeypatch.setattr(builders, "entry_points", lambda group: [entry])
    with pytest.raises(TimeNetBuildError, match=f"the entry point of {__name__}._Backend names no installed package"):
        installed_builder("a/b")
