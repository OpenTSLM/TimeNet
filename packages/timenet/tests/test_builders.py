from types import SimpleNamespace

import pytest

from timenet import builders
from timenet.builders import declared, installed_builder
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
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
    card = make_dataset().metadata
    a_b, x_y = _Backend("a/b", card=card), _Backend("x/y", card=card)
    # install_backends() gives entry points without a distribution, so this test makes its own.
    entries = [
        SimpleNamespace(dist=None, load=lambda: lambda: a_b),
        SimpleNamespace(dist=SimpleNamespace(name="pkg", version="1.2.3"), load=lambda: lambda: x_y),
    ]
    monkeypatch.setattr(builders, "entry_points", lambda group: entries)
    packaged, unpackaged = installed_builder("x/y"), installed_builder("a/b")
    assert packaged is not None and unpackaged is not None
    assert packaged.package == "pkg 1.2.3"
    # No distribution declares the first entry point, so the class of its backend names it.
    assert unpackaged.package == f"{_Backend.__module__}._Backend"
