from pathlib import Path

from timenet import builders
from timenet.builders import find_builder, find_card
from timenet.testing import make_dataset


class _Backend:
    def __init__(self, known, card=None):
        self._known = known
        self._card = card

    def knows(self, dataset_id):
        return dataset_id == self._known

    def card(self, dataset_id):
        return self._card

    def build(self, dataset_id, root, *, force=False):
        return Path(root) / dataset_id


class _Entry:
    def __init__(self, backend):
        self._backend = backend

    def load(self):
        return lambda: self._backend


def test_find_builder_returns_none_when_nothing_claims_the_id(monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [_Entry(_Backend("a/b"))])
    assert find_builder("x/y") is None


def test_find_builder_returns_the_backend_that_claims_the_id(monkeypatch):
    wanted = _Backend("x/y")
    monkeypatch.setattr(builders, "entry_points", lambda group: [_Entry(_Backend("a/b")), _Entry(wanted)])
    assert find_builder("x/y") is wanted


def test_find_builder_returns_none_with_no_registered_backends(monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    assert find_builder("x/y") is None


def test_find_card_returns_the_card_of_the_backend_that_claims_the_id(monkeypatch):
    card = make_dataset().metadata
    backends = [_Backend("a/b"), _Backend(card.dataset_id, card)]
    monkeypatch.setattr(builders, "entry_points", lambda group: [_Entry(backend) for backend in backends])
    assert find_card(card.dataset_id) is card
    assert find_card("x/y") is None
