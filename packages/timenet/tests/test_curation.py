from pathlib import Path

from timenet import curation
from timenet.curation import find_curator


class _Backend:
    def __init__(self, known):
        self._known = known

    def knows(self, dataset_id):
        return dataset_id == self._known

    def build(self, dataset_id, root, *, force=False):
        return Path(root) / dataset_id


class _Entry:
    def __init__(self, backend):
        self._backend = backend

    def load(self):
        return lambda: self._backend


def test_find_curator_returns_none_when_nothing_claims_the_id(monkeypatch):
    monkeypatch.setattr(curation, "entry_points", lambda group: [_Entry(_Backend("a/b"))])
    assert find_curator("x/y") is None


def test_find_curator_returns_the_backend_that_claims_the_id(monkeypatch):
    wanted = _Backend("x/y")
    monkeypatch.setattr(curation, "entry_points", lambda group: [_Entry(_Backend("a/b")), _Entry(wanted)])
    assert find_curator("x/y") is wanted


def test_find_curator_returns_none_with_no_registered_backends(monkeypatch):
    monkeypatch.setattr(curation, "entry_points", lambda group: [])
    assert find_curator("x/y") is None
