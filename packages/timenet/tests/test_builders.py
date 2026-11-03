from timenet import builders
from timenet.builders import exact_builder, find_builder, find_card
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

    def build_prepared(self, release, store, *, force, pinned_by, options):
        raise AssertionError(f"the lookup does not build {release}")


def test_find_builder_returns_none_when_nothing_claims_the_id(monkeypatch):
    install_backends(monkeypatch, _Backend("a/b"))
    assert find_builder("x/y") is None


def test_find_builder_returns_the_first_backend_whose_dataset_ids_hold_the_id(monkeypatch):
    wanted = _Backend("a/c", "x/y")
    install_backends(monkeypatch, _Backend("a/b"), wanted, _Backend("x/y"))
    assert find_builder("x/y") is wanted


def test_find_builder_asks_dataset_ids(monkeypatch):
    class _KnowsEverything(_Backend):
        def knows(self, dataset_id):
            return True

    # The first backend says yes to every ID through knows(), but its dataset IDs do not hold this one.
    wanted = _Backend("x/y")
    install_backends(monkeypatch, _KnowsEverything("a/b"), wanted)
    assert find_builder("x/y") is wanted


def test_find_builder_returns_none_with_no_registered_backends(monkeypatch):
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    assert find_builder("x/y") is None


def test_find_card_returns_the_card_of_the_backend_that_claims_the_id(monkeypatch):
    card = make_dataset().metadata
    backends = [_Backend("a/b"), _Backend(card.dataset_id, card=card)]
    install_backends(monkeypatch, *backends)
    assert find_card(card.dataset_id) is card
    assert find_card("x/y") is None


def test_exact_builder_returns_the_backend_only_for_the_version_of_its_card(monkeypatch):
    card = make_dataset().metadata
    backend = _Backend(card.dataset_id, card=card)
    install_backends(monkeypatch, backend)
    assert exact_builder(DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)) is backend
    assert exact_builder(DatasetRef.parse(f"{card.dataset_id}@2.0.0")) is None
    assert exact_builder(DatasetRef.parse("x/y@1.0.0")) is None
