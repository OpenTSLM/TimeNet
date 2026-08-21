import sys

import pytest

from timenet_connectors import discovery
from timenet_connectors.discovery import connector_dir, has_connector, requirements_for


def test_connector_dir_points_at_the_connector_package():
    path = connector_dir("physionet/ecg-qa-cot")
    assert path.name == "ecg_qa_cot"
    assert (path / "dataset.yaml").is_file()


def test_connector_dir_does_not_import_the_connector():
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    sys.modules.pop(leaf, None)
    sys.modules.pop("wfdb", None)

    connector_dir("physionet/ecg-qa-cot")

    assert leaf not in sys.modules
    assert "wfdb" not in sys.modules


def test_connector_dir_rejects_an_unknown_id():
    with pytest.raises(LookupError, match="no connector for 'nope/nothing'"):
        connector_dir("nope/nothing")


def test_connector_dir_rejects_a_bare_org():
    # An org namespace package has submodule_search_locations but no dataset.yaml, so it is not a
    # connector and must fail like any unknown id, not resolve to the org directory.
    with pytest.raises(LookupError, match="no connector for 'physionet'"):
        connector_dir("physionet")


def test_connector_dir_lists_the_known_ids_without_importing_a_connector(monkeypatch):
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    monkeypatch.delitem(sys.modules, leaf, raising=False)

    with pytest.raises(LookupError, match="physionet/ecg-qa-cot"):
        connector_dir("nope/nothing")

    assert leaf not in sys.modules


def test_requirements_for_returns_none_when_the_connector_declares_none():
    assert requirements_for("timenet/hello-world") is None


def test_requirements_for_returns_the_declared_file(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("wfdb>=4.1\n")
    monkeypatch.setattr(discovery, "connector_dir", lambda _: tmp_path)

    assert requirements_for("any/id") == tmp_path / "requirements.txt"


def test_has_connector_accepts_a_known_id():
    assert has_connector("physionet/ecg-qa-cot")


def test_has_connector_rejects_an_unknown_id():
    assert not has_connector("nope/nothing")


def test_has_connector_rejects_a_bare_org():
    # An org namespace package resolves to a spec with submodule_search_locations, but it holds no
    # dataset.yaml, so it is not a connector.
    assert not has_connector("physionet")
    assert not has_connector("timenet")
    assert not has_connector("chengsenwang")


def test_has_connector_does_not_import_any_connector_on_a_miss():
    # A miss must not import any connector: their dependencies are absent from this environment
    # by design, so importing one raises ModuleNotFoundError instead of answering the question.
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    sys.modules.pop(leaf, None)

    assert not has_connector("nope/nothing")

    assert leaf not in sys.modules
