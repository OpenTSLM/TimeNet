import re
import sys

import pytest

from timenet_connectors import discovery
from timenet_connectors.discovery import connector_dir, requirements_for


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


# A requirement line starts with the distribution name; a version specifier, an extra, a marker, an
# inline comment or a direct-reference URL all follow it.
_REQUIREMENT_NAME = re.compile(r"^ *([A-Za-z0-9][A-Za-z0-9._-]*)", re.MULTILINE)


def _declared_names(requirements):
    # pip reads huggingface_hub and huggingface-hub as the same requirement, so compare PEP 503
    # normalized names instead of the spelling the file happens to use.
    return {re.sub(r"[-_.]+", "-", name).lower() for name in _REQUIREMENT_NAME.findall(requirements)}


@pytest.mark.parametrize(
    ("dataset_id", "expected"),
    [
        ("chengsenwang/tsqa", {"huggingface-hub"}),
        ("physionet/ptb-xl", {"wfdb", "boto3"}),
    ],
)
def test_connectors_declare_their_own_requirements(dataset_id, expected):
    path = requirements_for(dataset_id)
    assert path is not None
    assert expected <= _declared_names(path.read_text())
