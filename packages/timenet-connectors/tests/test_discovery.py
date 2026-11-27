from importlib.metadata import EntryPoint
import re
import sys
from uuid import uuid4

import pytest

from timenet.errors import TimeNetBuildError, TimeNetDuplicateDatasetError
from timenet_connectors import discovery
from timenet_connectors.discovery import (
    connector_dir,
    get_connector_package_name,
    known_ids,
    requirements_for,
    resolve,
)


def test_connector_dir_points_at_the_connector_package():
    path = connector_dir("physionet/ecg-qa-cot")
    assert path.name == "ecg_qa_cot"
    assert (path / "dataset.yaml").is_file()


def test_connector_dir_does_not_import_a_connector(monkeypatch):
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    monkeypatch.delitem(sys.modules, leaf, raising=False)
    monkeypatch.delitem(sys.modules, "wfdb", raising=False)

    connector_dir("physionet/ecg-qa-cot")
    with pytest.raises(LookupError, match=r"no connector for 'nope/nothing'.*physionet/ecg-qa-cot"):
        connector_dir("nope/nothing")

    assert leaf not in sys.modules
    assert "wfdb" not in sys.modules


def test_connector_dir_rejects_a_bare_org():
    with pytest.raises(LookupError, match="no connector for 'physionet'"):
        connector_dir("physionet")


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


_CONNECTOR_INIT = """
class _Metadata:
    dataset_id = {dataset_id!r}


class CONNECTOR:
    def metadata(self):
        return _Metadata()
"""


def _add_datasets_root(tmp_path, monkeypatch, *dataset_ids):
    """Install a package on sys.path that adds a datasets root through the entry-point group."""
    package = f"acme_{uuid4().hex[:8]}"
    root = tmp_path / package / "datasets"
    root.mkdir(parents=True)
    (tmp_path / package / "__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    for dataset_id in dataset_ids:
        directory = root
        for part in dataset_id.replace("-", "_").split("/"):
            directory /= part
            directory.mkdir(exist_ok=True)
            (directory / "__init__.py").write_text("")
        (directory / "__init__.py").write_text(_CONNECTOR_INIT.format(dataset_id=dataset_id))
        (directory / "dataset.yaml").write_text(f"dataset_id: {dataset_id}\n")
    dist_info = tmp_path / "acme_datasets-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: acme-datasets\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text(f"[timenet_connectors.datasets]\nroot = {package}.datasets\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return root


def test_a_registered_root_adds_connectors(tmp_path, monkeypatch):
    root = _add_datasets_root(tmp_path, monkeypatch, "acme/widget", "physionet/acme-extra")

    assert {"acme/widget", "physionet/acme-extra", "physionet/ecg-qa-cot"} <= set(known_ids())
    assert connector_dir("acme/widget") == root / "acme" / "widget"
    assert resolve("acme/widget")().metadata().dataset_id == "acme/widget"
    assert get_connector_package_name("acme/widget") == "acme-datasets"
    assert connector_dir("physionet/acme-extra") == root / "physionet" / "acme_extra"
    assert connector_dir("physionet/ecg-qa-cot").parts[-4:-2] == ("timenet_connectors", "datasets")
    assert get_connector_package_name("timenet/hello-world") is None


def test_a_root_listed_twice_on_the_path_counts_once(tmp_path, monkeypatch):
    package = f"nsacme_{uuid4().hex[:8]}"
    directory = tmp_path / package / "datasets" / "acme" / "gadget"
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("")
    (directory / "dataset.yaml").write_text("dataset_id: acme/gadget\n")
    dist_info = tmp_path / "nsacme_datasets-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: nsacme-datasets\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text(f"[timenet_connectors.datasets]\nroot = {package}.datasets\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))

    assert connector_dir("acme/gadget") == directory


def test_a_dataset_id_in_two_roots_fails_only_for_that_dataset(tmp_path, monkeypatch):
    root = _add_datasets_root(tmp_path, monkeypatch, "acme/widget", "timenet/hello-world")

    assert "timenet/hello-world" in known_ids()
    with pytest.raises(TimeNetDuplicateDatasetError, match=r"more than one connector declares 'timenet/hello-world'"):
        connector_dir("timenet/hello-world")
    assert connector_dir("acme/widget") == root / "acme" / "widget"
    assert connector_dir("physionet/ecg-qa-cot").parts[-4:-2] == ("timenet_connectors", "datasets")


def test_an_entry_point_that_cannot_be_imported_names_its_package(tmp_path, monkeypatch):
    dist_info = tmp_path / "acme_datasets-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: acme-datasets\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text("[timenet_connectors.datasets]\nroot = acme_missing.datasets\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(TimeNetBuildError, match=r"root = acme_missing\.datasets of acme-datasets"):
        connector_dir("timenet/hello-world")


def test_an_entry_point_without_a_package_names_itself(monkeypatch):
    entry = EntryPoint("root", "acme.datasets", "timenet_connectors.datasets")
    monkeypatch.setattr(discovery, "entry_points", lambda group: [entry])

    with pytest.raises(TimeNetBuildError, match=r"root = acme\.datasets names no installed package"):
        connector_dir("timenet/hello-world")
