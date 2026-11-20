import re
import sys
from uuid import uuid4

import pytest

from timenet.errors import ConnectorCardWarning
from timenet_connectors import discovery
from timenet_connectors.discovery import (
    available,
    connector_dir,
    known_ids,
    owning_distribution,
    requirements_for,
    resolve,
)


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


_CONNECTOR_INIT = """
class _Metadata:
    dataset_id = {dataset_id!r}


class CONNECTOR:
    def metadata(self):
        return _Metadata()
"""


def _write_dist_info(tmp_path, distribution, value):
    dist_info = tmp_path / f"{distribution.replace('-', '_')}-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text(f"[timenet_connectors.datasets]\nroot = {value}\n")


@pytest.fixture
def add_datasets_root(tmp_path, monkeypatch):
    """Install a package on sys.path that adds a datasets root through the entry-point group."""
    created = []

    def add(*dataset_ids, distribution="acme-datasets"):
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
        _write_dist_info(tmp_path, distribution, f"{package}.datasets")
        created.append(package)
        return root

    monkeypatch.syspath_prepend(str(tmp_path))
    yield add
    for name in list(sys.modules):
        if name.split(".")[0] in created:
            del sys.modules[name]


def _add_entry_point(tmp_path, monkeypatch, value):
    _write_dist_info(tmp_path, "broken-datasets", value)
    monkeypatch.syspath_prepend(str(tmp_path))


def test_an_entry_point_with_a_missing_package_is_skipped_with_a_warning(tmp_path, monkeypatch):
    _add_entry_point(tmp_path, monkeypatch, "stale_gone.datasets")

    with pytest.warns(ConnectorCardWarning, match="stale_gone.datasets"):
        assert "timenet/hello-world" in known_ids()
    with pytest.warns(ConnectorCardWarning, match="stale_gone.datasets"):
        assert connector_dir("timenet/hello-world").name == "hello_world"


def test_an_entry_point_that_names_the_built_in_root_is_ignored(tmp_path, monkeypatch):
    _add_entry_point(tmp_path, monkeypatch, "timenet_connectors.datasets")

    assert owning_distribution("timenet/hello-world") is None


def test_a_registered_root_adds_its_connectors(add_datasets_root):
    root = add_datasets_root("acme/widget")

    assert {"acme/widget", "timenet/hello-world"} <= set(known_ids())
    assert connector_dir("acme/widget") == root / "acme" / "widget"
    assert resolve("acme/widget")().metadata().dataset_id == "acme/widget"


def test_available_walks_a_registered_root(add_datasets_root):
    add_datasets_root("acme/widget")

    assert "acme/widget" in available()


def test_owning_distribution_names_the_package_that_registers_the_root(add_datasets_root):
    add_datasets_root("acme/widget", distribution="acme-datasets")

    assert owning_distribution("acme/widget") == "acme-datasets"
    assert owning_distribution("timenet/hello-world") is None


def test_two_roots_can_share_an_org(add_datasets_root):
    root = add_datasets_root("physionet/acme-extra")

    assert {"physionet/acme-extra", "physionet/ecg-qa-cot"} <= set(known_ids())
    assert connector_dir("physionet/acme-extra") == root / "physionet" / "acme_extra"
    assert connector_dir("physionet/ecg-qa-cot").parts[-4:-2] == ("timenet_connectors", "datasets")


def test_an_id_in_two_roots_is_skipped_and_cannot_be_resolved(add_datasets_root):
    add_datasets_root("timenet/hello-world")

    with pytest.warns(ConnectorCardWarning, match="timenet/hello-world"):
        assert "timenet/hello-world" not in known_ids()
    with pytest.raises(LookupError, match="more than one"):
        connector_dir("timenet/hello-world")
    with pytest.raises(LookupError, match="more than one"):
        resolve("timenet/hello-world")
