import importlib
import subprocess
import sys

import pytest
import typer
from typer.testing import CliRunner

from timenet.cache import cached_datasets, raw_cache_size
from timenet.cli.app import _enum_list, app
from timenet.client import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import Access, Domain
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


runner = CliRunner()
# Import the app submodule directly so we can patch its module-global ``app`` when testing main()'s
# error handling.
_cli_module = importlib.import_module("timenet.cli.app")


@pytest.fixture
def registry_root(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "reg", dataset) as writer:
        writer.write()
    return tmp_path / "reg"


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "home" / "registry", dataset) as writer:
        writer.write()
    return tmp_path / "home"


def test_cache_info_lists_datasets(home):
    assert runner.invoke(app, ["cache", "info"]).exit_code == 0
    assert any(d.dataset_id == "timenet/hello-world" for d in cached_datasets())


def test_cache_info_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "empty"))
    assert runner.invoke(app, ["cache", "info"]).exit_code == 0
    assert cached_datasets() == []


def test_cache_info_reports_raw_cache_size(home):
    cache_dir = home / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "blob").write_bytes(b"x" * 10)
    assert runner.invoke(app, ["cache", "info"]).exit_code == 0
    assert raw_cache_size() == 10


def test_cache_clear_needs_confirmation(home):
    result = runner.invoke(app, ["cache", "clear"], input="n\n")
    assert result.exit_code != 0  # aborted
    assert (home / "registry").exists()


def test_cache_clear_all_removes_everything(home):
    result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
    assert result.exit_code == 0
    assert "Freed" in result.stdout
    assert not (home / "registry").exists()


def test_list(registry_root):
    assert runner.invoke(app, ["list", "--registry", str(registry_root)]).exit_code == 0
    assert any(d.dataset_id == "timenet/hello-world" for d in TimeNet(str(registry_root)).list())


def test_search_by_domain(registry_root):
    assert runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "general"]).exit_code == 0
    results = TimeNet(str(registry_root)).search(domain=[Domain("general")])
    assert any(d.dataset_id == "timenet/hello-world" for d in results)


def test_search_no_match(registry_root):
    assert runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "cardiology"]).exit_code == 0
    assert TimeNet(str(registry_root)).search(domain=[Domain("cardiology")]) == []


def test_info(registry_root):
    assert runner.invoke(app, ["info", "timenet/hello-world", "--registry", str(registry_root)]).exit_code == 0
    assert TimeNet(str(registry_root)).get("timenet/hello-world").dataset_id == "timenet/hello-world"


def _write(root, dataset):
    with TimeFWriter(root, dataset) as writer:
        writer.write()
    return root / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def test_info_lists_the_access_page_of_each_release_that_sets_one(tmp_path):
    root = tmp_path / "reg"
    parents = {}
    for name in ("a", "b"):
        parent = make_dataset()
        parent._metadata = parent.metadata.model_copy(
            update={
                "dataset_id": f"test/{name}",
                "access": Access.CREDENTIALED,
                "access_url": f"https://{name}.example",
            }
        )
        parents[ParentAlias(name)] = _write(root, parent)
    metadata = make_dataset().metadata.model_copy(
        update={
            "dataset_id": "test/child",
            "parents": {
                alias: DatasetRef(dataset_id=f"test/{alias}", version=make_dataset().metadata.dataset_version)
                for alias in parents
            },
        }
    )
    child = TimeFDataset(metadata=metadata)
    for path in parents.values():
        with TimeFReader(open_local_version(path)) as reader:
            child.import_from(reader)
    _write(root, child)

    result = runner.invoke(app, ["info", "test/child", "--registry", str(root)])

    assert result.exit_code == 0, result.output
    cells = [[cell.strip() for cell in line.split("│")[1:-1]] for line in result.output.splitlines() if "│" in line]
    start = cells.index(["access", "credentialed"])
    # The card of the child stays open. Its effective level comes from its parents, and each parent
    # names its own page.
    assert cells[start + 1 : start + 3] == [
        ["access_urls", "test/a@1.0.0 (credentialed): https://a.example"],
        ["", "test/b@1.0.0 (credentialed): https://b.example"],
    ]
    assert cells[start + 3][0] == "domains"


def test_download(registry_root, tmp_path):
    result = runner.invoke(
        app,
        ["download", "timenet/hello-world", "--registry", str(registry_root), "--storage", str(tmp_path / "store")],
    )
    assert result.exit_code == 0
    assert (tmp_path / "store" / "timenet/hello-world" / "1.0.0" / "manifest.json").exists()


def test_search_rejects_unknown_filter_value(registry_root):
    assert runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "bogus"]).exit_code == 2
    # The error message is a Rich-rendered panel, so assert it on the raising helper directly.
    with pytest.raises(typer.BadParameter, match="invalid --domain"):
        _enum_list(["bogus"], Domain, "--domain", [d.value for d in Domain])


def test_main_reports_expected_errors_without_traceback(monkeypatch, capsys):
    def raise_expected() -> None:
        raise TimeNetDatasetNotFoundError("no such dataset")

    monkeypatch.setattr(_cli_module, "app", raise_expected)
    monkeypatch.setattr(sys, "argv", ["timenet"])
    with pytest.raises(SystemExit) as exit_info:
        _cli_module.main()
    assert exit_info.value.code == 1
    assert "no such dataset" in capsys.readouterr().err


def test_importing_cli_package_stays_lazy():
    # The console entry point must load in a base install without the cli extra, so importing the
    # package must not pull in Typer/Rich (they live behind the lazily-loaded app module).
    code = (
        "import sys, timenet.cli; "
        "loaded = [m for m in sys.modules if m in ('typer', 'rich') or m.startswith(('typer.', 'rich.'))]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
