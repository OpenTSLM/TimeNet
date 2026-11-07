import importlib
import subprocess
import sys

import click
import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from timenet import builders
from timenet.cache import human_bytes, stored_releases
from timenet.cli.app import _enum_list, app
from timenet.config import make_cache_folder
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
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


def _downloads_size(cache):
    """Return the size of the raw downloads and the marker file in ``<cache>/downloads``."""
    return sum(path.stat().st_size for path in (cache / "downloads").rglob("*") if path.is_file())


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Put a release in the store and 10 bytes of raw downloads in the cache. Returns both paths."""
    store, cache = tmp_path / "store", tmp_path / "cache"
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(store))
    monkeypatch.setenv("TIMENET_CACHE", str(cache))
    LocalRegistry(store).put_dataset(make_dataset())
    make_cache_folder(cache / "downloads")
    raw = cache / "downloads/timenet/hello-world/raw.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"x" * 10)
    return store, cache


def test_local_info_shows_the_path_and_size_of_the_cache_and_the_local_registry(state):
    store, cache = state
    result = runner.invoke(app, ["local", "info"])
    assert result.exit_code == 0, result.output
    [release] = stored_releases(LocalRegistry(store))
    assert click.unstyle(result.stdout).splitlines() == [
        f"Cache {cache}: {human_bytes(_downloads_size(cache))}",
        f"Local registry {store}: 1 release, {human_bytes(release.size_bytes)}",
    ]


def test_local_cache_info_shows_the_path_and_size_of_the_cache(state):
    _, cache = state
    result = runner.invoke(app, ["local", "cache", "info"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"Cache {cache}: {human_bytes(_downloads_size(cache))}\n"


@pytest.mark.parametrize(
    ("command", "prompt"),
    [
        (["local", "clear"], "Delete the cache in {cache} and every release in {store}?"),
        (["local", "cache", "clear"], "Delete the raw source files and control databases in {cache}?"),
    ],
)
def test_each_clear_asks_first_and_names_the_paths(state, command, prompt):
    store, cache = state
    result = runner.invoke(app, command, input="n\n")
    assert result.exit_code != 0  # aborted
    assert f"{prompt.format(cache=cache, store=store)} [y/N]" in click.unstyle(result.output)
    assert (cache / "downloads").exists()
    assert LocalRegistry(store).exists("timenet/hello-world", "1.0.0")


def test_local_cache_clear_keeps_the_local_registry(state):
    store, cache = state
    size = human_bytes(_downloads_size(cache))
    result = runner.invoke(app, ["local", "cache", "clear"], input="y\n")
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 1 location(s)."
    assert not (cache / "downloads").exists()
    assert LocalRegistry(store).exists("timenet/hello-world", "1.0.0")


def test_local_cache_clear_warns_about_a_folder_without_the_marker_and_keeps_it(state):
    _, cache = state
    (cache / "control").mkdir()
    (cache / "control/notes.txt").write_text("not from TimeNet")
    result = runner.invoke(app, ["local", "cache", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    # The console wraps a long line, also in a path, so the check ignores whitespace.
    expected = f"Kept {cache / 'control'}: it has no CACHEDIR.TAG from TimeNet"
    assert "".join(expected.split()) in "".join(click.unstyle(result.stderr).split())
    assert click.unstyle(result.stdout).splitlines()[-1].endswith("from 1 location(s).")
    assert (cache / "control/notes.txt").exists()


def test_local_clear_deletes_the_cache_and_every_release(state):
    store, cache = state
    [release] = stored_releases(LocalRegistry(store))
    size = human_bytes(_downloads_size(cache) + release.size_bytes)
    result = runner.invoke(app, ["local", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 2 location(s)."
    assert stored_releases(LocalRegistry(store)) == []
    assert not (cache / "downloads").exists()


def test_list(registry_root):
    result = runner.invoke(app, ["list", "--registry", str(registry_root)])
    assert result.exit_code == 0, result.output
    assert "timenet/hello-world" in click.unstyle(result.output)


def test_search_by_domain(registry_root):
    result = runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "general"])
    assert result.exit_code == 0, result.output
    assert "timenet/hello-world" in click.unstyle(result.output)


def test_search_no_match(registry_root):
    result = runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "cardiology"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.output) == "No datasets match.\n"


def test_info(registry_root):
    result = runner.invoke(app, ["info", "timenet/hello-world", "--registry", str(registry_root)])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.output).startswith("timenet/hello-world")


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


def test_download_command_copies_into_the_store_and_then_has_nothing_to_do(registry_root, tmp_path, monkeypatch):
    # The default sources end with the installed connectors. Keep the real ones out of the resolution.
    monkeypatch.setattr(builders, "entry_points", lambda group: [])
    # The last line names a long path, so the console must not wrap it.
    monkeypatch.setattr(_cli_module.ui, "rich", Console(stderr=True, width=1000))
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    command = ["download", "timenet/hello-world", "--registry", str(registry_root)]
    target = tmp_path / "store" / "timenet/hello-world" / "1.0.0"

    copied = runner.invoke(app, command)
    again = runner.invoke(app, command)

    assert copied.exit_code == 0, copied.output
    assert copied.stdout == f"{target}\n"  # only the directory, for a script to capture
    assert (target / "manifest.json").exists()
    successes = [line for line in click.unstyle(copied.stderr).splitlines() if line.startswith("✅")]
    assert successes == ["✅ Copied 'timenet/hello-world' → 1.0.0"]
    assert again.exit_code == 0, again.output
    assert again.stdout == f"{target}\n"
    stored = tmp_path / "store"
    assert click.unstyle(again.stderr).splitlines() == [
        f"✅ Nothing to do: 'timenet/hello-world' 1.0.0 is already in local registry {stored}"
    ]


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
