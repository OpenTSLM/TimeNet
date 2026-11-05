import importlib
import shutil
import subprocess
import sys

import click
import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from timenet.cache import human_bytes, stored_releases
from timenet.cli.app import _enum_list, app
from timenet.client import TimeNet
from timenet.config import make_cache_folder
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetInvalidCardError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, Domain
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.writer import TimeFWriter


runner = CliRunner()
HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")
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


def _table_cells(output: str) -> list[list[str]]:
    """Return the cells of each table row in ``output``, without the borders."""
    return [[cell.strip() for cell in line.split("│")[1:-1]] for line in output.splitlines() if "│" in line]


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


def test_local_info_and_local_cache_info_show_the_paths_and_sizes(state):
    store, cache = state
    result = runner.invoke(app, ["local", "info"])
    assert result.exit_code == 0, result.output
    [release] = stored_releases(LocalRegistry(store))
    cache_line = f"Cache {cache}: {human_bytes(_downloads_size(cache))}"
    assert click.unstyle(result.stdout).splitlines() == [
        cache_line,
        f"Local registry {store}: 1 release, {human_bytes(release.size_bytes)}",
    ]
    result = runner.invoke(app, ["local", "cache", "info"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"{cache_line}\n"


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


def test_local_clear_deletes_no_release_when_the_cache_check_fails(state, monkeypatch):
    _, cache = state
    # The local registry lies in the downloads folder that the clear deletes.
    inside = LocalRegistry(cache / "downloads" / "store")
    inside.put_dataset(make_dataset())
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(inside.root))
    result = runner.invoke(app, ["local", "clear", "--yes"])
    assert isinstance(result.exception, TimeFValidationError)
    assert inside.exists("timenet/hello-world", "1.0.0")


def test_the_local_commands_take_the_local_registry_flag(tmp_path):
    # The variable of the autouse fixture names an empty store, so the release comes from the flag.
    store = tmp_path / "store"
    LocalRegistry(store).put_dataset(make_dataset())
    result = runner.invoke(app, ["local", "info", "--local-registry", str(store)])
    assert result.exit_code == 0, result.output
    assert f"Local registry {store}: 1 release" in click.unstyle(result.stdout)
    result = runner.invoke(app, ["local", "clear", "--yes", "--local-registry", str(store)])
    assert result.exit_code == 0, result.output
    assert stored_releases(LocalRegistry(store)) == []


def _flags(tmp_path, remote, *, auto_build=False):
    """Return the flags of a list of sources: an empty store, ``remote``, and the connectors or not."""
    build = "--auto-build" if auto_build else "--no-auto-build"
    return ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(remote), build]


def _first_cells(output):
    """Return the first cell of each body row of the table in ``output``."""
    return [row[0] for row in _table_cells(output)]


def test_list(registry_root, tmp_path):
    result = runner.invoke(app, ["list", *_flags(tmp_path, registry_root)])
    assert result.exit_code == 0, result.output
    assert _first_cells(result.stdout) == ["timenet/hello-world"]


def test_search_by_domain(registry_root, tmp_path):
    result = runner.invoke(app, ["search", "--domain", "general", *_flags(tmp_path, registry_root)])
    assert result.exit_code == 0, result.output
    assert _first_cells(result.stdout) == ["timenet/hello-world"]


def test_search_no_match(registry_root, tmp_path):
    result = runner.invoke(app, ["search", "--domain", "cardiology", *_flags(tmp_path, registry_root)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "No datasets match.\n"


class _Backend:
    """A builder backend with one connector, whose card declares ``test/gated@1.0.0``."""

    def dataset_ids(self):
        return ["test/gated"]

    def card(self, dataset_id):
        return make_dataset().metadata.model_copy(
            update={
                "dataset_id": dataset_id,
                "access": Access.CREDENTIALED,
                "access_url": "https://gated.example",
                "citation": "A citation.",
            }
        )


def test_list_shows_a_warning_as_a_line_without_a_file_and_a_line_number(tmp_path, monkeypatch):
    class _Broken(_Backend):
        def card(self, dataset_id):
            raise TimeNetInvalidCardError("the card cannot be read")

    install_backends(monkeypatch, _Broken())
    monkeypatch.setattr(_cli_module.ui, "rich", Console(stderr=True, width=1000))
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["list", *flags])

    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stderr).splitlines() == [
        "⚠️ test/gated: the card of its installed connector cannot be read, so the listing leaves it out: "
        "the card cannot be read"
    ]


def test_info_shows_the_card_of_a_dataset_that_only_a_connector_has(tmp_path, monkeypatch):
    install_backends(monkeypatch, _Backend())
    store = tmp_path / "store"
    flags = ["--local-registry", str(store), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/gated", *flags])

    assert result.exit_code == 0, result.output
    cells = _table_cells(result.stdout)
    card = _Backend().card("test/gated")
    assert cells == [
        ["version", "1.0.0"],
        ["name", card.name],
        ["license", str(card.license)],
        ["access", "credentialed"],
        ["access_urls", "test/gated@1.0.0 (credentialed): https://gated.example"],
        ["citation", "A citation."],
        ["domains", "general"],
    ]
    assert result.stdout.splitlines()[-1] == (
        "The schema and the counts appear after a build: "
        f"`timenet-build build test/gated --local-registry {store} --remote-registry {tmp_path / 'empty'}`."
    )
    # The command reads the card and builds nothing.
    assert not store.exists()


def test_info_of_an_unbuilt_child_says_that_its_access_leaves_out_its_parents(tmp_path, monkeypatch):
    class _Child(_Backend):
        def dataset_ids(self):
            return ["test/child"]

        def card(self, dataset_id):
            parents = {"a": "test/a@1.0.0", "b": "test/b@1.0.0"}
            return make_dataset(dataset_id=dataset_id, parents=parents).metadata

    install_backends(monkeypatch, _Child())
    # The access row is long, so the console must not wrap it.
    monkeypatch.setattr(_cli_module, "console", Console(width=1000))
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/child", *flags])

    assert result.exit_code == 0, result.output
    access = "open (this card only: the access of its 2 parents is known after a build)"
    assert ["access", access] in _table_cells(click.unstyle(result.stdout))


def test_info_of_a_dataset_that_only_a_connector_has_reads_its_card_once(tmp_path, monkeypatch):
    reads = []

    class _Counting(_Backend):
        def card(self, dataset_id):
            reads.append(dataset_id)
            return super().card(dataset_id)

    install_backends(monkeypatch, _Counting())
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/gated", *flags])

    assert result.exit_code == 0, result.output
    # Each resolution of the release reads the card, so one read means one resolution.
    assert reads == ["test/gated"]


def test_list_shows_the_card_of_a_dataset_that_only_a_connector_has(registry_root, tmp_path, monkeypatch):
    install_backends(monkeypatch, _Backend())

    result = runner.invoke(app, ["list", *_flags(tmp_path, registry_root, auto_build=True)])

    assert result.exit_code == 0, result.output
    assert _first_cells(result.stdout) == ["test/gated", "timenet/hello-world"]
    result = runner.invoke(app, ["search", "--query", "hello", *_flags(tmp_path, registry_root, auto_build=True)])
    assert _first_cells(result.stdout) == ["test/gated", "timenet/hello-world"]
    # Without the connectors in the list, only the registry answers.
    result = runner.invoke(app, ["list", *_flags(tmp_path, registry_root)])
    assert _first_cells(result.stdout) == ["timenet/hello-world"]


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

    result = runner.invoke(app, ["info", "test/child", "--remote-registry", str(root), "--no-auto-build"])

    assert result.exit_code == 0, result.output
    cells = _table_cells(click.unstyle(result.output))
    start = cells.index(["access", "credentialed"])
    # The card of the child stays open. Its effective level comes from its parents, and each parent
    # names its own page.
    assert cells[start + 1 : start + 3] == [
        ["access_urls", "test/a@1.0.0 (credentialed): https://a.example"],
        ["", "test/b@1.0.0 (credentialed): https://b.example"],
    ]
    assert cells[start + 3][0] == "domains"
    assert ["records", str(len(child.records))] in cells


def test_download_command_copies_into_the_store_and_then_has_nothing_to_do(registry_root, tmp_path, monkeypatch):
    # The last line names a long path, so the console must not wrap it.
    monkeypatch.setattr(_cli_module.ui, "rich", Console(stderr=True, width=1000))
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    command = ["download", "timenet/hello-world", "--remote-registry", str(registry_root)]
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


def test_download_command_with_force_copies_a_held_release_again(registry_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    command = ["download", "timenet/hello-world", "--remote-registry", str(registry_root)]
    assert runner.invoke(app, command).exit_code == 0

    forced = runner.invoke(app, [*command, "--force"])

    assert forced.exit_code == 0, forced.output
    successes = [line for line in click.unstyle(forced.stderr).splitlines() if line.startswith("✅")]
    assert successes == ["✅ Copied 'timenet/hello-world' → 1.0.0"]


def test_each_command_builds_the_list_from_its_flags(registry_root, tmp_path):
    # The variables of the autouse fixture name an empty store and an empty registry, so each result
    # comes from the flags.
    store = tmp_path / "store"
    flags = ["--local-registry", str(store), "--remote-registry", str(registry_root), "--no-auto-build"]
    for command in (["list"], ["search", "--domain", "general"], ["info", "timenet/hello-world"]):
        result = runner.invoke(app, [*command, *flags])
        assert result.exit_code == 0, result.output
        assert "timenet/hello-world" in result.stdout
    result = runner.invoke(app, ["download", "timenet/hello-world", *flags])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{store / 'timenet/hello-world/1.0.0'}\n"


def test_publish_copies_a_release_from_the_store(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    dest = tmp_path / "dest"
    result = runner.invoke(
        app, ["publish", "timenet/hello-world", "--dest", str(dest), "--local-registry", str(store.root)]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "timenet/hello-world@1.0.0"
    assert LocalRegistry(dest).exists("timenet/hello-world", "1.0.0")
    assert "📥 Copying 'timenet/hello-world'…" in result.stderr
    assert "✅ Published timenet/hello-world@1.0.0 to" in result.stderr


def test_publish_rejects_a_source_that_cannot_be_opened(tmp_path):
    result = runner.invoke(app, ["publish", "x/y", "--dest", str(tmp_path / "dest"), "--source", "foo://x"])
    assert result.exit_code == 1
    assert isinstance(result.exception, TimeNetRegistryError)
    assert str(result.exception).startswith("--source 'foo://x' has an unsupported registry scheme.")


def test_publish_to_an_empty_dest_fails_and_writes_nothing_into_the_working_directory(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    argv = ["publish", "timenet/hello-world", "--dest", "", "--local-registry", str(store.root)]
    result = runner.invoke(app, argv)
    assert isinstance(result.exception, TimeNetRegistryError)
    assert str(result.exception).startswith("dest is blank")
    assert list(cwd.iterdir()) == []


def test_publish_needs_dest(tmp_path):
    result = runner.invoke(app, ["publish", "timenet/hello-world"])
    assert result.exit_code == 2
    # Rich colours the usage error on a CI runner, which splits the option name with escape codes.
    assert "--dest" in click.unstyle(result.output)


def test_publish_of_a_release_that_the_store_lacks_names_the_fix(tmp_path):
    store, other, dest = tmp_path / "store", tmp_path / "other", str(tmp_path / "dest")
    result = runner.invoke(app, ["publish", "timenet/hello-world", "--dest", dest, "--local-registry", str(store)])
    assert type(result.exception) is TimeNetDatasetNotFoundError
    assert str(result.exception) == (
        f"timenet/hello-world is not in local registry {store}. "
        f"Build it into the local registry with `timenet-build build timenet/hello-world --local-registry {store}`, "
        f"or copy it there with `timenet download timenet/hello-world --local-registry {store}`."
    )
    # The hint keeps the version, and names no store without --local-registry.
    result = runner.invoke(app, ["publish", "timenet/hello-world@1.0.0", "--dest", dest])
    assert str(result.exception).endswith(
        "Build it into the local registry with `timenet-build build timenet/hello-world@1.0.0`, "
        "or copy it there with `timenet download timenet/hello-world@1.0.0`."
    )
    # A registry that --source names is not the store, so the error names only that registry.
    result = runner.invoke(app, ["publish", "timenet/hello-world", "--dest", dest, "--source", str(other)])
    assert str(result.exception) == f"timenet/hello-world is not in local registry {other}."


def test_publish_adds_no_hint_to_a_payload_source_that_the_store_lacks(tmp_path):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    child = TimeFDataset(
        metadata=make_dataset().metadata.model_copy(
            update={"dataset_id": "test/child", "parents": {ParentAlias("hello"): HELLO}}
        )
    )
    # The child reads the values of one Record of the parent, so the parent is its payload source.
    with TimeNet(sources=[store]).open(str(HELLO)) as reader:
        child.import_from(reader, alias=ParentAlias("hello"), records=[make_dataset().records[0].id])
        store.put_dataset(child)
    shutil.rmtree(store.root / "timenet")
    dest = tmp_path / "dest"

    result = runner.invoke(app, ["publish", "test/child", "--dest", str(dest), "--local-registry", str(store.root)])

    # The error of the payload source names its own registries, and the hint for the release does not apply.
    assert type(result.exception) is TimeNetMissingParentError
    assert str(result.exception).startswith(f"payload source {HELLO} of test/child@1.0.0 is not in ")
    assert "timenet-build" not in str(result.exception)


@pytest.mark.parametrize(("flag", "variable"), [("--auto-build", "false"), ("--no-auto-build", "true")])
def test_the_auto_build_flag_beats_its_variable(monkeypatch, flag, variable):
    monkeypatch.setenv("TIMENET_AUTO_BUILD", variable)
    result = runner.invoke(app, ["download", "x/y", flag])
    # The error names each source of the list.
    assert isinstance(result.exception, TimeNetDatasetNotFoundError)
    assert ("installed connectors" in str(result.exception)) == (flag == "--auto-build")


# The commands that read the list of sources share one path, so download stands for them.
_EMPTY_FLAGS = [
    ["download", "x/y", "--local-registry", ""],
    ["local", "clear", "--yes", "--local-registry", ""],
]


@pytest.mark.parametrize("argv", _EMPTY_FLAGS, ids=lambda argv: " ".join(argv[:-1]))
def test_an_empty_flag_is_an_error(tmp_path, monkeypatch, argv):
    # An empty value must not mean the current directory: a build would land there, and
    # `local clear` would delete its releases.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, argv)
    assert isinstance(result.exception, TimeFValidationError)
    assert f"must not be empty (also as {argv[-2]})" in str(result.exception)


def test_search_rejects_unknown_filter_value(registry_root, tmp_path):
    assert runner.invoke(app, ["search", "--domain", "bogus", *_flags(tmp_path, registry_root)]).exit_code == 2
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
