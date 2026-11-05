import importlib
import shutil
import subprocess
import sys

from _demo_child import _child_of_the_demo
import click
import httpx
import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from timenet import builders as builders_module, client as client_module
from timenet.cache import human_bytes, stored_releases
from timenet.cli import hints as hints_module
from timenet.cli.app import _enum_list, app
from timenet.client import TimeNet
from timenet.config import make_cache_folder
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeNetDatasetNotFoundError,
    TimeNetManifestError,
    TimeNetMissingParentError,
    TimeNetNotBuiltError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, RemoteRegistry, open_local_version
from timenet.registry.origin import record_origin
from timenet.testing import install_backends, make_dataset
from timenet.types import Access, Domain, Version
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


def _put_demo(store, dataset_id, origin=None, **update):
    """Put the demo dataset into the store under ``dataset_id``, with a provenance record of ``origin``.

    Returns:
        The release.
    """
    dataset = make_dataset()
    dataset._metadata = dataset.metadata.model_copy(update={"dataset_id": dataset_id, **update})
    registry = LocalRegistry(store)
    release = registry.put_dataset(dataset)
    if origin is not None:
        record_origin(registry, release, origin, f"https://{origin}.example.org")
    return release


def _stderr_has(result, text):
    """Return whether stderr holds ``text``.

    The console wraps a long line, also in a path, so the check ignores whitespace.
    """
    return "".join(text.split()) in "".join(click.unstyle(result.stderr).split())


def test_local_info_shows_the_path_and_size_of_the_cache_the_downloads_and_the_builds(state):
    store, cache = state
    _put_demo(store, "demo/copy", "copy")
    result = runner.invoke(app, ["local", "info"])
    assert result.exit_code == 0, result.output
    copy, build = stored_releases(LocalRegistry(store))
    assert click.unstyle(result.stdout).splitlines() == [
        f"Cache {cache}: {human_bytes(_downloads_size(cache))}",
        f"Downloads in {store}: 1 release, {human_bytes(copy.size_bytes)}",
        f"Builds in {store}: 1 release, {human_bytes(build.size_bytes)}",
    ]


def test_local_cache_info_shows_the_path_and_size_of_the_cache(state):
    _, cache = state
    result = runner.invoke(app, ["local", "cache", "info"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"Cache {cache}: {human_bytes(_downloads_size(cache))}\n"


@pytest.mark.parametrize(
    ("command", "rows"),
    [
        (["local", "downloads", "info"], [["demo/copy", "1.0.0", "https://copy.example.org"]]),
        (
            ["local", "builds", "info"],
            [["demo/built", "1.0.0", "https://build.example.org"], ["timenet/hello-world", "1.0.0", "-"]],
        ),
    ],
)
def test_downloads_info_and_builds_info_list_the_releases_of_their_kind(state, command, rows):
    store, _ = state
    _put_demo(store, "demo/copy", "copy")
    _put_demo(store, "demo/built", "build")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    lines = click.unstyle(result.stdout).splitlines()
    cells = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if "│" in line]
    sizes = {item.release.dataset_id: item.size_bytes for item in stored_releases(LocalRegistry(store))}
    total = sum(sizes[row[0]] for row in rows)
    # The header uses another border, so the cells are the body rows. A build without a record has no source.
    assert cells == [*([*row, human_bytes(sizes[row[0]])] for row in rows), ["total", "", "", human_bytes(total)]]


def test_downloads_info_of_a_local_registry_without_downloads(state):
    store, _ = state
    result = runner.invoke(app, ["local", "downloads", "info"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"No downloads in {store}.\n"


@pytest.mark.parametrize(
    ("command", "prompt"),
    [
        (["local", "clear"], "Delete the cache in {cache} and every release in {store}, 2 builds included?"),
        (["local", "cache", "clear"], "Delete the raw source files and control databases in {cache}?"),
        (["local", "downloads", "clear"], "Delete 1 download from {store}?"),
        (["local", "builds", "clear"], "Delete 2 builds from {store}?"),
    ],
)
def test_each_clear_asks_first_and_counts_what_it_deletes(state, command, prompt):
    store, cache = state
    _put_demo(store, "demo/copy", "copy")
    _put_demo(store, "demo/built", "build")
    releases = stored_releases(LocalRegistry(store))
    result = runner.invoke(app, command, input="n\n")
    assert result.exit_code != 0  # aborted
    assert f"{prompt.format(cache=cache, store=store)} [y/N]" in click.unstyle(result.output)
    assert (cache / "downloads").exists()
    assert stored_releases(LocalRegistry(store)) == releases


def test_local_cache_clear_keeps_the_local_registry(state):
    store, cache = state
    size = human_bytes(_downloads_size(cache))
    result = runner.invoke(app, ["local", "cache", "clear"], input="y\n")
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 1 location(s)."
    assert not (cache / "downloads").exists()
    assert [item.release for item in stored_releases(LocalRegistry(store))] == [HELLO]


def test_local_cache_clear_warns_about_a_folder_without_the_marker_and_keeps_it(state):
    _, cache = state
    (cache / "control").mkdir()
    (cache / "control/notes.txt").write_text("not from TimeNet")
    result = runner.invoke(app, ["local", "cache", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert _stderr_has(result, f"Kept {cache / 'control'}: it has no CACHEDIR.TAG from TimeNet")
    assert click.unstyle(result.stdout).splitlines()[-1].endswith("from 1 location(s).")
    assert (cache / "control/notes.txt").exists()


def test_downloads_clear_keeps_each_download_that_a_build_needs_and_names_it(state):
    store, cache = state
    parent = _put_demo(store, "demo/copy", "copy")
    other = _put_demo(store, "demo/other", "copy")
    child = _put_demo(store, "demo/built", "build", parents={ParentAlias("parent"): parent})
    [gone] = [item for item in stored_releases(LocalRegistry(store)) if item.release == other]
    size = human_bytes(gone.size_bytes)
    result = runner.invoke(app, ["local", "downloads", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert _stderr_has(result, f"Kept {parent}: {child} needs it.")
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 1 location(s)."
    assert [item.release for item in stored_releases(LocalRegistry(store))] == [child, parent, HELLO]
    assert (cache / "downloads").exists()


def test_builds_clear_keeps_each_build_that_a_download_needs_and_names_it(tmp_path, monkeypatch):
    # The download test/child reads files of the build timenet/hello-world.
    registry, payload_source = _child_of_the_demo(tmp_path / "store")
    child = DatasetRef.parse("test/child@1.0.0")
    record_origin(registry, child, "copy", "https://copy.example.org")
    _put_demo(tmp_path / "store", "demo/built", "build")
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    result = runner.invoke(app, ["local", "builds", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert _stderr_has(result, f"Kept {payload_source}: {child} needs it.")
    assert [item.release for item in stored_releases(registry)] == [child, payload_source]


def test_local_clear_deletes_the_cache_and_every_release(state):
    store, cache = state
    _put_demo(store, "demo/copy", "copy")
    releases = stored_releases(LocalRegistry(store))
    size = human_bytes(_downloads_size(cache) + sum(item.size_bytes for item in releases))
    result = runner.invoke(app, ["local", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 3 location(s)."
    assert stored_releases(LocalRegistry(store)) == []
    assert not (cache / "downloads").exists()


def test_the_local_commands_take_the_local_registry_flag(tmp_path):
    # The variable of the autouse fixture names an empty store, so the releases come from the flag.
    store = tmp_path / "store"
    LocalRegistry(store).put_dataset(make_dataset())
    _put_demo(store, "demo/copy", "copy")
    flag = ["--local-registry", str(store)]
    result = runner.invoke(app, ["local", "info", *flag])
    assert result.exit_code == 0, result.output
    assert f"Builds in {store}: 1 release" in click.unstyle(result.stdout)
    result = runner.invoke(app, ["local", "builds", "info", *flag])
    assert result.exit_code == 0, result.output
    assert "timenet/hello-world" in click.unstyle(result.stdout)
    result = runner.invoke(app, ["local", "downloads", "clear", "--yes", *flag])
    assert result.exit_code == 0, result.output
    assert [item.release for item in stored_releases(LocalRegistry(store))] == [HELLO]
    result = runner.invoke(app, ["local", "clear", "--yes", *flag])
    assert result.exit_code == 0, result.output
    assert stored_releases(LocalRegistry(store)) == []


def _flags(tmp_path, remote):
    """Return the flags of a list of sources: an empty store and ``remote``, without the connectors."""
    return ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(remote), "--no-auto-build"]


def _first_cells(output):
    """Return the first cell of each body row of the table in ``output``."""
    return [line.split("│")[1].strip() for line in output.splitlines() if line.startswith("│")]


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


def test_info(registry_root, tmp_path):
    flags = ["--remote-registry", str(registry_root), "--local-registry", str(tmp_path / "store"), "--no-auto-build"]

    result = runner.invoke(app, ["info", "timenet/hello-world", *flags])

    assert result.exit_code == 0, result.output
    cells = [[cell.strip() for cell in line.split("│")[1:-1]] for line in result.stdout.splitlines() if "│" in line]
    assert ["version", "1.0.0"] in cells
    assert ["records", str(len(make_dataset().records))] in cells


class _Backend:
    """A builder backend whose card of each dataset is the demo card under its ID, with ``changes``."""

    def __init__(self, *dataset_ids, **changes):
        self.ids = dataset_ids
        self.changes = changes

    def dataset_ids(self):
        return list(self.ids)

    def card(self, dataset_id):
        return make_dataset().metadata.model_copy(update={"dataset_id": dataset_id, **self.changes})


_GATED = {"access": Access.CREDENTIALED, "access_url": "https://gated.example", "citation": "A citation."}


def test_info_shows_the_card_of_a_dataset_that_only_a_connector_has(tmp_path, monkeypatch):
    install_backends(monkeypatch, _Backend("test/gated", **_GATED))
    store = tmp_path / "store"
    flags = ["--local-registry", str(store), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/gated", *flags])

    assert result.exit_code == 0, result.output
    cells = [[cell.strip() for cell in line.split("│")[1:-1]] for line in result.stdout.splitlines() if "│" in line]
    card = _Backend("test/gated", **_GATED).card("test/gated")
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
    lines = click.unstyle(result.output).splitlines()
    cells = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if "│" in line]
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
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
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


def test_publish_takes_a_source(registry_root, tmp_path):
    # The variable of the autouse fixture names an empty store, so the release comes from --source.
    dest = tmp_path / "dest"
    result = runner.invoke(
        app, ["publish", "timenet/hello-world@1.0.0", "--dest", str(dest), "--source", str(registry_root)]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "timenet/hello-world@1.0.0\n"
    assert LocalRegistry(dest).exists("timenet/hello-world", "1.0.0")


def test_publish_opens_a_remote_source_with_the_store_of_the_flag(registry_root, tmp_path, monkeypatch):
    opened = []
    open_any = hints_module.open_registry

    def open_registry(uri, *, local_registry=None, label="registry"):
        if uri != "https://registry.example.test":
            return open_any(uri, local_registry=local_registry, label=label)
        opened.append((uri, local_registry))
        return LocalRegistry(registry_root)  # no network: a local registry stands in for the remote one

    monkeypatch.setattr(hints_module, "open_registry", open_registry)
    store, dest = tmp_path / "store", tmp_path / "dest"
    argv = ["publish", "timenet/hello-world@1.0.0", "--dest", str(dest), "--source", "https://registry.example.test"]
    result = runner.invoke(app, [*argv, "--local-registry", str(store)])
    assert result.exit_code == 0, result.output
    # The remote registry reads the copies of the same releases in the store of --local-registry.
    assert opened == [("https://registry.example.test", store)]


def _words(output):
    """Return the unstyled output with all whitespace removed, since a usage panel wraps its lines."""
    return "".join(click.unstyle(output).split())


def test_publish_rejects_a_source_that_cannot_be_opened(tmp_path):
    result = runner.invoke(app, ["publish", "x/y", "--dest", str(tmp_path / "dest"), "--source", "foo://x"])
    # A bad flag is a usage error that names the flag.
    assert result.exit_code == 2
    assert "--source'foo://x'hasanunsupportedregistryscheme." in _words(result.output)


def test_publish_to_an_empty_dest_fails_and_writes_nothing_into_the_working_directory(tmp_path, monkeypatch):
    store = LocalRegistry(tmp_path / "store")
    store.put_dataset(make_dataset())
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    argv = ["publish", "timenet/hello-world", "--dest", "", "--local-registry", str(store.root)]
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "--destisblank" in _words(result.output)
    assert list(cwd.iterdir()) == []


def test_publish_needs_dest(tmp_path):
    result = runner.invoke(app, ["publish", "timenet/hello-world"])
    assert result.exit_code == 2
    # Rich colours the usage error on a CI runner, which splits the option name with escape codes.
    assert "--dest" in click.unstyle(result.output)


@pytest.mark.parametrize("provider", ["registry", "connector", "source"])
def test_publish_of_a_release_that_the_store_lacks_names_the_fix(tmp_path, monkeypatch, provider):
    store, other, dest = tmp_path / "store", tmp_path / "other", str(tmp_path / "dest")
    install_backends(monkeypatch, _Backend("timenet/hello-world"))
    # The remote registry of the environment is the one that `timenet download` reads.
    if provider == "registry":
        LocalRegistry(tmp_path / "remote").put_dataset(make_dataset())
        monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(tmp_path / "remote"))
    argv = ["publish", "timenet/hello-world@1.0.0", "--dest", dest, "--local-registry", str(store)]
    if provider == "source":
        argv += ["--source", str(other)]

    result = runner.invoke(app, argv)

    assert type(result.exception) is TimeNetDatasetNotFoundError
    registry = other if provider == "source" else store
    # The hint repeats the version and the store. A registry that --source names is not the store, so it gets none.
    hint = {
        "registry": " Copy it into the local registry with "
        f"`timenet download timenet/hello-world@1.0.0 --local-registry {store}`.",
        "connector": " Build it into the local registry with "
        f"`timenet-build build timenet/hello-world@1.0.0 --local-registry {store}`.",
        "source": "",
    }[provider]
    assert str(result.exception) == f"timenet/hello-world@1.0.0 is not in local registry {registry}.{hint}"


def test_publish_names_the_download_of_a_payload_source_that_the_store_lacks(tmp_path, monkeypatch):
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
    LocalRegistry(tmp_path / "remote").put_dataset(make_dataset())
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(tmp_path / "remote"))
    dest = tmp_path / "dest"

    result = runner.invoke(app, ["publish", "test/child", "--dest", str(dest), "--local-registry", str(store.root)])

    # The hint names the payload source, not the release of the command.
    assert type(result.exception) is TimeNetMissingParentError
    assert result.exception.hint == (
        f"Copy it into the local registry with `timenet download {HELLO} --local-registry {store.root}`."
    )


@pytest.mark.parametrize(("flag", "variable"), [("--auto-build", "false"), ("--no-auto-build", "true")])
def test_the_auto_build_flag_beats_its_variable(monkeypatch, flag, variable):
    monkeypatch.setenv("TIMENET_AUTO_BUILD", variable)
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    result = runner.invoke(app, ["download", "x/y", flag])
    # The error names each source of the list.
    assert isinstance(result.exception, TimeNetDatasetNotFoundError)
    assert ("installed connectors" in str(result.exception)) == (flag == "--auto-build")


# The commands that read the list of sources share one path, so download stands for them.
_EMPTY_FLAGS = [
    ["download", "x/y", "--local-registry", ""],
    ["download", "x/y", "--remote-registry", ""],
    ["local", "clear", "--yes", "--local-registry", ""],
]


@pytest.mark.parametrize("argv", _EMPTY_FLAGS, ids=lambda argv: " ".join(argv[:-1]))
def test_an_empty_flag_is_an_error(tmp_path, monkeypatch, argv):
    # An empty value must not mean the current directory: a build would land there, and
    # `local clear` would delete its releases.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, argv)
    # A usage error that names the flag.
    assert result.exit_code == 2
    assert f"{argv[-2]}isblank" in _words(result.output)


def test_a_bad_variable_is_an_error_that_names_the_variable(monkeypatch):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", "https://registry.example")
    result = runner.invoke(app, ["list"])
    # The command line is fine, so this is no usage error.
    assert result.exit_code == 1
    assert isinstance(result.exception, TimeNetRegistryError)
    assert str(result.exception).startswith("TIMENET_LOCAL_REGISTRY 'https://registry.example' is remote")


def test_info_of_a_release_that_no_source_provides_names_the_flags_that_add_one(monkeypatch):
    install_backends(monkeypatch, _Backend("test/x"))

    result = runner.invoke(app, ["info", "test/x", "--no-auto-build"])

    # The CLI names its flags, not the calls of the SDK.
    assert isinstance(result.exception, TimeNetDatasetNotFoundError)
    assert result.exception.hint == (
        "Pass a registry that holds it with --remote-registry <url>, or build it from its installed connector "
        "with --auto-build."
    )


def test_download_of_the_latest_version_names_the_build_and_the_version_that_a_registry_holds(
    registry_root, tmp_path, monkeypatch
):
    # The connector declares 2.0.0, and the registry holds 1.0.0.
    backend = _Backend("timenet/hello-world", dataset_version=Version(2, 0, 0))
    install_backends(monkeypatch, backend)
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(registry_root)]

    result = runner.invoke(app, ["download", "timenet/hello-world@latest", *flags])

    assert isinstance(result.exception, TimeNetNotBuiltError)
    assert result.exception.hint == (
        f"Build it with `timenet-build build timenet/hello-world --local-registry {tmp_path / 'store'} "
        f"--remote-registry {registry_root}`, or download the version that a registry holds with "
        f"`timenet download timenet/hello-world@1.0.0 --local-registry {tmp_path / 'store'} "
        f"--remote-registry {registry_root}`."
    )


def test_info_of_the_latest_version_offline_names_the_version_in_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(builders_module, "entry_points", lambda group: [])
    store = tmp_path / "store"
    LocalRegistry(store).put_dataset(make_dataset())

    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    down = RemoteRegistry("https://down.example", transport=httpx.MockTransport(offline))
    monkeypatch.setattr(client_module, "open_registry", lambda uri, **_: down)

    result = runner.invoke(app, ["info", "timenet/hello-world@latest", "--local-registry", str(store)])

    # "Latest" asks every source, and an exact version reads the store first.
    assert isinstance(result.exception, TimeNetRegistryError)
    assert result.exception.hint == (
        "Ask for timenet/hello-world@1.0.0 to read the copy in the local registry, or pass a registry that you can "
        "reach with --remote-registry <url>."
    )


@pytest.mark.parametrize("command", [["list", "--no-auto-build"], ["local", "downloads", "clear", "--yes"]])
def test_a_manifest_that_is_not_valid_names_the_delete_that_replaces_its_release(state, command):
    store, _ = state
    directory = store / "timenet/hello-world/1.0.0"
    (directory / "manifest.json").write_text("{cut")

    result = runner.invoke(app, command)

    assert result.exit_code == 1
    assert isinstance(result.exception, TimeNetManifestError)
    assert str(result.exception).startswith(f"{directory / 'manifest.json'} is not a valid manifest: ")
    # Download and build read the manifest first, so only a delete lets them replace the release.
    again = f"Delete {directory} and download or build timenet/hello-world@1.0.0 again"
    assert result.exception.hint == (f"{again}, then clear again." if command[0] == "local" else f"{again}.")


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
