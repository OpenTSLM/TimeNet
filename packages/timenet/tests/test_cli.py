import importlib
import shlex
import shutil
import subprocess
import sys

from _origin_record import origin_file
import click
import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from timenet import _hints as hints_module
from timenet.cache import human_bytes, stored_releases
from timenet.cli.app import _enum_list, app
from timenet.client import TimeNet
from timenet.config import make_cache_folder
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetInvalidCardError,
    TimeNetManifestError,
    TimeNetMissingParentError,
    TimeNetNotBuiltError,
    TimeNetOriginError,
    TimeNetRegistryError,
)
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry.origin import OriginKind
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


def _put(store, dataset, kind):
    """Put ``dataset`` into the store, with a provenance record of ``kind``.

    Returns:
        The release.
    """
    registry = LocalRegistry(store)
    release = registry.put_dataset(dataset)
    registry.record_origin(release, kind, f"https://{kind.value}.example.org")
    return release


def _stderr_has(result, text):
    """Return whether stderr holds ``text``.

    The console wraps a long line, also in a path, so the check ignores whitespace.
    """
    return "".join(text.split()) in "".join(click.unstyle(result.stderr).split())


def test_local_info_and_local_cache_info_show_the_paths_and_sizes(state):
    store, cache = state
    _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
    result = runner.invoke(app, ["local", "info"])
    assert result.exit_code == 0, result.output
    copy, build = stored_releases(LocalRegistry(store))
    cache_line = f"Cache {cache}: {human_bytes(_downloads_size(cache))}"
    assert click.unstyle(result.stdout).splitlines() == [
        cache_line,
        f"Downloads in {store}: 1 release, {human_bytes(copy.size())}",
        f"Builds in {store}: 1 release, {human_bytes(build.size())}",
    ]
    result = runner.invoke(app, ["local", "cache", "info"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout) == f"{cache_line}\n"


@pytest.mark.parametrize(
    ("command", "rows"),
    [
        (["local", "downloads", "info"], [["demo/copy", "1.0.0", "https://download.example.org"]]),
        (
            ["local", "builds", "info"],
            [["demo/built", "1.0.0", "https://build.example.org"], ["timenet/hello-world", "1.0.0", "-"]],
        ),
    ],
)
def test_downloads_info_and_builds_info_list_the_releases_of_their_kind(state, command, rows):
    store, _ = state
    _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
    _put(store, make_dataset(dataset_id="demo/built"), OriginKind.BUILD)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    cells = _table_cells(click.unstyle(result.stdout))
    sizes = {item.release.dataset_id: item.size() for item in stored_releases(LocalRegistry(store))}
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
    _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
    _put(store, make_dataset(dataset_id="demo/built"), OriginKind.BUILD)
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
    parent = _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
    other = _put(store, make_dataset(dataset_id="demo/other"), OriginKind.DOWNLOAD)
    child = _put(
        store, make_dataset(dataset_id="demo/built", parents={ParentAlias("parent"): parent}), OriginKind.BUILD
    )
    [gone] = [item for item in stored_releases(LocalRegistry(store)) if item.release == other]
    size = human_bytes(gone.size())
    result = runner.invoke(app, ["local", "downloads", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert _stderr_has(result, f"Kept {parent}: {child} needs it.")
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 1 location(s)."
    assert [item.release for item in stored_releases(LocalRegistry(store))] == [child, parent, HELLO]
    assert (cache / "downloads").exists()


def test_local_clear_deletes_the_cache_and_every_release(state):
    store, cache = state
    _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
    releases = stored_releases(LocalRegistry(store))
    size = human_bytes(_downloads_size(cache) + sum(item.size() for item in releases))
    result = runner.invoke(app, ["local", "clear", "--yes"])
    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stdout).splitlines()[-1] == f"Freed {size} from 3 location(s)."
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
    # The variable of the autouse fixture names an empty store, so the releases come from the flag.
    store = tmp_path / "store"
    LocalRegistry(store).put_dataset(make_dataset())
    _put(store, make_dataset(dataset_id="demo/copy"), OriginKind.DOWNLOAD)
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


_NO_PROVIDER = (
    "Pass a registry that holds it with --remote-registry <url>, or install a connector package that builds it, such "
    "as timenet-connectors."
)


def test_a_failed_read_computes_its_hint_once(tmp_path, monkeypatch):
    computed = []
    remedy = hints_module._remedy
    monkeypatch.setattr(hints_module, "_remedy", lambda error, *rest: computed.append(error) or remedy(error, *rest))

    result = runner.invoke(app, ["info", "x/y", *_flags(tmp_path, tmp_path / "empty")])

    assert isinstance(result.exception, TimeNetDatasetNotFoundError)
    assert result.exception.hint == _NO_PROVIDER
    assert computed == [result.exception]


def test_an_sdk_call_after_a_command_gets_the_hint_of_the_sdk(tmp_path):
    command = runner.invoke(app, ["info", "x/y", *_flags(tmp_path, tmp_path / "empty")])
    assert isinstance(command.exception, TimeNetDatasetNotFoundError)
    assert command.exception.hint == _NO_PROVIDER

    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        TimeNet(sources=[LocalRegistry(tmp_path / "store")]).open("x/y")

    assert caught.value.hint == (
        "Add a registry that holds it to the sources, or install a connector package that builds it, such as "
        "timenet-connectors."
    )


def test_search_no_match(registry_root, tmp_path):
    result = runner.invoke(app, ["search", "--domain", "cardiology", *_flags(tmp_path, registry_root)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "No datasets match.\n"


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


def test_list_shows_a_warning_as_a_line_without_a_file_and_a_line_number(tmp_path, monkeypatch):
    class _Broken(_Backend):
        def card(self, dataset_id):
            raise TimeNetInvalidCardError("the card cannot be read")

    install_backends(monkeypatch, _Broken("test/gated"))
    monkeypatch.setattr(_cli_module.ui, "rich", Console(stderr=True, width=1000))
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["list", *flags])

    assert result.exit_code == 0, result.output
    assert click.unstyle(result.stderr).splitlines() == [
        "⚠️ test/gated: the card of its installed connector cannot be read, so the listing leaves it out: "
        "the card cannot be read"
    ]


def test_info_shows_the_card_of_a_dataset_that_only_a_connector_has(tmp_path, monkeypatch):
    install_backends(monkeypatch, _Backend("test/gated", **_GATED))
    store = tmp_path / "store"
    flags = ["--local-registry", str(store), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/gated", *flags])

    assert result.exit_code == 0, result.output
    cells = _table_cells(result.stdout)
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


def test_info_of_an_unbuilt_child_says_that_its_access_leaves_out_its_parents(tmp_path, monkeypatch):
    parents = {ParentAlias(name): DatasetRef.parse(f"test/{name}@1.0.0") for name in ("a", "b")}
    install_backends(monkeypatch, _Backend("test/child", parents=parents))
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

    install_backends(monkeypatch, _Counting("test/gated"))
    flags = ["--local-registry", str(tmp_path / "store"), "--remote-registry", str(tmp_path / "empty"), "--auto-build"]

    result = runner.invoke(app, ["info", "test/gated", *flags])

    assert result.exit_code == 0, result.output
    # Each resolution of the release reads the card, so one read means one resolution.
    assert reads == ["test/gated"]


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


def test_download_command_copies_into_the_local_registry_and_then_has_nothing_to_do(
    registry_root, tmp_path, monkeypatch
):
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


def test_publish_copies_a_release_from_the_local_registry(tmp_path):
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


@pytest.mark.parametrize("provider", ["connector", "no connector", "source"])
def test_publish_of_a_release_that_the_local_registry_lacks_names_the_fix(tmp_path, monkeypatch, provider):
    local, other, dest = tmp_path / "local", tmp_path / "other", str(tmp_path / "dest")
    if provider != "no connector":
        install_backends(monkeypatch, _Backend("timenet/hello-world"))
    argv = ["publish", "timenet/hello-world@1.0.0", "--dest", dest, "--local-registry", str(local)]
    if provider == "source":
        argv += ["--source", str(other)]

    result = runner.invoke(app, argv)

    assert type(result.exception) is TimeNetDatasetNotFoundError
    registry = other if provider == "source" else local
    # The hint repeats the version and the local registry. A registry that --source names gets none.
    # Without a connector, the hint names the download, although it asks no registry for the release.
    hint = {
        "connector": " Build it into the local registry with "
        f"`timenet-build build timenet/hello-world@1.0.0 --local-registry {local}`.",
        "no connector": " Copy it into the local registry with "
        f"`timenet download timenet/hello-world@1.0.0 --local-registry {local}`.",
        "source": "",
    }[provider]
    assert str(result.exception) == f"timenet/hello-world@1.0.0 is not in local registry {registry}.{hint}"


def test_publish_names_the_download_of_a_payload_source_that_the_local_registry_lacks(tmp_path):
    local = LocalRegistry(tmp_path / "local")
    local.put_dataset(make_dataset())
    child = TimeFDataset(
        metadata=make_dataset().metadata.model_copy(
            update={"dataset_id": "test/child", "parents": {ParentAlias("hello"): HELLO}}
        )
    )
    # The child reads the values of one Record of the parent, so the parent is its payload source.
    with TimeNet(sources=[local]).open(str(HELLO)) as reader:
        child.import_from(reader, alias=ParentAlias("hello"), records=[make_dataset().records[0].id])
        local.put_dataset(child)
    shutil.rmtree(local.root / "timenet")
    dest = tmp_path / "dest"

    result = runner.invoke(app, ["publish", "test/child", "--dest", str(dest), "--local-registry", str(local.root)])

    # The hint names the payload source, not the release of the command.
    assert type(result.exception) is TimeNetMissingParentError
    assert result.exception.hint == (
        f"Copy it into the local registry with `timenet download {HELLO} --local-registry {local.root}`."
    )


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


@pytest.mark.parametrize(
    "command", [["local", "downloads", "clear", "--yes"], ["local", "info"], ["local", "downloads", "info"]]
)
def test_a_damaged_provenance_record_names_the_forced_download_and_build(tmp_path, command):
    store = tmp_path / "store"
    LocalRegistry(store).put_dataset(make_dataset())
    origin_file(LocalRegistry(store), HELLO).write_text("{")

    result = runner.invoke(app, [*command, "--local-registry", str(store)])

    assert result.exit_code == 1
    assert isinstance(result.exception, TimeNetOriginError)
    flag = f"--local-registry {shlex.quote(str(store))}"
    assert result.exception.hint == (
        f"Get the release again with `timenet download {HELLO} {flag} --force` if it came from a registry, or "
        f"with `timenet-build build {HELLO} {flag} --force` if it was built. Both replace the release and its record."
    )


def test_the_local_registry_flag_replaces_a_variable_that_is_not_valid(registry_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", "")
    result = runner.invoke(app, ["list", *_flags(tmp_path, registry_root)])
    assert result.exit_code == 0, result.output
    assert "timenet/hello-world" in result.stdout


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
