"""The consumer command-line interface, the command-line mirror of the :class:`~timenet.client.TimeNet` SDK."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar, cast

from rich.console import Console
from rich.table import Table
import typer

from timenet._hints import Facts, build_command
from timenet.cache import (
    ReleaseKind,
    cache_size,
    clear_cache,
    clear_releases,
    human_bytes,
    releases_to_clear,
    stored_releases,
)
from timenet.cli.hints import cli_hints, registry_flag, sources_from_flags, store_from_flag
from timenet.cli.options import AUTO_BUILD, LOCAL_REGISTRY, REMOTE_REGISTRY, STORE
from timenet.cli.runner import run_cli
from timenet.cli.ui import console as ui, progress_renderer
from timenet.client import TimeNet
from timenet.config import settings
from timenet.errors import TimeNetNotBuiltError
from timenet.producer import publish as publish_release
from timenet.progress import ProgressCallback, ProgressEvent, StepStarted
from timenet.registry import LocalRegistry
from timenet.types import DatasetMetadata, Domain, License, TaskType
from timenet.types.access import access_requirements, effective_access
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.references import split_ref
from timenet.types.tasks import TASKS


app = typer.Typer(help="Find, read, download and publish TimeNet datasets.", no_args_is_help=True)
console = Console()

T = TypeVar("T")


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Find, read, download and publish TimeNet datasets."""
    ui.quiet = quiet


def _enum_list(values: list[str], factory: Callable[[str], T], flag: str, choices: Iterable[str]) -> list[T] | None:
    """Parse a repeated string option into enum values. Report a bad value as a clean CLI error.

    Args:
        values: The raw strings passed for the option.
        factory: Builds the target value from one string (raises on an unknown value).
        flag: The option name, for the error message.
        choices: The valid values, for the error message.

    Returns:
        The parsed values, or ``None`` if the caller passed no values.

    Raises:
        BadParameter: If the factory does not recognize a value.
    """
    try:
        parsed = [factory(value) for value in values]
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(f"invalid {flag} value ({exc}); choose from: {', '.join(choices)}") from exc
    return parsed or None


def _client(
    local_registry: str | None,
    remote_registry: str | None,
    auto_build: bool | None,
    *,
    request: Literal["read", "download"] = "read",
    wanted: str | None = None,
) -> tuple[TimeNet, Facts]:
    """Create a client and error-hint details from CLI options.

    A ``None`` flag uses the matching ``TIMENET_*`` environment variable or its default.

    Args:
        local_registry: The local registry path.
        remote_registry: The remote registry URL or path.
        auto_build: Whether to build missing datasets with installed connectors.
        request: What the command does.
        wanted: The requested dataset ID, or ``None``.

    Returns:
        The client and the facts.
    """
    sources = sources_from_flags(local_registry, remote_registry, auto_build)
    # The default list starts with the local registry.
    store = cast("LocalRegistry", sources[0])
    facts = Facts(
        sources,
        store,
        request=request,
        local_registry=local_registry,
        remote_registry=remote_registry,
        wanted=wanted,
    )
    return TimeNet(sources=sources), facts


def _metadata_table(rows: list[DatasetMetadata], title: str) -> Table:
    """Build a table of dataset metadata (id, name, license, domains).

    Args:
        rows: The dataset metadata to render.
        title: The table title.

    Returns:
        A populated :class:`rich.table.Table`.
    """
    table = Table(title=title, title_justify="left", title_style="bold")
    table.add_column("Dataset", style="cyan")
    table.add_column("Name")
    table.add_column("License")
    table.add_column("Domains")
    for metadata in rows:
        table.add_row(
            metadata.dataset_id,
            metadata.name,
            str(metadata.license),
            ", ".join(str(domain) for domain in metadata.domains) or "-",
        )
    return table


@app.command("list")
def list_datasets(
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
) -> None:
    """List each dataset at its highest version across all sources."""
    client, facts = _client(local_registry, remote_registry, auto_build)
    with cli_hints(facts):
        rows = client.list()
    if not rows:
        typer.echo("No datasets in the sources.")
        return
    console.print(_metadata_table(rows, "Datasets"))


@app.command()
def search(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    query: list[str] = typer.Option([], "--query", "-q", help="Free-text terms."),
    domain: list[str] = typer.Option([], "--domain", help="Domain(s)."),
    task: list[str] = typer.Option([], "--task", help="Task type(s)."),
    license: list[str] = typer.Option([], "--license", help="License(s)."),
    spec: list[str] = typer.Option([], "--spec", help="time_series_spec type(s)."),
    dataset_id: list[str] = typer.Option([], "--id", help="Dataset id(s)."),
    tag: list[str] = typer.Option([], "--tag", help="Tag(s)."),
    limit: int = typer.Option(100, "--limit", min=0, help="Maximum results."),
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
) -> None:
    """Search datasets across all sources with filters.

    Repeat a filter flag to pass several values.
    """
    client, facts = _client(local_registry, remote_registry, auto_build)
    with cli_hints(facts):
        results = client.search(
            query=query or None,
            domain=_enum_list(domain, Domain, "--domain", [d.value for d in Domain]),
            task=_enum_list(task, lambda value: TASKS[TaskType(value)], "--task", [t.value for t in TaskType]),
            license=_enum_list(license, License, "--license", [lic.value for lic in License]),
            time_series_spec=spec or None,
            dataset_id=dataset_id or None,
            tag=tag or None,
            limit=limit,
        )
    if not results:
        typer.echo("No datasets match.")
        return
    console.print(_metadata_table(results, "Search results"))


@app.command()
def info(
    dataset_id: str,
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
) -> None:
    """Print a dataset's metadata, schema, and counts.

    For an unbuilt release, show its metadata card and build command instead.
    """
    client, facts = _client(local_registry, remote_registry, auto_build, wanted=dataset_id)
    with cli_hints(facts):
        try:
            manifest = client.get_manifest(dataset_id)
        except TimeNetNotBuiltError:
            console.print(_card_table(client.get_card(dataset_id)))
            # A plain line, so that a long path does not wrap.
            build = build_command(split_ref(dataset_id)[0], facts)
            typer.echo(f"The schema and the counts appear after a build: `{build}`.")
            return
    counts = manifest.counts
    table = _card_table(manifest.metadata, manifest.lineage)
    table.add_row("specs", ", ".join(spec.spec_type for spec in manifest.dataset_schema.time_series_specs) or "-")
    table.add_row("tasks", ", ".join(str(task.task_type) for task in manifest.dataset_schema.tasks) or "-")
    table.add_section()
    table.add_row("records", str(counts.records))
    table.add_row("sources", str(counts.sources))
    table.add_row("signals", str(counts.signals))
    table.add_row("annotation contents", str(counts.annotation_contents))
    table.add_row("annotation occurrences", str(counts.annotation_occurrences))
    table.add_row("chunks", str(counts.signal_chunks))
    console.print(table)
    if counts.tasks:
        tasks = Table(title="Tasks", title_justify="left", title_style="bold")
        tasks.add_column("type", style="cyan")
        tasks.add_column("count", justify="right")
        for task_type, count in sorted(counts.tasks.items()):
            tasks.add_row(task_type, str(count))
        console.print(tasks)


def _card_table(card: DatasetMetadata, lineage: Sequence[ParentDatasetCard] = ()) -> Table:
    """Create a table of dataset metadata and access requirements.

    Args:
        card: The dataset card or manifest metadata.
        lineage: Ancestor cards, or an empty sequence for an unbuilt release.

    Returns:
        A table without a header.
    """
    table = Table(title=card.dataset_id, title_justify="left", title_style="bold", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("version", str(card.dataset_version))
    table.add_row("name", card.name)
    table.add_row("license", str(card.license))
    if card.license_url:
        table.add_row("license_url", card.license_url)
    # The strictest level of the card and its lineage applies, also when the card itself says open.
    table.add_row("access", str(effective_access(card, lineage)))
    # Each card whose access is not open, the dataset or an ancestor, names its own page.
    requirements = access_requirements(card, lineage)
    if requirements:
        pages = (f"{item.release} ({item.access}): {item.access_url}" for item in requirements)
        table.add_row("access_urls", "\n".join(pages))
    if card.citation:
        table.add_row("citation", card.citation)
    table.add_row("domains", ", ".join(str(domain) for domain in card.domains) or "-")
    return table


@app.command()
def download(
    dataset_id: str,
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
) -> None:
    """Copy a release and its data files into the local registry.

    The command prints the release directory to stdout.
    """
    started: list[StepStarted] = []
    client, facts = _client(local_registry, remote_registry, auto_build, request="download", wanted=dataset_id)
    with cli_hints(facts), progress_renderer() as show:
        target = client.download(dataset_id, progress=partial(_keep_steps, started, show))
    if not started:
        # The local registry already held the release and its payload sources.
        ui.success(f"Nothing to do: '{split_ref(dataset_id)[0]}' {target.name} is already in {facts.store}")
    typer.echo(str(target))


def _keep_steps(started: list[StepStarted], show: ProgressCallback | None, event: ProgressEvent) -> None:
    """Keep each step event in ``started``, and pass every event on to ``show``.

    Args:
        started: Receives the step events.
        show: The callback that shows the events, or ``None`` for a quiet console.
        event: The progress event.
    """
    if isinstance(event, StepStarted):
        started.append(event)
    if show is not None:
        show(event)


@app.command()
def publish(
    dataset_id: str,
    dest: str = typer.Option(..., "--dest", metavar="URL", help="The registry to copy to: a URL or a local path."),
    source: str | None = typer.Option(
        None,
        "--source",
        metavar="URL",
        help="The registry to copy from: a URL or a local path (default: the local registry).",
    ),
    local_registry: str | None = STORE,
) -> None:
    """Copy a release and its payload sources to another registry.

    Read from `--source` or the local registry, and print the release to stdout.
    Skip identical releases and fail if the destination has different files for the same version.
    """
    store = store_from_flag(local_registry)
    target = registry_flag(dest, "--dest")
    # A remote source reads the copies of the same releases in the local registry.
    origin = store if source is None else registry_flag(source, "--source", store=store.root)
    # Only the local registry gets the hints that copy or build a release into it.
    facts = Facts(
        [origin],
        store if source is None else None,
        request="publish",
        local_registry=local_registry,
        dest=target,
        wanted=dataset_id,
    )
    with cli_hints(facts), progress_renderer() as progress:
        release = publish_release(dataset_id, dest=target, source=origin, progress=progress)
    ui.success(f"Published {release} to {dest}")
    typer.echo(str(release))


local_app = typer.Typer(help="Inspect and clear the local files of TimeNet.", no_args_is_help=True)
app.add_typer(local_app, name="local")
cache_app = typer.Typer(
    help="The cache in $TIMENET_CACHE: raw source files of connectors and control databases of remote registries.",
    no_args_is_help=True,
)
local_app.add_typer(cache_app, name="cache")
downloads_app = typer.Typer(help="The releases in the local registry that came from a registry.", no_args_is_help=True)
local_app.add_typer(downloads_app, name="downloads")
builds_app = typer.Typer(
    help="The releases in the local registry that a connector built. A release without a record counts as a build.",
    no_args_is_help=True,
)
local_app.add_typer(builds_app, name="builds")

_YES = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt.")
_KINDS: tuple[ReleaseKind, ...] = ("download", "build")


@local_app.command("info")
def local_info(local_registry: str | None = STORE) -> None:
    """Show paths and sizes for the cache, downloads, and builds."""
    store, cache_dir = store_from_flag(local_registry), settings().cache_dir
    releases = stored_releases(store)
    # Plain lines, so that a long path does not wrap.
    typer.echo(_cache_line(cache_dir))
    for kind in _KINDS:
        of_kind = [item for item in releases if item.kind == kind]
        size = human_bytes(sum(item.size_bytes for item in of_kind))
        typer.echo(f"{kind.capitalize()}s in {store.root}: {_count(len(of_kind), 'release')}, {size}")


@local_app.command("clear")
def local_clear(yes: bool = _YES, local_registry: str | None = STORE) -> None:
    """Delete the cache, every download and every build."""
    store, cache_dir = store_from_flag(local_registry), settings().cache_dir
    with cli_hints(Facts([store], store, request="clear", local_registry=local_registry)):
        if not yes:
            builds = _count(sum(item.kind == "build" for item in stored_releases(store)), "build")
            typer.confirm(
                f"Delete the cache in {cache_dir} and every release in {store.root}, {builds} included?", abort=True
            )
        releases = clear_releases(store, kinds=_KINDS)
        cache = clear_cache(cache_dir, local_registry=store)
    _warn_unmarked(cache.kept)
    _freed(releases.freed + cache.freed, len(releases.removed) + len(cache.removed))


@cache_app.command("info")
def cache_info() -> None:
    """Show the path and size of the cache."""
    typer.echo(_cache_line(settings().cache_dir))


@cache_app.command("clear")
def cache_clear(yes: bool = _YES) -> None:
    """Clear cached source files and control databases."""
    cache_dir = settings().cache_dir
    if not yes:
        typer.confirm(f"Delete the raw source files and control databases in {cache_dir}?", abort=True)
    freed, removed, kept = clear_cache(cache_dir, local_registry=store_from_flag(None))
    _warn_unmarked(kept)
    _freed(freed, len(removed))


@downloads_app.command("info")
def downloads_info(local_registry: str | None = STORE) -> None:
    """Show each download in the local registry with its registry and size."""
    _show_releases("download", local_registry)


@downloads_app.command("clear")
def downloads_clear(yes: bool = _YES, local_registry: str | None = STORE) -> None:
    """Delete the downloads in the local registry, except those that a remaining release needs."""
    _clear_releases("download", yes, local_registry)


@builds_app.command("info")
def builds_info(local_registry: str | None = STORE) -> None:
    """Show each build in the local registry with its connector package and size."""
    _show_releases("build", local_registry)


@builds_app.command("clear")
def builds_clear(yes: bool = _YES, local_registry: str | None = STORE) -> None:
    """Delete the builds in the local registry, except those that a remaining release needs."""
    _clear_releases("build", yes, local_registry)


def _show_releases(kind: ReleaseKind, local_registry: str | None) -> None:
    """Show releases of one kind in a table, or a message if none exist."""
    store = store_from_flag(local_registry)
    releases = [item for item in stored_releases(store) if item.kind == kind]
    if not releases:
        typer.echo(f"No {kind}s in {store.root}.")
        return
    table = Table(title=f"{kind.capitalize()}s in {store.root}", title_justify="left", title_style="bold")
    table.add_column("Dataset", style="cyan")
    table.add_column("Version")
    table.add_column("Source", overflow="fold")
    table.add_column("Size", justify="right")
    for item in releases:
        # A build without a record has no source.
        source = "-" if item.origin is None else item.origin.source
        table.add_row(item.release.dataset_id, str(item.release.version), source, human_bytes(item.size_bytes))
    table.add_section()
    table.add_row("total", "", "", human_bytes(sum(item.size_bytes for item in releases)), style="bold")
    console.print(table)


def _clear_releases(kind: ReleaseKind, yes: bool, local_registry: str | None) -> None:
    """Delete the releases of one kind after a prompt that counts them, and name each release that stays."""
    store = store_from_flag(local_registry)
    with cli_hints(Facts([store], store, request="clear", local_registry=local_registry)):
        if not yes:
            releases, _ = releases_to_clear(store, kinds=[kind])
            typer.confirm(f"Delete {_count(len(releases), kind)} from {store.root}?", abort=True)
        freed, removed, kept = clear_releases(store, kinds=[kind])
    for item in kept:
        ui.warn(f"Kept {item.release}: {item.needed_by} needs it.")
    _freed(freed, len(removed))


def _cache_line(cache_dir: Path) -> str:
    """Return the path and the size of the cache."""
    return f"Cache {cache_dir}: {human_bytes(cache_size(cache_dir))}"


def _warn_unmarked(folders: list[Path]) -> None:
    """Warn about cache folders kept because they lack TimeNet's marker."""
    for folder in folders:
        ui.warn(
            f"Kept {folder}: it has no CACHEDIR.TAG from TimeNet, so TimeNet did not create it. Delete it "
            "yourself if you do not need it."
        )


def _freed(freed: int, locations: int) -> None:
    """Print the bytes that a clear freed, and from how many locations."""
    typer.echo(f"Freed {human_bytes(freed)} from {locations} location(s).")


def _count(number: int, noun: str) -> str:
    """Format a count with a noun, adding ``s`` unless the count is 1.

    Returns:
        The count and noun, such as ``"1 build"`` or ``"2 builds"``.
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def main() -> None:
    """Entry point for the ``timenet`` console script.

    Expected failures (unknown dataset, unreachable registry) print a one-line message. Only
    unexpected errors show a traceback.
    """
    run_cli(app)
