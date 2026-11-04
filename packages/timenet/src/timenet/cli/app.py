"""The consumer command-line interface, the command-line mirror of the :class:`~timenet.client.TimeNet` SDK."""

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
import shlex
from typing import TypeVar

from rich.console import Console
from rich.table import Table
import typer

from timenet.cache import cache_size, clear_cache, clear_releases, human_bytes, stored_releases
from timenet.cli.options import AUTO_BUILD, LOCAL_REGISTRY, REMOTE_REGISTRY
from timenet.cli.runner import run_cli
from timenet.cli.ui import console as ui, progress_renderer
from timenet.client import TimeNet, default_local_registry, default_sources
from timenet.config import settings
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.producer import publish as publish_release
from timenet.progress import ProgressCallback, ProgressEvent, StepStarted
from timenet.registry import open_registry
from timenet.types import DatasetMetadata, Domain, License, TaskType
from timenet.types.access import access_requirements, effective_access
from timenet.types.references import split_ref
from timenet.types.tasks import TASKS


app = typer.Typer(help="Browse a TimeNet registry and fetch datasets.", no_args_is_help=True)
console = Console()

T = TypeVar("T")


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Browse a TimeNet registry and fetch datasets."""
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


def _client(local_registry: str | None, remote_registry: str | None, auto_build: bool | None) -> TimeNet:
    """Create a client with the selected registries and build setting.

    Pass ``None`` to use the matching ``TIMENET_*`` environment variable or its default.

    Args:
        local_registry: The local registry path.
        remote_registry: The remote registry URL or path.
        auto_build: Whether to build missing datasets with installed connectors.

    Returns:
        The configured client.
    """
    sources = default_sources(local_registry=local_registry, remote_registry=remote_registry, auto_build=auto_build)
    return TimeNet(sources=sources)


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
    """List the datasets in the registry."""
    rows = _client(local_registry, remote_registry, auto_build).list()
    if not rows:
        typer.echo("No datasets in the registry.")
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
    """Search for datasets that match the filters.

    Repeat a filter flag to supply multiple values.
    """
    results = _client(local_registry, remote_registry, auto_build).search(
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
    """Show a dataset's metadata, schema, and counts."""
    manifest = _client(local_registry, remote_registry, auto_build).get(dataset_id)
    counts = manifest.counts
    table = Table(title=manifest.dataset_id, title_justify="left", title_style="bold", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("version", str(manifest.metadata.dataset_version))
    table.add_row("name", manifest.metadata.name)
    table.add_row("license", str(manifest.metadata.license))
    if manifest.metadata.license_url:
        table.add_row("license_url", manifest.metadata.license_url)
    # The strictest level of the card and its lineage applies, also when the card itself says open.
    table.add_row("access", str(effective_access(manifest.metadata, manifest.lineage)))
    # Each card whose access is not open, the dataset or an ancestor, names its own page.
    requirements = access_requirements(manifest.metadata, manifest.lineage)
    if requirements:
        pages = (f"{item.release} ({item.access}): {item.access_url}" for item in requirements)
        table.add_row("access_urls", "\n".join(pages))
    if manifest.metadata.citation:
        table.add_row("citation", manifest.metadata.citation)
    table.add_row("domains", ", ".join(str(domain) for domain in manifest.metadata.domains) or "-")
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


@app.command()
def download(
    dataset_id: str,
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
    force: bool = typer.Option(
        False, "--force", help="Copy the release again, even if the local registry holds a complete copy."
    ),
) -> None:
    """Copy a release and its data files into the local registry.

    The command prints the release directory to stdout.
    """
    started: list[StepStarted] = []
    with progress_renderer() as show:
        client = _client(local_registry, remote_registry, auto_build)
        target = client.download(dataset_id, force=force, progress=partial(_keep_steps, started, show))
    if not started:
        # The local registry already held the release and its payload sources.
        store = default_local_registry(local_registry=local_registry)
        ui.success(f"Nothing to do: '{split_ref(dataset_id)[0]}' {target.name} is already in {store}")
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
    dest: str = typer.Option(..., "--dest", metavar="URL|PATH", help="The registry to copy to: a URL or a local path."),
    source: str | None = typer.Option(
        None,
        "--source",
        metavar="URL|PATH",
        help="The registry to copy from: a URL or a local path (default: the local registry).",
    ),
    local_registry: str | None = LOCAL_REGISTRY,
) -> None:
    """Copy a release and its payload sources to another registry.

    Read from ``--source`` or the local registry, and print the release to stdout.
    Skip identical releases and fail if the destination has different files for the same version.
    \f

    Raises:
        TimeNetDatasetNotFoundError: If the source does not contain the release.
    """  # ruff: ignore[escape-sequence-in-docstring] - the form feed ends the help text of the command
    store = default_local_registry(local_registry=local_registry)
    origin = store if source is None else open_registry(source, label="--source")
    try:
        with progress_renderer() as progress:
            release = publish_release(dataset_id, dest=dest, source=origin, progress=progress)
    except TimeNetDatasetNotFoundError as error:
        # The hint fits only a release that the local registry lacks. A subclass, such as a missing payload
        # source, names its own registries.
        if source is not None or type(error) is not TimeNetDatasetNotFoundError:
            raise
        bare_id, version = split_ref(dataset_id)
        wanted = bare_id if version is None else f"{bare_id}@{version}"
        # Repeat the path of --local-registry, so that each command of the hint uses it too.
        flag = "" if local_registry is None else f" --local-registry {shlex.quote(str(store.root))}"
        hint = (
            f"Build it into the local registry with `timenet-build build {wanted}{flag}`, "
            f"or copy it there with `timenet download {wanted}{flag}`."
        )
        raise TimeNetDatasetNotFoundError(f"{str(error).rstrip('.')}. {hint}") from error
    ui.success(f"Published {release} to {dest}")
    typer.echo(str(release))


local_app = typer.Typer(help="Inspect and clear the local files of TimeNet.", no_args_is_help=True)
app.add_typer(local_app, name="local")
cache_app = typer.Typer(
    help="The cache in $TIMENET_CACHE: raw source files of connectors and control databases of remote registries.",
    no_args_is_help=True,
)
local_app.add_typer(cache_app, name="cache")

_YES = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt.")


@local_app.command("info")
def local_info(local_registry: str | None = LOCAL_REGISTRY) -> None:
    """Show cache and local registry paths and sizes."""
    store, cache_dir = default_local_registry(local_registry=local_registry), settings().cache_dir
    releases = stored_releases(store)
    size = human_bytes(sum(item.size_bytes for item in releases))
    # Plain lines, so that a long path does not wrap.
    typer.echo(_cache_line(cache_dir))
    typer.echo(f"Local registry {store.root}: {_count(len(releases), 'release')}, {size}")


@local_app.command("clear")
def local_clear(yes: bool = _YES, local_registry: str | None = LOCAL_REGISTRY) -> None:
    """Delete the cache and every release in the local registry."""
    store, cache_dir = default_local_registry(local_registry=local_registry), settings().cache_dir
    if not yes:
        typer.confirm(f"Delete the cache in {cache_dir} and every release in {store.root}?", abort=True)
    # The cache clear checks the folders before it deletes anything, so it runs first.
    cache = clear_cache(cache_dir, local_registry=store)
    releases = clear_releases(store)
    _warn_unmarked(cache.kept)
    _freed(releases.freed + cache.freed, len(releases.removed) + len(cache.removed))


@cache_app.command("info")
def cache_info() -> None:
    """Show the path and size of the cache."""
    typer.echo(_cache_line(settings().cache_dir))


@cache_app.command("clear")
def cache_clear(yes: bool = _YES) -> None:
    """Delete the raw source files and the control databases in the cache."""
    cache_dir = settings().cache_dir
    if not yes:
        typer.confirm(f"Delete the raw source files and control databases in {cache_dir}?", abort=True)
    cleared = clear_cache(cache_dir, local_registry=default_local_registry())
    _warn_unmarked(cleared.kept)
    _freed(cleared.freed, len(cleared.removed))


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
