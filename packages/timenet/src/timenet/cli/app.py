"""The consumer command-line interface, the command-line mirror of the :class:`~timenet.client.TimeNet` SDK."""

from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar

from rich.console import Console
from rich.table import Table
import typer

from timenet.cache import CachedDataset, cached_datasets, clear_cache, human_bytes, raw_cache_size
from timenet.cli.runner import run_cli
from timenet.cli.ui import console as ui, progress_renderer
from timenet.client import TimeNet, default_sources
from timenet.progress import ProgressCallback, ProgressEvent, StepStarted
from timenet.types import DatasetMetadata, Domain, License, TaskType
from timenet.types.access import access_requirements, effective_access
from timenet.types.references import split_ref
from timenet.types.tasks import TASKS


app = typer.Typer(help="Browse a TimeNet registry and fetch datasets.", no_args_is_help=True)
console = Console()

_registry_option = typer.Option(
    None,
    "--registry",
    "-r",
    help="Registry URL or path that this command reads (else $TIMENET_REGISTRY, then timenet://).",
)
# TODO(adrian): Remove temporary glue which maps --registry onto the list after the store, until the CLI
# takes the flags of the defaults.
_download_registry_option = typer.Option(
    None,
    "--registry",
    "-r",
    help="Registry URL or path to put after the store in the list of sources. The release resolves through "
    "the whole list.",
)

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


def _client(registry: str | None) -> TimeNet:
    """Create a client with the default sources and an optional registry.

    Args:
        registry: The registry URL or path to read after the store, or ``None`` for the default.

    Returns:
        A client with the configured sources.
    """
    # TODO(adrian): Remove temporary glue which maps --registry onto default_sources() until the
    # CLI takes the flags of the defaults.
    return TimeNet(sources=default_sources(remote_registry=registry))


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
def list_datasets(registry: str | None = _registry_option) -> None:
    """Display metadata for every dataset in the selected registry."""
    rows = _client(registry).list()
    if not rows:
        typer.echo("No datasets in the registry.")
        return
    console.print(_metadata_table(rows, "Datasets"))


@app.command()
def search(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    registry: str | None = _registry_option,
    query: list[str] = typer.Option([], "--query", "-q", help="Free-text terms."),
    domain: list[str] = typer.Option([], "--domain", help="Domain(s)."),
    task: list[str] = typer.Option([], "--task", help="Task type(s)."),
    license: list[str] = typer.Option([], "--license", help="License(s)."),
    spec: list[str] = typer.Option([], "--spec", help="time_series_spec type(s)."),
    dataset_id: list[str] = typer.Option([], "--id", help="Dataset id(s)."),
    tag: list[str] = typer.Option([], "--tag", help="Tag(s)."),
    limit: int = typer.Option(100, "--limit", min=0, help="Maximum results."),
) -> None:
    """Find datasets that match the filters.

    Repeat a filter flag to pass several values.
    """
    results = _client(registry).search(
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
def info(dataset_id: str, registry: str | None = _registry_option) -> None:
    """Display a dataset's metadata, schema, and counts."""
    manifest = _client(registry).get(dataset_id)
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
def download(dataset_id: str, registry: str | None = _download_registry_option) -> None:
    """Copy a release and its payload sources into the store.

    Print the release directory to standard output.
    """
    sources = default_sources(remote_registry=registry)
    started: list[StepStarted] = []
    with progress_renderer() as show:
        target = TimeNet(sources=sources).download(dataset_id, progress=partial(_keep_steps, started, show))
    if not started:
        # The store already held the release and its payload sources.
        ui.success(f"Nothing to do: '{split_ref(dataset_id)[0]}' {target.name} is already in {sources[0]}")
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


cache_app = typer.Typer(help="Inspect and clear the local cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("info")
def cache_info() -> None:
    """Show the datasets in the local cache with their sizes."""
    datasets = cached_datasets()
    if datasets:
        console.print(_cache_table(datasets))
    else:
        console.print("No cached datasets.")
    raw = raw_cache_size()
    if raw:
        console.print(f"Raw download cache: {human_bytes(raw)}")


def _cache_table(datasets: list[CachedDataset]) -> Table:
    """Build a table of cached dataset versions with a total-size footer.

    Args:
        datasets: The cached dataset versions to render.

    Returns:
        A populated :class:`rich.table.Table`.
    """
    table = Table(title="TimeNet cache", title_justify="left", title_style="bold")
    table.add_column("Location")
    table.add_column("Dataset", style="cyan")
    table.add_column("Version")
    table.add_column("Size", justify="right")
    for dataset in datasets:
        table.add_row(dataset.location, dataset.dataset_id, dataset.version, human_bytes(dataset.size_bytes))
    table.add_section()
    total = sum(dataset.size_bytes for dataset in datasets)
    table.add_row("total", "", "", human_bytes(total), style="bold")
    return table


@cache_app.command("clear")
def cache_clear(
    all_: bool = typer.Option(
        False, "--all", help="Also remove the local registry, the default store (builds and downloads)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete the storage directory and raw cache.

    Use ``--all`` to also remove the default store.
    """
    if not yes:
        what = "the storage directory and the raw cache"
        if all_:
            what += " and the local registry (builds and downloads)"
        typer.confirm(f"Remove {what}?", abort=True)
    freed, removed = clear_cache(include_registry=all_)
    typer.echo(f"Freed {human_bytes(freed)} from {len(removed)} location(s).")


def main() -> None:
    """Entry point for the ``timenet`` console script.

    Expected failures (unknown dataset, unreachable registry) print a one-line message. Only
    unexpected errors show a traceback.
    """
    run_cli(app)
