"""The consumer command-line interface, the command-line mirror of the :class:`~timenet.client.TimeNet` SDK."""

from collections.abc import Callable, Iterable
from typing import TypeVar

from rich.console import Console
from rich.table import Table
import typer

from timenet.cache import CachedDataset, cached_datasets, clear_cache, human_bytes, raw_cache_size
from timenet.cli.runner import run_cli
from timenet.cli.ui import console as ui
from timenet.client import TimeNet
from timenet.types import DatasetMetadata, Domain, License, TaskType
from timenet.types.tasks import TASKS


app = typer.Typer(help="Browse a TimeNet registry and fetch datasets.", no_args_is_help=True)
console = Console()

_registry_option = typer.Option(None, "--registry", "-r", help="Registry URL or path (else $TIMENET_REGISTRY).")

T = TypeVar("T")


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Browse a TimeNet registry and fetch datasets."""
    ui.quiet = quiet


def _enum_list(values: list[str], factory: Callable[[str], T], flag: str, choices: Iterable[str]) -> list[T] | None:
    """Parse a repeated string option into enum values, reporting a bad value as a clean CLI error.

    Args:
        values: The raw strings passed for the option.
        factory: Builds the target value from one string (raises on an unknown value).
        flag: The option name, for the error message.
        choices: The valid values, for the error message.

    Returns:
        The parsed values, or ``None`` if none were given.

    Raises:
        BadParameter: If any value is not recognized.
    """
    try:
        parsed = [factory(value) for value in values]
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(f"invalid {flag} value ({exc}); choose from: {', '.join(choices)}") from exc
    return parsed or None


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
    """List every dataset in the registry."""
    rows = TimeNet(registry).list()
    if not rows:
        typer.echo("No datasets in the registry.")
        return
    console.print(_metadata_table(rows, "Datasets"))


@app.command()
def search(
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
    """Search datasets by filter. Repeat a flag to pass several values."""
    results = TimeNet(registry).search(
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
def info(dataset_id: str, version: str | None = None, registry: str | None = _registry_option) -> None:
    """Print a dataset's manifest summary: metadata, schema, and counts."""
    manifest = TimeNet(registry).get(dataset_id, version)
    counts = manifest.counts
    table = Table(title=manifest.dataset_id, title_justify="left", title_style="bold", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("version", str(manifest.metadata.dataset_version))
    table.add_row("name", manifest.metadata.name)
    table.add_row("license", str(manifest.metadata.license))
    table.add_row("domains", ", ".join(str(domain) for domain in manifest.metadata.domains) or "-")
    table.add_row("specs", ", ".join(spec.spec_type for spec in manifest.schema.time_series_specs) or "-")
    table.add_row("tasks", ", ".join(str(task.task_type) for task in manifest.schema.tasks) or "-")
    table.add_section()
    table.add_row("samples", str(counts.samples))
    table.add_row("annotations", str(counts.annotations))
    table.add_row("chunks", str(counts.time_series_chunks))
    console.print(table)


@app.command()
def download(
    dataset_id: str,
    version: str | None = None,
    registry: str | None = _registry_option,
    storage: str | None = typer.Option(None, "--storage", help="Local storage dir (else $TIMENET_STORAGE)."),
) -> None:
    """Fetch a complete TimeF version to local storage. Prints its directory to stdout."""
    target = TimeNet(registry, storage_path=storage).download(dataset_id, version)
    ui.success(f"Downloaded '{dataset_id}' ({target.name})")
    typer.echo(str(target))


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
    all_: bool = typer.Option(False, "--all", help="Also remove the local registry (curated datasets)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete cached data. Clears downloads by default; ``--all`` also removes curated datasets."""
    if not yes:
        what = "downloaded datasets and the raw cache"
        if all_:
            what += " and the local registry (curated datasets)"
        typer.confirm(f"Remove {what}?", abort=True)
    freed, removed = clear_cache(include_registry=all_)
    typer.echo(f"Freed {human_bytes(freed)} from {len(removed)} location(s).")


def main() -> None:
    """Entry point for the ``timenet`` console script.

    Expected failures (unknown dataset, unreachable registry) print a one-line message; only
    unexpected errors surface a traceback.
    """
    run_cli(app)
