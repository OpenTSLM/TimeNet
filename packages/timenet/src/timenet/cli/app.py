"""The consumer command-line interface, the command-line mirror of the :class:`~timenet.client.TimeNet` SDK."""

from collections.abc import Callable, Iterable

import typer

from timenet.cli.runner import run_cli
from timenet.client import TimeNet
from timenet.types import Domain, License, TaskType
from timenet.types.tasks import TASKS


app = typer.Typer(help="Browse a TimeNet registry and fetch datasets.", no_args_is_help=True)

_registry_option = typer.Option(None, "--registry", "-r", help="Registry URL or path (else $TIMENET_REGISTRY).")


def _enum_list[T](values: list[str], factory: Callable[[str], T], flag: str, choices: Iterable[str]) -> list[T] | None:
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


@app.command("list")
def list_datasets(registry: str | None = _registry_option) -> None:
    """List every dataset in the registry."""
    for metadata in TimeNet(registry).list():
        typer.echo(f"{metadata.dataset_id}\t{metadata.name}")


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
    for metadata in results:
        typer.echo(f"{metadata.dataset_id}\t{metadata.name}")


@app.command()
def info(dataset_id: str, version: str | None = None, registry: str | None = _registry_option) -> None:
    """Print a dataset's manifest summary: metadata, schema, and counts."""
    manifest = TimeNet(registry).get(dataset_id, version)
    counts = manifest.counts
    typer.echo(f"{manifest.dataset_id} {manifest.metadata.dataset_version} — {manifest.metadata.name}")
    typer.echo(f"  license: {manifest.metadata.license}")
    typer.echo(f"  domains: {', '.join(manifest.metadata.domains) or '-'}")
    typer.echo(f"  specs: {', '.join(s.spec_type for s in manifest.schema.time_series_specs) or '-'}")
    typer.echo(f"  tasks: {', '.join(str(t.task_type) for t in manifest.schema.tasks) or '-'}")
    typer.echo(
        f"  counts: samples={counts.samples} annotations={counts.annotations} chunks={counts.time_series_chunks}"
    )


@app.command()
def download(
    dataset_id: str,
    version: str | None = None,
    registry: str | None = _registry_option,
    storage: str | None = typer.Option(None, "--storage", help="Local storage dir (else $TIMENET_DATA_ROOT)."),
) -> None:
    """Fetch a dataset's parquet to local storage and print the directory."""
    target = TimeNet(registry, storage_path=storage).download(dataset_id, version)
    typer.echo(str(target))


def main() -> None:
    """Entry point for the ``timenet`` console script.

    Expected failures (unknown dataset, unreachable registry) print a one-line message; only
    unexpected errors surface a traceback.
    """
    run_cli(app)
