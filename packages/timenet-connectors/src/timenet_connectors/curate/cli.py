"""``timenet-curate``: the producer CLI that runs connectors through the engine.

Distinct from the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory (itself a
valid local registry) that the SDK can then load. Connectors are resolved lazily by dataset id, so
adding one is just dropping a ``datasets/<org>/<name>/`` package (a ``connector.py`` exposing
``CONNECTOR`` plus a ``dataset.yaml`` card). No registration here.
"""

from pathlib import Path

import typer

from timenet.cli.runner import run_cli
from timenet.cli.ui import console
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.errors import RegistryError
from timenet.registry import default_registry_path
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors.discovery import resolve


app = typer.Typer(help="Curate TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Curate TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)
    console.quiet = quiet


def _default_root() -> Path:
    """The registry directory a build writes to when ``--out`` is not given.

    Mirrors the consumer side's selection order, so the CLI that writes a dataset and the SDK that
    reads it land on the same directory.

    Returns:
        ``$TIMENET_REGISTRY`` when it names a local directory, else ``<home>/registry``.

    Raises:
        BadParameter: If ``$TIMENET_REGISTRY`` names a remote registry, which cannot be built into.
    """
    try:
        return default_registry_path()
    except RegistryError as exc:
        raise typer.BadParameter(f"$TIMENET_REGISTRY {settings().registry!r} is remote; pass --out <dir>") from exc


@app.command()
def build(
    dataset_id: str,
    out: str | None = typer.Option(
        None, "--out", help="Output registry directory (default: $TIMENET_REGISTRY, else the local registry)."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if the version is already curated."),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="Keep the raw download cache after building (default: remove it)."
    ),
) -> None:
    """Run a connector through the engine and write its dataset.

    Prints an emoji build summary to stderr and the version directory to stdout (for scripts to
    capture). An already-curated version is reused unless ``--force`` is given.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector, or ``$TIMENET_REGISTRY`` is remote.
    """
    try:
        connector_cls = resolve(dataset_id)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = Path(out).expanduser() if out is not None else _default_root()
    console.status("🔧", f"Building '{dataset_id}'…")
    version_dir = run_pipeline(
        connector_cls(), root, progress_cb=_report_progress, force=force, clean_cache=not keep_cache
    )
    console.success(f"Built '{dataset_id}' → {version_dir.name}")
    typer.echo(str(version_dir))


def _report_progress(event: WriteProgressEvent) -> None:
    """Relay writer progress to the shared console.

    Args:
        event: The writer progress event.
    """
    if event.stage is ProgressStage.SHARD_FINALIZED:
        console.status("💾", f"wrote shard {event.completed}")


def main() -> None:
    """Entry point for the ``timenet-curate`` console script.

    Expected failures print a one-line message; only unexpected errors surface a traceback.
    """
    run_cli(app)
