"""``timenet-curate``: the producer CLI that runs connectors through the engine.

Distinct from the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory (itself a
valid local registry) that the SDK can then load. Connectors are resolved lazily by dataset id, so
adding one is just dropping a ``datasets/<org>/<name>.py`` module — no registration here.
"""

from pathlib import Path

import typer

from timenet.cli.runner import run_cli
from timenet.cli.ui import console
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors.discovery import resolve


app = typer.Typer(help="Curate TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Curate TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)
    console.quiet = quiet


@app.command()
def build(
    dataset_id: str,
    out: str | None = typer.Option(None, "--out", help="Output registry directory (default: the local registry)."),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if the version is already curated."),
) -> None:
    """Run a connector through the engine and write its dataset.

    Prints an emoji build summary to stderr and the version directory to stdout (for scripts to
    capture). An already-curated version is reused unless ``--force`` is given.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector.
    """
    try:
        connector_cls = resolve(dataset_id)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = Path(out) if out is not None else settings().registry_path
    console.status("🔧", f"Building '{dataset_id}'…")
    version_dir = run_pipeline(connector_cls(), root, progress_cb=_report_progress, force=force)
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
