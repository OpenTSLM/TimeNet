"""``timenet-curate``: the producer CLI that runs connectors through the engine.

Distinct from the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory (itself a
valid local registry) that the SDK can then load. Connectors are resolved lazily by dataset id, so
adding one is just dropping a ``datasets/<org>/<name>.py`` module — no registration here.
"""

from pathlib import Path

import typer

from timenet.cli.runner import run_cli
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet_connectors.discovery import resolve


app = typer.Typer(help="Curate TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Curate TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)


@app.command()
def build(
    dataset_id: str,
    out: str | None = typer.Option(None, "--out", help="Output registry directory (default: the local registry)."),
) -> None:
    """Run a connector through the engine and write its dataset, printing the version directory.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector.
    """
    try:
        connector_cls = resolve(dataset_id)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = Path(out) if out is not None else settings().registry_path
    version_dir = run_pipeline(connector_cls(), root)
    typer.echo(str(version_dir))


def main() -> None:
    """Entry point for the ``timenet-curate`` console script.

    Expected failures print a one-line message; only unexpected errors surface a traceback.
    """
    run_cli(app)
