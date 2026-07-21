"""``timenet-curate``: the producer CLI that runs connectors through the engine.

Distinct from the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory (itself a
valid local registry) that the SDK can then load.
"""

from pathlib import Path

import typer

from timenet.cli.runner import run_cli
from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.engine import run_pipeline
from timenet_connectors import HelloWorldConnector


CONNECTORS: dict[str, type[BaseConnector]] = {"hello_world": HelloWorldConnector}

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
    if dataset_id not in CONNECTORS:
        raise typer.BadParameter(f"unknown dataset id {dataset_id!r}; known: {sorted(CONNECTORS)}")
    root = Path(out) if out is not None else settings().registry_path
    version_dir = run_pipeline(CONNECTORS[dataset_id](), root)
    typer.echo(str(version_dir))


def main() -> None:
    """Entry point for the ``timenet-curate`` console script.

    Expected failures print a one-line message; only unexpected errors surface a traceback.
    """
    run_cli(app)
