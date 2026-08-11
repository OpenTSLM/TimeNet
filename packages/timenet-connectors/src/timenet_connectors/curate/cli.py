"""``timenet-curate``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sys

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
import typer

from timenet.cache import human_bytes
from timenet.cli.runner import run_cli
from timenet.cli.ui import console
from timenet.engine import run_pipeline
from timenet.errors import RegistryError
from timenet.registry import default_registry_path
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors.discovery import resolve
from timenet_connectors.download import DownloadProgress, ProgressCallback, progress_sink


app = typer.Typer(help="Curate TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Curate TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)
    console.quiet = quiet


def _default_root() -> Path:
    """Return the registry directory a build writes to when ``--out`` is not given.

    This uses the same selection order as the consumer side. The CLI that writes a dataset and the
    SDK that reads it land on the same directory.

    Returns:
        ``$TIMENET_REGISTRY`` when it names a local directory, else ``<home>/registry``.

    Raises:
        BadParameter: If ``$TIMENET_REGISTRY`` cannot be resolved to a local output directory.
    """
    try:
        return default_registry_path()
    except RegistryError as exc:
        raise typer.BadParameter(f"$TIMENET_REGISTRY: {exc}; pass --out <dir>") from exc


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

    This prints an emoji build summary to stderr and the version directory to stdout, so scripts can
    capture it. If a version is already curated, the build reuses it unless you give ``--force``.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector, or ``$TIMENET_REGISTRY`` is remote.
    """
    try:
        connector_cls = resolve(dataset_id)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = Path(out).expanduser() if out is not None else _default_root()
    console.status("🔧", f"Building '{dataset_id}'…")
    # A connector's downloads report through the ambient progress sink; _download_progress renders them.
    with _download_progress():
        version_dir = run_pipeline(
            connector_cls(), root, progress_cb=_report_progress, force=force, clean_cache=not keep_cache
        )
    console.success(f"Built '{dataset_id}' → {version_dir.name}")
    typer.echo(str(version_dir))


def _report_progress(event: WriteProgressEvent) -> None:
    """Relay writer progress to the console.

    The shared console coordinates with any live download bars, so this renders above them.

    Args:
        event: The writer progress event.
    """
    if event.stage is ProgressStage.SHARD_FINALIZED:
        console.status("💾", f"wrote shard {event.completed}")


@contextmanager
def _download_progress() -> Iterator[Progress | None]:
    """Install the download-progress sink for a build and render it.

    In an interactive terminal, downloads show as live progress bars, one row per file, updating in
    parallel. When output is piped or ``--quiet`` is set, falls back to throttled text lines (or
    silence), so redirected logs stay readable.

    Yields:
        The live progress display when rendering bars, else ``None``; the sink is installed for the
        duration of the block.
    """
    if console.quiet:
        with progress_sink(None):
            yield None
        return
    if not sys.stderr.isatty():
        with progress_sink(_text_download_reporter()):
            yield None
        return
    progress = Progress(
        TextColumn("[bold blue]{task.fields[name]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console.rich,
    )
    with progress, progress_sink(_bar_reporter(progress)):
        yield progress


def _bar_reporter(progress: Progress) -> ProgressCallback:
    """Build a download-progress sink that maps each URL to a live progress-bar task.

    The first event for a URL adds a task (keyed by URL so concurrent downloads stay on their own
    row); later events update its position.

    Args:
        progress: The display to add and update per-file tasks on.

    Returns:
        A callback for :func:`~timenet_connectors.download.progress_sink`.
    """
    tasks: dict[str, TaskID] = {}

    def report(event: DownloadProgress) -> None:
        task_id = tasks.get(event.url)
        if task_id is None:
            task_id = progress.add_task("download", name=event.url.rsplit("/", 1)[-1], total=event.total)
            tasks[event.url] = task_id
        progress.update(task_id, completed=event.downloaded, total=event.total)

    return report


_UNKNOWN_TOTAL_STEP_BYTES = 50 * 1024 * 1024  # progress cadence when the download size is unknown


def _text_download_reporter() -> ProgressCallback:
    """Build a download-progress sink that prints throttled status lines to the console.

    Reports each file at roughly 20% steps (or every 50 MB when the total size is unknown), keyed per
    URL so concurrent downloads don't interleave into a flood of lines.

    Returns:
        A callback for :func:`~timenet_connectors.download.progress_sink`.
    """
    last: dict[str, int] = {}

    def report(event: DownloadProgress) -> None:
        # ~20% steps when the size is known, else one line every 50 MB.
        step = event.downloaded * 5 // event.total if event.total else event.downloaded // _UNKNOWN_TOTAL_STEP_BYTES
        if last.get(event.url) == step:
            return
        last[event.url] = step
        name = event.url.rsplit("/", 1)[-1]
        if event.total:
            console.status("⬇️", f"{name} {event.downloaded * 100 // event.total}% ({human_bytes(event.total)})")
        else:
            console.status("⬇️", f"{name} {human_bytes(event.downloaded)}")

    return report


def main() -> None:
    """Entry point for the ``timenet-curate`` console script.

    Expected failures print a one-line message. Only unexpected errors surface a traceback.
    """
    run_cli(app)
