"""``timenet-build``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
import json
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

from timenet.builders import BuilderBackend, find_builder, find_card
from timenet.cache import human_bytes
from timenet.cli.runner import run_cli
from timenet.cli.ui import console
from timenet.config import settings
from timenet.engine import publish_pipeline, run_pipeline
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetRegistryError,
)
from timenet.manifest import PayloadSource
from timenet.registry import (
    BaseRegistry,
    LocalRegistry,
    WritableRegistry,
    local_registry_path,
    open_registry,
    open_writable_registry,
)
from timenet.types.references import DatasetRef
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors.builder.env import run_isolated
from timenet_connectors.builder.plan import ParentStep, prepare_parents
from timenet_connectors.discovery import connector_dir, resolve
from timenet_connectors.download import DownloadProgress, ProgressCallback, progress_sink


app = typer.Typer(help="Build TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Build TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)
    console.quiet = quiet


def _resolve_target(out: str | None) -> WritableRegistry | Path:
    """Resolve a build's output target: a writable registry for a remote URL, else a local directory.

    The selection order matches the consumer side, so the CLI that writes a dataset and the SDK that
    reads it agree on where it lands: the ``--out`` argument, then ``$TIMENET_REGISTRY``, then the
    default local registry.

    Args:
        out: The ``--out`` value, or ``None`` to fall back to ``$TIMENET_REGISTRY`` then the default.

    Returns:
        A :class:`~timenet.registry.WritableRegistry` for a ``timenet://`` / ``http(s)://`` / ``s3://``
        target, or the local directory ``Path`` for a plain path or ``file://`` URI.
    """
    target = out if out is not None else settings().registry
    if target is None:
        return settings().registry_path
    try:
        return local_registry_path(target)
    except TimeNetRegistryError:
        return open_writable_registry(target)


@app.command()
def build(  # noqa: PLR0913, PLR0917 (a typer command: one parameter per option)
    dataset_id: str,
    out: str | None = typer.Option(
        None,
        "--out",
        help="Output registry: a local directory or a timenet:// / http(s):// / s3:// URL "
        "(default: $TIMENET_REGISTRY, else the local registry).",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if the version is already built."),
    fetch_from: str | None = typer.Option(
        None,
        "--fetch-from",
        help="Copy missing parent releases from this registry URI (local --out only).",
    ),
    build_parents: bool = typer.Option(
        False, "--build-parents", help="Build missing parent releases from their installed connectors."
    ),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="Keep the raw download cache after building (default: remove it)."
    ),
    values_backend: str | None = typer.Option(
        None,
        "--values-backend",
        help="Values-plane backend: 'parquet' or 'zarr' (default: the connector's declared backend).",
    ),
    isolation: bool | None = typer.Option(
        None,
        "--isolation/--no-isolation",
        help="Run the build in an environment built from the connector's requirements.",
    ),
    pinned_by: Path | None = typer.Option(
        None,
        "--pinned-by",
        hidden=True,
        help="A JSON file of the releases that read files of this release, with their pins. The parent "
        "of an isolated build passes it.",
    ),
) -> None:
    """Run a connector through the engine and write or publish its dataset.

    A local ``--out`` writes a dataset-layout directory and prints its path to stdout. A remote
    ``--out`` (``timenet://`` / ``http(s)://`` / ``s3://``) publishes through the registry and prints
    the version. Either way an emoji summary goes to stderr, so scripts can capture stdout. A built
    version is reused unless you give ``--force``.

    The build runs in an environment built from the connector's ``requirements.txt``. Pass
    ``--no-isolation`` (or set ``TIMENET_ISOLATION=off``) to run it in the current interpreter
    instead, which is what you want while writing a connector.

    A dataset whose card declares ``parents`` needs those exact parent releases in the output
    registry. Missing ones fail the build before any download, unless ``--fetch-from`` names a
    registry to copy them from or ``--build-parents`` builds them from their own connectors. Every
    parent is prepared before the dataset itself builds, and a parent is never rebuilt over a
    release the output already holds.

    ``--values-backend`` overrides the values-plane storage the connector declares. This process
    resolves the connector's default for an in-process build; an isolated child resolves its own,
    because importing the connector there would need the dependencies it is about to install.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector, or ``--values-backend`` names a
            backend TimeF does not support.
        TimeNetMissingParentError: If a parent is missing, with a hint that names the flags.
        TimeNetFileError: If a parent copy in the output is damaged, or a fetched file does not match
            its manifest record, with a hint that names the flags.
        TimeNetAccessError: If a remote output would get a release that is not open.
    """
    # Reject an unknown backend before anything expensive happens, so neither the isolated child nor
    # a connector import runs for a build that cannot succeed.
    if values_backend is not None and values_backend not in SUPPORTED_VALUES_BACKENDS:
        raise typer.BadParameter(
            f"unknown values backend {values_backend!r}; supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
        )
    try:
        connector_dir(dataset_id)  # import-free, so an unknown id fails before any environment exists
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc
    forwarded = out if out is not None else settings().registry
    if forwarded is None:
        forwarded = str(settings().registry_path)
    try:
        result = build_one(
            dataset_id,
            forwarded,
            fetch_from=fetch_from,
            build_parents=build_parents,
            force=force,
            values_backend=values_backend,
            keep_cache=keep_cache,
            isolation=isolation,
            narrate=True,
            pinned_by=None if pinned_by is None else _read_pins(pinned_by),
        )
    except TimeNetMissingParentError as error:
        error.hint = "Fetch it with --fetch-from <registry> or build it with --build-parents."
        raise
    except TimeNetFileError as error:
        error.hint = (
            f"Fetch an intact copy with --fetch-from <registry>, or rebuild it with `timenet-build build "
            f"{error.release.dataset_id} --force`. A rebuild changes its files, and the releases that "
            "pin the old files then fail to open."
        )
        raise
    except TimeNetAccessError as error:
        error.hint = "A remote --out holds open releases only. Build it into a local --out instead."
        raise
    typer.echo(result)


def build_one(  # noqa: PLR0913
    dataset_id: str,
    output: str | Path,
    *,
    fetch_from: str | BaseRegistry | None = None,
    build_parents: bool = False,
    force: bool = False,
    values_backend: str | None = None,
    keep_cache: bool = False,
    isolation: bool | None = None,
    narrate: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> str:
    """Build one dataset: prepare its parent releases, then run its connector.

    Every build entry point runs through this function: the ``build`` command,
    :class:`~timenet_connectors.builder.backend.ConnectorBuilder`, and :func:`timenet_connectors.build`.
    A missing parent fails the build before any download, unless ``fetch_from`` or ``build_parents``
    obtains it. A parent builds the same way as this dataset, but never with ``force``. A parent
    that another builder backend claims builds through that backend's ``build``, with its pins. A
    version that the output already holds returns without a connector run, unless ``force``. The
    connector runs in an environment built from its requirements, or in this interpreter when
    isolation is off. A local output gets a version directory, and a remote output gets a published
    version.

    Args:
        dataset_id: The dataset id.
        output: The output registry: a local directory or a ``timenet://`` / ``http(s)://`` /
            ``s3://`` URL. An isolated child receives it unchanged as ``--out``.
        fetch_from: The registry, or its URI, to copy missing parent releases from.
        build_parents: Build missing parent releases from their installed connectors.
        force: Rebuild even if the version is already built.
        values_backend: Values-plane backend, or ``None`` for the connector's declared backend.
        keep_cache: Keep the raw download cache after building.
        isolation: Build in an environment built from the connector's requirements, or ``None``
            for the ``TIMENET_ISOLATION`` setting.
        narrate: Print status lines and download progress for an in-process build.
        pinned_by: The releases that already read files of this release, with their pins. The build
            publishes the release only if it reproduces each pinned file.

    Returns:
        The committed version directory for a local output, or the published version.
    """
    target = _resolve_target(str(output))
    registry = LocalRegistry(target) if isinstance(target, Path) else target
    override = None if isolation is None else ("on" if isolation else "off")
    isolated = settings(isolation=override).isolation == "on"

    def build_parent(step: ParentStep) -> object:
        builder = find_builder(step.reference.dataset_id)
        if builder is not None and not _in_repo(builder):
            return _build_with(builder, step, target)
        # Each parent builds like the dataset itself. An isolated child plans again, finds every
        # parent in place, and only opens them.
        return _build(
            step.reference.dataset_id,
            target,
            output,
            isolated=isolated,
            keep_cache=keep_cache,
            narrate=narrate,
            pinned_by=step.pinned_by,
        )

    prepare_parents(
        dataset_id,
        registry,
        build=build_parent,
        fetch_from=open_registry(fetch_from) if isinstance(fetch_from, str) else fetch_from,
        build_parents=build_parents,
        force_build=force,
    )
    committed = None if force else _committed(dataset_id, target, registry)
    if committed is not None:
        if narrate:
            built = f"Built '{dataset_id}' → {Path(committed).name}"
            console.success(built if isinstance(target, Path) else f"Published '{dataset_id}' → {committed}")
        return committed
    return _build(
        dataset_id,
        target,
        output,
        isolated=isolated,
        force=force,
        keep_cache=keep_cache,
        values_backend=values_backend,
        narrate=narrate,
        pinned_by=pinned_by,
    )


def _in_repo(builder: BuilderBackend) -> bool:
    """Report whether a builder backend is the one of this package, which builds through :func:`_build`.

    Returns:
        Whether ``builder`` is a :class:`~timenet_connectors.builder.backend.ConnectorBuilder`.
    """
    from timenet_connectors.builder.backend import ConnectorBuilder  # noqa: PLC0415 - the backend imports this module

    return isinstance(builder, ConnectorBuilder)


def _build_with(builder: BuilderBackend, step: ParentStep, target: WritableRegistry | Path) -> Path:
    """Build a parent through the builder backend that claims it, with the pins of its step.

    The plan already prepared the parent's own parents, so the backend gets no parent options.

    Returns:
        The committed version directory.

    Raises:
        TimeFValidationError: If the output is not local, since a builder backend builds into a
            local directory.
    """
    if not isinstance(target, Path):
        raise TimeFValidationError(
            f"parent release {step.reference} builds through another builder backend, which writes to a "
            "local directory; build it into a local output and publish it first"
        )
    return builder.build(step.reference.dataset_id, target, pinned_by=step.pinned_by or None)


def _committed(dataset_id: str, target: WritableRegistry | Path, registry: WritableRegistry) -> str | None:
    """Return the result of a build whose version the output already holds.

    Returns:
        The version directory for a local output or the version for a remote one, or ``None`` if
        the output lacks the version.
    """
    card = find_card(dataset_id)
    if card is None or not registry.exists(card.dataset_id, str(card.dataset_version)):
        return None
    version = str(card.dataset_version)
    return str(target / card.dataset_id / version) if isinstance(target, Path) else version


def _build(  # noqa: PLR0913
    dataset_id: str,
    target: WritableRegistry | Path,
    forwarded: str | Path,
    *,
    isolated: bool,
    force: bool = False,
    keep_cache: bool = False,
    values_backend: str | None = None,
    narrate: bool = False,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> str:
    """Build one dataset into the output and return its result line.

    Args:
        dataset_id: The dataset id.
        target: The resolved output for an in-process build.
        forwarded: The output as given, for an isolated child's ``--out``.
        isolated: Build in an environment built from the connector's requirements.
        force: Rebuild even if the version is already built.
        keep_cache: Keep the raw download cache after building.
        values_backend: Values-plane backend, or ``None`` for the connector's declared backend.
        narrate: Print status lines and download progress for an in-process build.
        pinned_by: The releases that already read files of this release, with their pins.

    Returns:
        The committed version directory for a local output, or the published version.
    """
    pinned_by = pinned_by or None
    if isolated:
        # The child is this same CLI and inherits stderr, so it is the only narrator: it prints the
        # status lines and the richer download and shard progress. This branch stays quiet and only
        # relays what the child printed on stdout. Do not call resolve() here; it imports the connector
        # module, which needs the dependencies the child is about to install. run_isolated forwards
        # --out unchanged, so a remote target publishes from inside the environment exactly as a local
        # one writes to a directory.
        return run_isolated(
            dataset_id,
            forwarded,
            force=force,
            keep_cache=keep_cache,
            values_backend=values_backend,
            quiet=console.quiet,
            pinned_by=pinned_by,
        )
    if narrate:
        console.status("🔧", f"Building '{dataset_id}'…")
    # Resolve the connector's declared backend here, so the in-process path and the isolated child
    # agree on what "no --values-backend" means. An isolated build cannot resolve it: importing the
    # connector needs the dependencies the child is about to install.
    connector = resolve(dataset_id)()
    resolved_backend = values_backend if values_backend is not None else connector.values_backend
    progress_cb = _report_progress if narrate else None
    # A connector's downloads report through the ambient progress sink. _download_progress renders them.
    with _download_progress() if narrate else nullcontext():
        if isinstance(target, Path):
            version_dir = run_pipeline(
                connector,
                target,
                values_backend=resolved_backend,
                progress_cb=progress_cb,
                force=force,
                keep_cache=keep_cache,
                pinned_by=pinned_by,
            )
            if narrate:
                console.success(f"Built '{dataset_id}' → {version_dir.name}")
            return str(version_dir)
        version = publish_pipeline(
            connector,
            target,
            values_backend=resolved_backend,
            progress_cb=progress_cb,
            force=force,
            keep_cache=keep_cache,
            pinned_by=pinned_by,
        )
        if narrate:
            console.success(f"Published '{dataset_id}' → {version}")
        return version


def _read_pins(path: Path) -> dict[DatasetRef, PayloadSource]:
    """Read the pins that the parent of an isolated build wrote for this child.

    Returns:
        Each release that reads files of the build, with its pins.
    """
    pins = json.loads(path.read_text(encoding="utf-8"))
    return {DatasetRef.parse(key): PayloadSource.model_validate(value) for key, value in pins.items()}


def _report_progress(event: WriteProgressEvent) -> None:
    """Relay writer progress to the console.

    The shared console coordinates with any live download bars, so this renders above them.

    Args:
        event: The writer progress event.
    """
    if event.stage is ProgressStage.SHARD_FINALIZED:
        console.status("💾", f"wrote shard {event.completed}")


@contextmanager
def _download_progress() -> Generator[Progress | None]:
    """Install the download-progress sink for a build and render it.

    In an interactive terminal, downloads show as live progress bars, one row per file, updating in
    parallel. When output goes through a pipe or you set ``--quiet``, this falls back to throttled
    text lines (or silence), so redirected logs stay readable.

    Yields:
        The live progress display when rendering bars, else ``None``. This installs the sink for
        the duration of the block.
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
    row). Later events update its position.

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

    Reports each file at roughly 20% steps, or every 50 MB when the total size is unknown. Keys each
    report per URL so concurrent downloads do not interleave into a flood of lines.

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
    """Entry point for the ``timenet-build`` console script.

    Expected failures print a one-line message. Only unexpected errors surface a traceback.
    """
    run_cli(app)
