"""``timenet-build``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

import json
from pathlib import Path

import typer

from timenet.builders import BuildOptions, declared
from timenet.cli.runner import run_cli
from timenet.cli.ui import console, progress_renderer
from timenet.config import settings
from timenet.coordinator.build import build_release
from timenet.errors import (
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
    copy_release,
    local_registry_path,
    open_registry,
    open_writable_registry,
)
from timenet.registry._paths import release_dir  # ruff: ignore[import-private-name]
from timenet.types.references import DatasetRef, is_latest, split_ref
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.discovery import known_ids


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
# ruff: ignore[too-many-arguments, too-many-positional-arguments] (a typer command: one parameter per option)
def build(
    release: str = typer.Argument(..., metavar="DATASET_ID[@VERSION]"),
    out: str | None = typer.Option(
        None,
        "--out",
        help="Output registry: a local directory or a timenet:// / http(s):// / s3:// URL "
        "(default: $TIMENET_REGISTRY, else the local registry).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Build the release again, even if the local registry holds it."
    ),
    fetch_from: str | None = typer.Option(
        None,
        "--fetch-from",
        help="Copy missing parent releases from this registry URI (default for a remote --out: that registry).",
    ),
    build_parents: bool = typer.Option(
        False, "--build-parents", help="Build missing parent releases from their installed connectors."
    ),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="Keep the raw download cache after building (default: remove it)."
    ),
    isolation: bool | None = typer.Option(
        None,
        "--isolation/--no-isolation",
        help="Run the build in an environment built from the connector's requirements.",
    ),
) -> None:
    """Run a connector through the engine and write or publish its dataset.

    A local ``--out`` writes a dataset-layout directory and prints its path to stdout. A remote
    ``--out`` (``timenet://`` / ``http(s)://`` / ``s3://``) gets the release in two steps. First, the
    release builds into the default local registry. Then a copy publishes it with its payload
    sources, and the command prints the version. If the remote registry already holds the version
    and ``--force`` is not set, nothing builds and nothing is copied. Either way an emoji summary goes
    to stderr, so scripts can capture stdout. A built version is reused unless you give ``--force``.
    A published version never changes. So ``--force`` with a remote ``--out`` that holds the version
    is a usage error, and nothing runs.

    The build runs in an environment built from the connector's ``requirements.txt``. Pass
    ``--no-isolation`` (or set ``TIMENET_ISOLATION=off``) to run it in the current interpreter
    instead, which is what you want while writing a connector.

    A dataset whose card declares ``parents`` needs those exact parent releases in the local
    registry. ``--fetch-from`` names a registry to copy a missing parent from. For a remote
    ``--out``, that registry is the default, because it holds the parents of its releases.
    ``--build-parents`` builds a missing parent from its own connector. If neither option obtains a
    missing parent, the build fails before any download. The build prepares every parent before the
    dataset itself. It never builds a parent over a release that the local registry already holds.
    If the local registry already holds the version, the build needs only the payload sources of
    the version. It prepares them the same way as a parent.

    Raises:
        BadParameter: If ``dataset_id`` has no known connector. Also if ``--force`` is set and the
            remote ``--out`` holds the version. Nothing runs then.
        TimeNetVersionUnavailableError: If the connector cannot build the requested version.
        TimeNetMissingParentError: If a parent is missing. The hint of the error names the flags.
        TimeNetFileError: If a copy of a parent in the local registry is damaged, or a copied file
            does not match its manifest record. The hint of the error names the flags.
        TimeNetAccessError: If the output is remote and gets a release without access declared
            ``open``.
        TimeNetVersionConflictError: If the copy to the remote ``--out`` finds other files of the
            version, or of a payload source of it, in that registry. Nothing is copied then.
    """  # ruff: ignore[docstring-extraneous-exception] - the copy raises TimeNetVersionConflictError
    # known_ids() imports no connector. So an unknown id fails before any environment exists.
    dataset_id, version = split_ref(release)
    known = known_ids()
    if dataset_id not in known:
        raise typer.BadParameter(f"no connector for {dataset_id!r}; known: {', '.join(known) or '(none)'}")
    if not is_latest(version):
        declared(dataset_id, version)
    try:
        # The parent shows each copy and build step. An isolated child shows its own downloads and shards.
        with progress_renderer() as progress:
            result = _build_to(
                dataset_id,
                _resolve_target(out),
                fetch_from=None if fetch_from is None else open_registry(fetch_from),
                build_parents=build_parents,
                force=force,
                options=BuildOptions(
                    keep_cache=keep_cache,
                    progress=progress,
                    quiet=console.quiet,
                    isolation=None if isolation is None else ("on" if isolation else "off"),
                ),
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


def _build_to(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    target: WritableRegistry | Path,
    *,
    fetch_from: BaseRegistry | None,
    build_parents: bool,
    force: bool,
    options: BuildOptions,
) -> str:
    """Build a release locally and copy it to a remote output if needed.

    Reuse an existing release unless ``force`` is true.

    Args:
        dataset_id: The dataset ID.
        target: The local directory or remote registry to write to.
        fetch_from: The registry to copy missing parents from. ``None`` uses a remote output, if any.
        build_parents: Build missing parents with their installed connectors.
        force: Rebuild the release even if the local registry holds it.
        options: Settings for each build.

    Returns:
        The version directory for a local output, or the version string for a remote output.

    Raises:
        BadParameter: If ``force`` is set and the remote output already holds the version.
    """
    release = declared(dataset_id).release
    if isinstance(target, Path):
        store, remote = LocalRegistry(target), None
    elif target.exists(release.dataset_id, str(release.version)):
        if force:
            raise typer.BadParameter(
                f"the remote --out holds {release}, and a published version never changes. --force builds "
                "a release again only for a local --out, or for a version that the remote --out does not hold.",
                param_hint="--force",
            )
        console.success(f"Nothing to do: '{release.dataset_id}' {release.version} is already in the remote --out")
        return str(release.version)
    else:
        store, remote = LocalRegistry(settings().registry_path), target
        # A build into the remote registry found the parents there. So without --fetch-from, the build
        # copies a missing parent from the remote registry into the local one.
        fetch_from = fetch_from if fetch_from is not None else remote
    # A remote registry holds only open releases. So for a remote output, the build refuses another
    # release before any download, as a build into that registry would.
    plan = build_release(
        dataset_id,
        store=store,
        fetch_from=fetch_from,
        build_parents=build_parents,
        force=force,
        open_access_only=remote is not None,
        options=options,
    )
    if remote is not None:
        copy_release(release, store, remote, progress=options.progress)
        return str(release.version)
    if not plan.steps:
        console.success(f"Nothing to do: '{release.dataset_id}' {release.version} is already in {store}")
    return str(release_dir(store.root, release))


@app.command("build-prepared-release", hidden=True)
def build_prepared_release_command(
    release: str,
    store: Path = typer.Option(..., "--store", help="The local registry to build into."),
    force: bool = typer.Option(False, "--force", help="Build the release again, even if the local registry holds it."),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep the raw download cache after building."),
    pinned_by: Path | None = typer.Option(
        None, "--pinned-by", help="A JSON file of the releases that read files of this release, with their pins."
    ),
) -> None:
    """Build one release into a local registry.

    The registry must already hold the release's parents and their payload sources. The command prints status
    and progress to stderr.
    """
    reference = DatasetRef.parse(release)
    pins = {} if pinned_by is None else _read_pins(pinned_by)
    with progress_renderer() as progress:
        options = BuildOptions(keep_cache=keep_cache, progress=progress, quiet=console.quiet)
        ConnectorBuilder().build(reference, LocalRegistry(store), force=force, pinned_by=pins, options=options)


def _read_pins(path: Path) -> dict[DatasetRef, PayloadSource]:
    """Read the pins that the parent of an isolated build wrote for this child.

    Returns:
        Each release that reads files of the build, with its pins.
    """
    pins = json.loads(path.read_text(encoding="utf-8"))
    return {DatasetRef.parse(key): PayloadSource.model_validate(value) for key, value in pins.items()}


def main() -> None:
    """Entry point for the ``timenet-build`` console script.

    Expected failures print a one-line message. Only unexpected errors surface a traceback.
    """
    run_cli(app)
