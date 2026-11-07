"""``timenet-build``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

import json
from pathlib import Path
from typing import cast

import typer

from timenet import BuildStep, Plan, Step, plan_build, publish
from timenet.builders import BuildOptions, declared
from timenet.cli.hints import Facts, cli_hints, registry_flag, sources_from_flags
from timenet.cli.options import AUTO_BUILD, LOCAL_REGISTRY, REMOTE_REGISTRY
from timenet.cli.runner import run_cli
from timenet.cli.ui import console, progress_renderer
from timenet.coordinator.build import run_plan
from timenet.errors import TimeNetVersionConflictError
from timenet.manifest import PayloadSource, same_release
from timenet.registry import LocalRegistry, WritableRegistry
from timenet.registry._paths import release_dir  # ruff: ignore[import-private-name]
from timenet.sources import InstalledConnectors, ReleaseSource
from timenet.types.references import DatasetRef, is_latest, split_ref
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.discovery import known_ids


app = typer.Typer(help="Build TimeNet datasets from connectors.", no_args_is_help=True)


@app.callback()
def _root(quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status output.")) -> None:
    """Build TimeNet datasets from connectors."""  # forces subcommand mode (so ``build`` is named)
    console.quiet = quiet


@app.command()
# ruff: ignore[too-many-arguments, too-many-positional-arguments] (a typer command: one parameter per option)
def build(
    release: str = typer.Argument(..., metavar="DATASET_ID[@VERSION]"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Build the release again, even if the local registry holds it."
    ),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="Keep the raw download cache after building (default: remove it)."
    ),
    isolation: bool | None = typer.Option(
        None,
        "--isolation/--no-isolation",
        help="Run the build in an environment built from the connector's requirements "
        "(default: $TIMENET_ISOLATION, else on).",
    ),
    publish_to: str | None = typer.Option(
        None,
        "--publish-to",
        metavar="URL",
        help="After the build, publish the release and its payload sources to this registry: a URL or a local path.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and stop."),
    local_registry: str | None = LOCAL_REGISTRY,
    remote_registry: str | None = REMOTE_REGISTRY,
    auto_build: bool | None = AUTO_BUILD,
) -> None:
    """Run a connector through the engine and write or publish its dataset.

    The build writes the release into the local registry, a dataset-layout directory, and prints its path to
    stdout. With ``--publish-to``, the release gets to that registry in two steps. First, the release
    builds into the local registry. Then a copy publishes it with its payload sources, and the command prints
    the version. If that registry already holds the version and ``--force`` is not set, nothing builds
    and nothing is copied. ``--dry-run`` prints the plan to stdout and runs nothing. Either way an
    emoji summary goes to stderr, so scripts can capture stdout. A built version is reused unless you
    give ``--force``.

    The build runs in an environment built from the connector's ``requirements.txt``. Pass
    ``--no-isolation`` (or set ``TIMENET_ISOLATION=off``) to run it in the current interpreter
    instead, which is what you want while writing a connector.

    A dataset whose card declares ``parents`` needs those exact parent releases in the local registry. A
    parent that the local registry lacks comes from the default list of sources, which ``--local-registry``,
    ``--remote-registry`` and ``--auto-build`` set. If no source provides a missing parent, the build
    fails before any download. The build prepares every parent before the dataset itself. It never
    builds a parent over a release that the local registry already holds. If the local registry already holds the
    version, the build needs only the payload sources of the version. It prepares them the same way
    as a parent.
    \f

    Raises:
        BadParameter: If ``dataset_id`` has no known connector, or a registry flag names no registry.
        TimeNetVersionUnavailableError: If the connector cannot build the requested version.
        TimeNetMissingParentError: If a parent is missing. The hint of the error names the flags.
        TimeNetFileError: If a copy of a parent in the local registry is damaged, or a copied file
            does not match its manifest record. The hint of the error names the flags.
        TimeNetAccessError: If ``--publish-to`` is remote and gets a release without access declared
            ``open``.
        TimeNetVersionConflictError: If the copy to ``--publish-to`` finds other files of the
            version, or of a payload source of it, in that registry. Nothing is copied then.
    """  # ruff: ignore[docstring-extraneous-exception, escape-sequence-in-docstring] - raised by the build and the publish; the form feed ends the help text
    # known_ids() imports no connector. So an unknown id fails before any environment exists.
    dataset_id, version = split_ref(release)
    known = known_ids()
    if dataset_id not in known:
        raise typer.BadParameter(f"no connector for {dataset_id!r}; known: {', '.join(known) or '(none)'}")
    sources = sources_from_flags(local_registry, remote_registry, auto_build)
    # The default list starts with the local registry, and the build has it already.
    store, parents_from = cast("LocalRegistry", sources[0]), sources[1:]
    dest = None if publish_to is None else registry_flag(publish_to, "--publish-to")
    facts = Facts(
        sources,
        store,
        request="build",
        local_registry=local_registry,
        remote_registry=remote_registry,
        dest=dest,
        wanted=release,
    )
    # The parent shows each copy and build step. An isolated child shows its own downloads and shards.
    with cli_hints(facts), progress_renderer() as progress:
        if not is_latest(version):
            declared(dataset_id, version)
        result = _build_and_publish(
            dataset_id,
            store=store,
            parents_from=parents_from,
            publish_to=dest,
            force=force,
            options=BuildOptions(
                keep_cache=keep_cache,
                progress=progress,
                quiet=console.quiet,
                isolation=None if isolation is None else ("on" if isolation else "off"),
            ),
            dry_run=dry_run,
        )
    if result is not None:
        typer.echo(result)


def _build_and_publish(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    store: LocalRegistry,
    parents_from: list[ReleaseSource],
    publish_to: WritableRegistry | None,
    force: bool,
    options: BuildOptions,
    dry_run: bool,
) -> str | None:
    """Build a release in the local registry and optionally publish it.

    The plan goes to the console before the first step. If nothing needs to run, for example because ``publish_to``
    holds the version and ``force`` is false, a status line says so.

    Args:
        dataset_id: The dataset ID.
        store: The local registry to build into.
        parents_from: Sources for missing parents and payload sources.
        publish_to: The destination registry, or ``None`` to keep the release local.
        force: Whether to rebuild a release already in the local registry.
        options: Options for each build.
        dry_run: Whether to return the plan without running it.

    Returns:
        The release path, the version when publishing, or the plan text (``None`` if empty) for a dry run.

    Raises:
        TimeNetVersionConflictError: If the local registry holds the release and ``publish_to`` holds it with other
            files or another card. Nothing is built then.
    """
    release = declared(dataset_id).release
    # With force, the build runs, and the publish compares its files with the destination.
    published = None if publish_to is None or force else publish_to.find_manifest(release)
    if publish_to is not None and published is not None:
        held = store.find_manifest(release)
        # If the local registry holds the release, the destination must hold the same one.
        if held is not None and not same_release(held, published):
            difference = "files" if held.files != published.files else "card"
            raise TimeNetVersionConflictError(release, str(publish_to), str(store), difference=difference)
        console.success(f"Nothing to do: '{release.dataset_id}' {release.version} is already in {publish_to}")
        return None if dry_run else str(release.version)
    # A destination that requires open access holds open releases only. So for such a destination, the
    # plan refuses another release before any download.
    plan = plan_build(
        dataset_id,
        local_registry=store,
        parents_from=parents_from,
        force=force,
        open_access_only=publish_to is not None and publish_to.requires_open_access,
    )
    if dry_run:
        lines = _plan_lines(plan)
        if publish_to is not None:
            lines.append(f"then publish {release} to {publish_to}")
        elif not plan.steps:
            console.success(f"Nothing to do: '{release.dataset_id}' {release.version} is already in {store}")
        return "\n".join(lines) or None
    if plan.steps:
        console.status("📋", f"Plan for '{release.dataset_id}' {release.version}:")
        for line in _plan_lines(plan):
            # Three columns, the width of the emoji and its space, put the tree under the text above it.
            console.detail(f"   {line}")
    run_plan(plan, store, options=options)
    if publish_to is not None:
        publish(str(release), dest=publish_to, source=store, progress=options.progress)
        return str(release.version)
    if not plan.steps:
        console.success(f"Nothing to do: '{release.dataset_id}' {release.version} is already in {store}")
    return str(release_dir(store.root, release))


_ACTIONS = {"read": "read from", "copy": "copy from", "build": "build with"}
"""Display text for each plan action."""

_BRANCHES = {"├── ": "│   ", "└── ": "    "}
"""Indentation below each tree branch."""


def _plan_lines(plan: Plan) -> list[str]:
    """Format a build plan as a tree with the requested release at the top.

    Each line shows a release, its action, and its source.
    Repeated releases show ``see above``.

    Args:
        plan: The build plan.

    Returns:
        The tree lines without newline characters.
    """
    steps = {step.release: step for step in plan.steps}
    needed = {release for step in plan.steps for release in step.depends_on}
    lines: list[str] = []
    drawn: set[DatasetRef] = set()

    def draw(step: Step, indent: str, branch: str) -> None:
        if step.release in drawn:
            lines.append(f"{indent}{branch}{step.release}: see above")
            return
        drawn.add(step.release)
        source = InstalledConnectors() if isinstance(step, BuildStep) else step.registry
        lines.append(f"{indent}{branch}{step.release}: {_ACTIONS[step.action]} {source}")
        below = [steps[release] for release in step.depends_on if release in steps]
        for number, child in enumerate(below, start=1):
            draw(child, indent + _BRANCHES.get(branch, ""), "└── " if number == len(below) else "├── ")

    for step in reversed(plan.steps):
        if step.release not in needed:
            draw(step, "", "")
    return lines


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
