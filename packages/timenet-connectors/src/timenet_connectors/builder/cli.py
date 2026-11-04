"""``timenet-build``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

import json
from pathlib import Path
import shlex
from typing import cast

import typer

from timenet import publish
from timenet.builders import BuildOptions, declared
from timenet.cli.options import AUTO_BUILD, LOCAL_REGISTRY, REMOTE_REGISTRY
from timenet.cli.runner import run_cli
from timenet.cli.ui import console, progress_renderer
from timenet.client import default_sources
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import BuildStep, Plan, Step
from timenet.errors import (
    TimeNetAccessError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.manifest import PayloadSource
from timenet.registry import LocalRegistry, WritableRegistry, open_registry
from timenet.registry._paths import release_dir  # ruff: ignore[import-private-name]
from timenet.sources import InstalledConnectors
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
        help="Run the build in an environment built from the connector's requirements.",
    ),
    publish_to: str | None = typer.Option(
        None,
        "--publish-to",
        metavar="URL|PATH",
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

    Raises:
        BadParameter: If ``dataset_id`` has no known connector.
        TimeNetVersionUnavailableError: If the connector cannot build the requested version.
        TimeNetMissingParentError: If a parent is missing. The hint of the error names the flags.
        TimeNetFileError: If a copy of a parent in the local registry is damaged, or a copied file
            does not match its manifest record. The hint of the error names the flags.
        TimeNetAccessError: If ``--publish-to`` is remote and gets a release without access declared
            ``open``.
        TimeNetVersionConflictError: If the copy to ``--publish-to`` finds other files of the
            version, or of a payload source of it, in that registry. Nothing is copied then.
    """  # ruff: ignore[docstring-extraneous-exception] - the publish raises TimeNetVersionConflictError
    # known_ids() imports no connector. So an unknown id fails before any environment exists.
    dataset_id, version = split_ref(release)
    known = known_ids()
    if dataset_id not in known:
        raise typer.BadParameter(f"no connector for {dataset_id!r}; known: {', '.join(known) or '(none)'}")
    # A connector builds only the version of its card, so "latest" asks for that version.
    wanted = declared(dataset_id, None if is_latest(version) else version).release
    sources = default_sources(local_registry=local_registry, remote_registry=remote_registry, auto_build=auto_build)
    # The default list starts with the local registry, which the build writes into. The planner leaves it out of
    # the sources of the copies and builds.
    store = cast("LocalRegistry", sources[0])
    dest = None if publish_to is None else open_registry(publish_to, label="--publish-to")
    has_connectors = any(isinstance(source, InstalledConnectors) for source in sources)
    try:
        if _already_published(wanted, dest, force=force):
            console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {dest}")
            if not dry_run:
                typer.echo(str(wanted.version))
            return
        # A destination that requires open access holds open releases only. So for such a destination, the
        # plan refuses another release before any download.
        plan = plan_release_build(
            str(wanted),
            store=store,
            parents_from=sources,
            force=force,
            open_access_only=dest is not None and dest.requires_open_access,
        )
        if dry_run:
            lines = _plan_lines(plan)
            if dest is not None:
                lines.append(f"then publish {wanted} to {dest}")
            elif not plan.steps:
                console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {store}")
            if lines:
                typer.echo("\n".join(lines))
            return
        _show_plan(wanted, plan)
        # The parent shows each copy and build step. An isolated child shows its own downloads and shards.
        with progress_renderer() as progress:
            options = BuildOptions(
                keep_cache=keep_cache,
                progress=progress,
                quiet=console.quiet,
                isolation=None if isolation is None else ("on" if isolation else "off"),
            )
            run_plan(plan, store, options=options)
            if dest is not None:
                publish(wanted, dest=dest, source=store, progress=progress)
    except TimeNetMissingParentError as error:
        error.hint = _missing_hint(has_connectors=has_connectors)
        raise
    except TimeNetFileError as error:
        flags = _registry_flags(local_registry=local_registry, remote_registry=remote_registry)
        error.hint = (
            f"Copy an intact release into the local registry with `timenet download {error.release}{flags}`, or rebuild "
            f"it with `timenet-build build {error.release.dataset_id} --force{flags}`. A rebuild changes its "
            "files, and the releases that pin the old files then fail to open."
        )
        raise
    except TimeNetAccessError as error:
        error.hint = _access_hint(dest, has_connectors=has_connectors)
        raise
    if dest is None and not plan.steps:
        console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {store}")
    # A publish prints the version, and a local build prints the release directory.
    typer.echo(str(wanted.version) if dest is not None else str(release_dir(store.root, wanted)))


def _missing_hint(*, has_connectors: bool) -> str:
    """Say how a build gets a parent that no source of its list provides.

    Args:
        has_connectors: Whether the list of sources holds the installed connectors.

    Returns:
        The hint, a full sentence that names ``--auto-build`` too if the list has no connectors.
    """
    ways = (
        "copy it from a registry that holds it with --remote-registry <url>, or publish it to such a registry "
        "first with `timenet publish`."
    )
    if not has_connectors:
        return f"Build it from its installed connector with --auto-build, {ways}"
    return f"{ways[0].upper()}{ways[1:]}"


def _access_hint(dest: WritableRegistry | None, *, has_connectors: bool) -> str | None:
    """Say how a build gets a release that is not open.

    Args:
        dest: The registry that ``--publish-to`` names, or ``None``.
        has_connectors: Whether the list of sources holds the installed connectors.

    Returns:
        The hint as a full sentence, or ``None`` if no flag of the command helps.
    """
    if dest is not None and dest.requires_open_access:
        return (
            f"{dest} serves open releases only, so --publish-to takes open releases only. Build it into the "
            "local registry without --publish-to instead."
        )
    if not has_connectors:
        return "Build it from its installed connector with --auto-build."
    return None


def _registry_flags(*, local_registry: str | None, remote_registry: str | None) -> str:
    """Repeat the registry flags that the command got, for a command in a hint.

    Args:
        local_registry: The value of ``--local-registry``, or ``None``.
        remote_registry: The value of ``--remote-registry``, or ``None``.

    Returns:
        Each given flag with its quoted value and a space before it, or an empty string.
    """
    flags = (("--local-registry", local_registry), ("--remote-registry", remote_registry))
    return "".join(f" {flag} {shlex.quote(value)}" for flag, value in flags if value is not None)


def _already_published(release: DatasetRef, dest: WritableRegistry | None, *, force: bool) -> bool:
    """Check whether the destination of the publish holds the release, so that nothing needs to run.

    Args:
        release: The release to build.
        dest: The publish destination, or ``None``.
        force: Whether the build runs again anyway.

    Returns:
        ``True`` if ``dest`` holds the release and ``force`` is false.
    """
    return dest is not None and not force and dest.exists(release.dataset_id, str(release.version))


def _show_plan(release: DatasetRef, plan: Plan) -> None:
    """Show the steps of a plan on the console before the first step runs.

    Args:
        release: The requested release.
        plan: The plan.
    """
    if not plan.steps:
        return
    console.status("📋", f"Plan for '{release.dataset_id}' {release.version}:")
    for line in _plan_lines(plan):
        # Three columns, the width of the emoji and its space, put the tree under the text above it.
        console.detail(f"   {line}")


_ACTIONS = {"read": "read from", "copy": "copy from", "build": "build with"}
"""The words of each action of a step in a line of a plan."""

_BRANCHES = {"├── ": "│   ", "└── ": "    "}
"""For each branch of a tree line: what the lines below it show in its column."""


def _plan_lines(plan: Plan) -> list[str]:
    """Draw a plan as a tree, with the requested release at the top.

    Below each release come the releases that it needs. Each line names the release, the action and
    the source, for example ``<release>: copy from <registry>``, or ``<release>: see above`` the second
    time.

    Args:
        plan: The plan.

    Returns:
        One line for each node of the tree, without line ends.
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
