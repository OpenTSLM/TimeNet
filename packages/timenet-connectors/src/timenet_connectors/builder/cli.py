"""``timenet-build``: the producer CLI that runs connectors through the engine.

This is not the consumer ``timenet`` CLI. A ``build`` writes a dataset-layout directory that the SDK
can load. That directory is a valid local registry. The CLI resolves connectors by dataset id at run
time. To add one, drop a ``datasets/<org>/<name>/`` package with a ``connector.py`` that exposes
``CONNECTOR`` and a ``dataset.yaml`` card. You do not register it here.
"""

import json
from pathlib import Path
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argument vectors, never a shell string

import typer
from uv import find_uv_bin

from timenet import publish
from timenet._hints import (  # ruff: ignore[import-private-name]
    CliWording,
    Facts,
    Operation,
    RequestedRelease,
    explain,
    use_wording,
)
from timenet.builders import BuildOptions, declared
from timenet.cli.hints import local_registry_from_flag, registry_flag, sources_from_flags
from timenet.cli.options import AUTO_BUILD, LOCAL_REGISTRY, REMOTE_REGISTRY
from timenet.cli.runner import run_cli
from timenet.cli.ui import console, progress_renderer
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import BuildStep, Plan, Step
from timenet.errors import TimeNetIsolationError, TimeNetVersionConflictError
from timenet.manifest import PayloadSource
from timenet.registry import LocalRegistry, WritableRegistry
from timenet.registry._paths import release_dir  # ruff: ignore[import-private-name]
from timenet.sources import InstalledConnectors
from timenet.types.references import DatasetRef, is_latest, split_ref
from timenet_connectors import datasets
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.discovery import find_card_paths, known_ids


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
    \f

    Raises:
        BadParameter: If ``dataset_id`` has no known connector, or a registry flag names no registry.
        TimeNetVersionUnavailableError: If the connector cannot build the requested version.
        TimeNetMissingParentError: If a parent is missing. The hint of the error names the flags.
        TimeNetFileError: If a copy of a parent in the local registry is damaged, or a copied file
            does not match its manifest record. The hint of the error names the flags.
        TimeNetAccessError: If a registry lists a parent but does not serve it to this caller.
        TimeNetVersionConflictError: If the copy to ``--publish-to`` finds other files of the
            version, or of a payload source of it, in that registry. Nothing is copied then.
    """  # ruff: ignore[docstring-extraneous-exception, escape-sequence-in-docstring] - raised by the build and the publish; the form feed ends the help text
    # known_ids() imports no connector. So an unknown id fails before any environment exists.
    dataset_id, version = split_ref(release)
    known = known_ids()
    if dataset_id not in known:
        raise typer.BadParameter(f"no connector for {dataset_id!r}; known: {', '.join(known) or '(none)'}")
    # The build writes into the local registry, which starts the list. The planner leaves it out of the sources of
    # the copies and builds.
    local = local_registry_from_flag(local_registry)
    sources = sources_from_flags(local, remote_registry, auto_build)
    dest = None if publish_to is None else registry_flag(publish_to, "--publish-to")
    facts = Facts(sources, local, request=Operation.BUILD, dest=dest, wanted=RequestedRelease(dataset_id, version))
    with use_wording(CliWording(local_registry, remote_registry)):
        with explain(facts):
            # A connector builds only the version of its card, so "latest" asks for that version.
            wanted = declared(dataset_id, None if is_latest(version) else version).release
            if _already_published(wanted, local, dest, force=force):
                console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {dest}")
                if not dry_run:
                    typer.echo(str(wanted.version))
                return
            plan = plan_release_build(str(wanted), target_registry=local, parents_from=sources, force=force)
        if dry_run:
            lines = _plan_lines(plan)
            if dest is not None:
                lines.append(f"then publish {wanted} to {dest}")
            elif not plan.steps:
                console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {local}")
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
            with explain(facts):
                run_plan(plan, local, options=options)
            if dest is not None:
                # The publish gives its own hint, the hint of a publish.
                publish(wanted, dest=dest, source=local, progress=progress)
    if dest is None and not plan.steps:
        console.success(f"Nothing to do: '{wanted.dataset_id}' {wanted.version} is already in {local}")
    # A publish prints the version, and a local build prints the release directory.
    typer.echo(str(wanted.version) if dest is not None else str(release_dir(local.root, wanted)))


def _already_published(
    release: DatasetRef, local_registry: LocalRegistry, dest: WritableRegistry | None, *, force: bool
) -> bool:
    """Check whether the destination of the publish holds the release, so that nothing needs to run.

    Args:
        release: The release to build.
        local_registry: The local registry to build into.
        dest: The publish destination, or ``None``.
        force: Whether the build runs again anyway, so that the publish compares its files with ``dest``.

    Returns:
        ``True`` if ``dest`` holds the release and ``force`` is false.

    Raises:
        TimeNetVersionConflictError: If the local registry holds the release and ``dest`` holds it with other
            files or another card. Nothing is built then.
    """
    published = None if dest is None or force else dest.find_manifest(release)
    if dest is None or published is None:
        return False
    held = local_registry.find_manifest(release)
    # If the local registry holds the release, the destination must hold the same one.
    difference = None if held is None else held.difference_from(published)
    if difference is not None:
        raise TimeNetVersionConflictError(release, str(dest), dest.location, str(local_registry), difference)
    return True


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


@app.command(hidden=True)
def check(
    datasets_dirs: list[Path] | None = typer.Argument(
        None,
        metavar="[DATASETS_DIR]...",
        help="A datasets directory (default: the built-in datasets directory).",
        exists=True,
        file_okay=False,
    ),
) -> None:
    """Run the tests and type check of each connector in an environment built from its requirements.

    A connector directory has a ``dataset.yaml`` card and a ``tests`` directory. For each connector,
    the command runs pytest on its tests and ``ty`` on its directory. Both run through ``uv run`` in the
    current project environment, with the requirements.txt of the connector added if it exists. Run the command from the project
    whose environment has ``timenet``, pytest and ``ty``.
    \f

    Raises:
        BadParameter: If no connector with a ``tests`` directory is found.
        TimeNetIsolationError: If the uv executable is missing.
        Exit: With code 1 if a check failed.
    """  # ruff: ignore[escape-sequence-in-docstring] - the form feed ends the help text
    datasets_dirs = datasets_dirs or [Path(datasets.__path__[0])]
    connectors: list[tuple[str, Path]] = []
    for datasets_dir in datasets_dirs:
        card_paths = find_card_paths(datasets_dir)
        connectors += [
            (str(card_path.parent.relative_to(datasets_dir)), card_path.parent)
            for card_path in card_paths
            if (card_path.parent / "tests").is_dir()
        ]
    if not connectors:
        raise typer.BadParameter(
            f"found no connectors with a tests directory under {', '.join(map(str, datasets_dirs))}"
        )
    try:
        uv = find_uv_bin()
    except FileNotFoundError as exc:
        raise TimeNetIsolationError("cannot run connector checks: no uv executable found.") from exc
    failures: list[str] = []
    for name, connector_dir in connectors:
        typer.echo(f"\n=== {name} ===")
        uv_run = [uv, "run"]
        if (requirements := connector_dir / "requirements.txt").is_file():
            uv_run += ["--with-requirements", str(requirements)]
        for command in (["pytest", str(connector_dir / "tests"), "-q"], ["ty", "check", str(connector_dir)]):
            # ruff: ignore[subprocess-without-shell-equals-true]
            if subprocess.run([*uv_run, *command], check=False).returncode != 0:
                failures.append(f"{name}: {command[0]}")
    if failures:
        typer.echo("\nconnector checks failed:")
        for failure in failures:
            typer.echo(f"  - {failure}")
        raise typer.Exit(1)
    typer.echo("\nall connector checks passed")


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
