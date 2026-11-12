# /// script
# requires-python = ">=3.11"
# dependencies = ["GitPython>=3.1", "PyGithub>=2.5"]
#
# [tool.ty.rules]
# unresolved-import = "ignore"
# ///
"""The steps of .github/workflows/release.yml that need logic.

- ``prepare`` sets the versions, writes the release notes, and opens the release PR. The PR body
  has the notes between ``<!-- release-notes:<package> -->`` markers, so you can edit them there.
- ``released`` finds the packages whose version the merged release PR changed.
- ``smoke`` imports each built wheel and sdist in an isolated environment. It makes sure that each
  file records the commit that it was built from.
- ``github-release`` creates the tag and the GitHub release of each released package. It reads the
  notes from the merged PR and attaches the files in ``dist/``.

PyGithub makes the GitHub calls and GitPython makes the git calls. Only ``uv`` and git-cliff run as
subprocesses, because they have no Python API. The option ``--dry-run`` does not push and does not
open a PR. It changes the working tree, so run it in a scratch checkout::

    GIT_CLIFF=git-cliff==2.14.2 uv run .github/scripts/release.py prepare --package both --dry-run
"""

import argparse
import os
from pathlib import Path
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors
import tomllib
from typing import NoReturn

from connector_notes import commits, notes as connector_notes, repo
from git import Repo
from github import Auth, Github, UnknownObjectException
from github.Repository import Repository


PACKAGES = ("timenet", "timenet-connectors")
CONNECTORS = "packages/timenet-connectors/pyproject.toml"
FLOOR = re.compile(r'"timenet\[build\]>=[^"]*"')
VERSION = re.compile(r"\d+\.\d+\.\d+")
NOTES = re.compile(r"<!-- release-notes:(?P<package>[\w-]+) -->\n(?P<text>.*?)\n<!-- /release-notes -->", re.S)
EXPLAIN = (
    "Merge this PR to tag each package below and to create its GitHub release. The release notes are\n"
    "the text between the `release-notes` markers. Edit the notes here before you merge. The same run\n"
    "then waits for your approval to upload to TestPyPI. It waits for your approval again to upload\n"
    "to PyPI.\n"
)


def fail(message: str) -> NoReturn:
    """Stop the step with an error message."""
    raise SystemExit(message)


def run(*command: str) -> str:
    """Run a tool and return its output. Stop the step with the output of the tool when it fails.

    Returns:
        The standard output without the trailing newline.
    """
    result = subprocess.run(command, check=False, capture_output=True, text=True)  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed argument vector
    if result.returncode:
        fail(f"{' '.join(command)} failed with exit code {result.returncode}:\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def version_of(package: str) -> str:
    """Read the version of a workspace package.

    Returns:
        The version in its ``pyproject.toml``.
    """
    return run("uv", "version", "--package", package, "--short", "--no-sync")


def cliff(package: str, *args: str) -> str:
    """Run git-cliff for one package. The legacy ``vX.Y.Z`` tags are the last release of both packages.

    Returns:
        Its output.
    """
    pattern = rf"^({package}-)?v[0-9]+\.[0-9]+\.[0-9]+$"
    return run(
        "uvx", os.environ["GIT_CLIFF"], "--include-path", f"packages/{package}/**", "--tag-pattern", pattern, *args
    )


def last_tag(git: Repo, package: str, ref: str = "HEAD") -> str:
    """Find the newest release tag of a package that ``ref`` can reach.

    Returns:
        The tag name.
    """
    return git.git.describe("--tags", "--abbrev=0", "--match", f"{package}-v*", "--match", "v*", ref)


def append(variable: str, text: str) -> None:
    """Append text to the file that GitHub names in ``variable``. Print the text outside a workflow."""
    path = os.environ.get(variable)
    if path:
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(text)
    else:
        print(text, end="")


def github() -> Repository:
    """Open the repository on GitHub with the token in ``GH_TOKEN``.

    Returns:
        The repository.
    """
    return Github(auth=Auth.Token(os.environ["GH_TOKEN"])).get_repo(os.environ["GITHUB_REPOSITORY"])


def open_pr(branch: str, title: str, body: str, *, dry_run: bool) -> None:
    """Commit the working tree to ``branch``, push it, and open its PR against main.

    Update the PR if it is already open.
    """
    if dry_run:
        print(f"{title}\n{branch}\n\n{body}")
        return
    git = repo()
    with git.config_writer() as config:
        config.set_value("user", "name", "github-actions[bot]")
        config.set_value("user", "email", "41898282+github-actions[bot]@users.noreply.github.com")
    git.git.switch("-c", branch)
    git.git.commit("-am", title)
    git.git.push("--force", "origin", branch)
    remote = github()
    open_prs = list(remote.get_pulls(state="open", head=f"{remote.owner.login}:{branch}"))
    if open_prs:
        pull = open_prs[0]
        pull.edit(body=body)
    else:
        pull = remote.create_pull(title=title, body=body, head=branch, base="main")
    append("GITHUB_STEP_SUMMARY", f"### {title}\n\n{pull.html_url}\n")


def next_version(git: Repo, package: str, requested: str) -> str:
    """Choose the version to release, make sure that it is new, and write it to the package.

    Returns:
        The version.
    """
    tag_names = {tag.name for tag in git.tags}
    version = (requested or cliff(package, "--bumped-version")).rpartition("v")[2]
    if not VERSION.fullmatch(version):
        fail(f"'{version}' is not X.Y.Z.")
    for tag in (f"{package}-v{version}", f"v{version}"):
        if tag in tag_names:
            fail(f"Tag {tag} already exists. Is {package} unchanged since then?")
    run("uv", "version", "--package", package, version, "--frozen")
    return version


def badge(package: str, version: str) -> str:
    """Render the PyPI badge of a release.

    Returns:
        The markdown badge and a blank line.
    """
    label = package.replace("-", "--")
    image = f"https://img.shields.io/badge/{label}-{version}-blue?logo=pypi&logoColor=white"
    return f"[![{package} {version}]({image})](https://pypi.org/project/{package}/{version}/)\n\n"


def package_notes(git: Repo, package: str, version: str) -> str:
    """Write the notes section of a release. The part that you can edit is between markers.

    Returns:
        The markdown.
    """
    if package == "timenet-connectors":
        body = connector_notes(last_tag(git, package), "HEAD")
    else:
        body = cliff(package, "--unreleased", "--strip", "header")
    notes = badge(package, version) + body
    return f"## {package} {version}\n\n<!-- release-notes:{package} -->\n{notes.strip()}\n<!-- /release-notes -->\n\n"


def raise_floor() -> None:
    """Make timenet-connectors require the timenet release that goes with it."""
    floor = f'"timenet[build]>={version_of("timenet")}"'
    path = Path(CONNECTORS)
    text = FLOOR.sub(floor, path.read_text(encoding="utf-8"))
    if floor not in text:
        fail(f"{CONNECTORS} has no timenet floor to raise.")
    path.write_text(text, encoding="utf-8")


def leftovers_warning(git: Repo, package: str) -> str:
    """Warn when the other package has changes since its last release. Use this for one package.

    Returns:
        A GitHub alert, or an empty string.
    """
    other = "timenet-connectors" if package == "timenet" else "timenet"
    last = last_tag(git, other)
    pending = commits(last, "HEAD", f"packages/{other}")
    if not pending:
        return ""
    floor = FLOOR.findall(Path(CONNECTORS).read_text(encoding="utf-8"))[0]
    return (
        f"> [!WARNING]\n> {other} has {len(pending)} commits since {last}. This release leaves them out.\n"
        f"> timenet-connectors keeps {floor}. If those commits belong with the {package} changes,\n"
        "> release both packages instead.\n\n"
    )


def prepare(package: str, requested: str, *, dry_run: bool) -> None:
    """Set the versions, write the notes, and open the release PR."""
    git = repo()
    names = PACKAGES if package == "both" else (package,)
    versions = {name: next_version(git, name, requested) for name in names}
    sections = "".join(package_notes(git, name, version) for name, version in versions.items())
    if package == "both":
        raise_floor()
        warning = ""
    else:
        warning = leftovers_warning(git, package)
    run("uv", "lock")
    title = "chore(release): " + ", ".join(f"{name} {version}" for name, version in versions.items())
    branch = "release/" + "-".join(f"{name}-v{version}" for name, version in versions.items())
    open_pr(branch, title, f"{EXPLAIN}\n---\n\n{warning}{sections}", dry_run=dry_run)


def released() -> None:
    """Find the packages whose version the merged release PR changed. Write them to the outputs."""
    git = repo()
    found = {}
    for package in PACKAGES:
        version = version_of(package)
        before = (git.commit("HEAD^").tree / f"packages/{package}/pyproject.toml").data_stream.read().decode()
        if version == tomllib.loads(before)["project"]["version"]:
            continue
        if not VERSION.fullmatch(version):
            fail(f"{package} is at {version}, not X.Y.Z.")
        found[package] = version
    if not found:
        fail("This PR changed no package version.")
    append("GITHUB_OUTPUT", "released=" + " ".join(f"{name}={version}" for name, version in found.items()) + "\n")


def merged_notes(remote: Repository, sha: str) -> dict[str, str]:
    """Read the release notes from the body of the merged release PR.

    Returns:
        The notes of each package.
    """
    for pull in remote.get_commit(sha).get_pulls():
        if pull.merged and pull.head.ref.startswith("release/"):
            return {
                match["package"]: match["text"] for match in NOTES.finditer((pull.body or "").replace("\r\n", "\n"))
            }
    fail(f"No merged release PR holds {sha}.")


def github_release(entries: dict[str, str]) -> None:
    """Create the tag and the GitHub release of each package. Attach its files from ``dist/``.

    A re-run creates only what is missing.
    """
    git = repo()
    sha = git.head.commit.hexsha
    remote = github()
    notes = merged_notes(remote, sha)
    for package, version in entries.items():
        tag = f"{package}-v{version}"
        if package not in notes:
            fail(f"The release PR has no notes for {package}.")
        try:
            release = remote.get_release(tag)
            if remote.get_git_ref(f"tags/{tag}").object.sha != sha:
                fail(f"{tag} exists on another commit.")
        except UnknownObjectException:
            release = remote.create_git_release(tag, f"{package} {version}", notes[package], target_commitish=sha)
        attached = {asset.name for asset in release.get_assets()}
        stem = package.replace("-", "_")
        files = [*Path("dist").glob(f"{stem}-{version}-*.whl"), *Path("dist").glob(f"{stem}-{version}.tar.gz")]
        for file in files:
            if file.name not in attached:
                release.upload_asset(str(file))
        indexes = f"[TestPyPI](https://test.pypi.org/project/{package}/{version}/), [PyPI](https://pypi.org/project/{package}/{version}/)"
        append("GITHUB_STEP_SUMMARY", f"- {package} {version}: {release.html_url} ({indexes} once uploaded)\n")


SMOKE = """
import {module}
from timenet.manifest import GitSource
from timenet.provenance import PackageSnapshot
commit = PackageSnapshot.read({dist!r}).source
expected = GitSource(sha={sha!r}, contains_local_changes=False)
assert commit == expected, (commit, expected)
print('smoke ok:', {module}.__name__, commit)
"""


def smoke() -> None:
    """Import each built wheel and sdist. Make sure that each file records this commit, clean.

    timenet-connectors uses the timenet wheel from the same run, if there is one. If there is none,
    timenet comes from the index. Then timenet must be on the index already.
    """
    sha = repo().head.commit.hexsha
    stems = {package: package.replace("-", "_") for package in PACKAGES}
    core_wheels = sorted(Path("dist").glob(f"{stems[PACKAGES[0]]}-*.whl"))
    for package, stem in stems.items():
        extras = core_wheels if package == PACKAGES[1] else []
        for artifact in sorted(Path("dist").glob(f"{stem}-*")):
            command = ["uv", "run", "--isolated", "--no-project"]
            for file in (*extras, artifact):
                command += ["--with", str(file)]
            print(f"::group::{artifact.name}")
            print(run(*command, "python", "-c", SMOKE.format(module=stem, dist=package, sha=sha)))
            print("::endgroup::")


def parse(text: str) -> dict[str, str]:
    """Read ``package=version`` pairs.

    Returns:
        The versions by package.
    """
    return dict(pair.split("=") for pair in text.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--package", choices=["both", *PACKAGES], required=True)
    prepare_parser.add_argument("--version", default="")
    prepare_parser.add_argument("--dry-run", action="store_true")
    commands.add_parser("released")
    commands.add_parser("smoke")
    commands.add_parser("github-release").add_argument("--released", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.package, args.version, dry_run=args.dry_run)
    elif args.command == "released":
        released()
    elif args.command == "smoke":
        smoke()
    else:
        github_release(parse(args.released))
