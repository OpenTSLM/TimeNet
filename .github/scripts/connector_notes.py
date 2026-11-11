# /// script
# requires-python = ">=3.11"
# ///
"""Print the release notes of timenet-connectors between two refs.

The notes list the connectors that were added, changed or removed, and the changes to the shared
connector code (the "framework"). Core changes that only touched connector files on the way are
left out; the timenet notes list them.

- A connector is a ``datasets/<org>/<name>/`` folder with a ``dataset.yaml`` card, the rule that
  ``scripts/gen_dataset_docs.py`` uses. Connectors are matched by the card's ``dataset_id``, so a
  moved folder that keeps its id is changed, and a new id is one removal and one addition.
- An added connector cites the PR that added it, a removed one the PR that removed it. A changed
  connector cites its connector PRs, and is left out when it has none.
- The framework section lists the connector PRs outside ``datasets/``, breaking ones first.
- A connector PR is a ``feat``, a ``fix`` or a breaking change that did not touch
  ``packages/timenet``. A core PR edits connectors on the way, and the timenet notes list it.

Called by .github/workflows/release.yml. Run it the way the workflow does::

    uv run --no-project .github/scripts/connector_notes.py v0.11.0 HEAD \
        --release timenet-connectors-v0.12.0
"""

import argparse
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argument vectors, never a shell string


PACKAGE = "packages/timenet-connectors"
DATASETS = f"{PACKAGE}/src/timenet_connectors/datasets"
PR_URL = "https://github.com/OpenTSLM/TimeNet/pull/"
# A squash-merged PR title: type(scope)!: description (#123).
SUBJECT = re.compile(r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<text>.+?)(?: \(#(?P<pr>\d+)\))?$")
# The commits cliff.toml skips: release and development-version commits.
SKIP = re.compile(r"^chore(\(release\)|: start .* development)")


def git(*args: str) -> str:
    """Run git and return its output.

    Returns:
        The standard output.
    """
    command = [shutil.which("git") or "git", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # ruff: ignore[subprocess-without-shell-equals-true]


def commits(base: str, head: str, *paths: str) -> list[tuple[str, str]]:
    """List the commits between ``base`` and ``head`` that touched ``paths``, oldest first.

    Returns:
        ``(short hash, subject)`` pairs, without release and development-version commits.
    """
    lines = git("log", "--reverse", "--format=%h %s", f"{base}..{head}", "--", *paths).splitlines()
    pairs = [line.split(" ", 1) for line in lines]
    return [(sha, subject) for sha, subject in pairs if not SKIP.match(subject)]


def notable(subject: str) -> bool:
    """Return whether a commit is a feature, a fix or a breaking change."""
    match = SUBJECT.match(subject)
    return bool(match) and (match["type"] in {"feat", "fix"} or bool(match["bang"]))


def link(sha: str, subject: str) -> str:
    """Link the PR of a commit, or name the commit when it has no PR number.

    Returns:
        A markdown link, or the short hash.
    """
    number = re.search(r"\(#(\d+)\)$", subject)
    return f"[#{number[1]}]({PR_URL}{number[1]})" if number else sha


def connectors(ref: str) -> dict[str, tuple[str, str]]:
    """Map each dataset id at ``ref`` to its folder and display name.

    Returns:
        ``{dataset_id: (folder, name)}``.
    """
    found = {}
    for card in git("ls-tree", "-r", "--name-only", ref, "--", DATASETS).splitlines():
        if not re.fullmatch(re.escape(DATASETS) + r"/[^/]+/[^/]+/dataset\.yaml", card):
            continue
        text = git("show", f"{ref}:{card}")
        folder = card.rsplit("/", 1)[0]
        dataset_id = re.search(r"^dataset_id:\s*(\S+)", text, re.M)
        name = re.search(r"^name:\s*(.+)$", text, re.M)
        key = dataset_id[1] if dataset_id else "/".join(folder.split("/")[-2:])
        found[key] = (folder, name[1].strip() if name else key)
    return found


def connector_sections(base: str, head: str, core: set[str]) -> tuple[dict[str, list[str]], set[str]]:
    """List the connectors that were added, changed or removed.

    Returns:
        The entries of each section, and the commits that added a connector.
    """
    old, new = connectors(base), connectors(head)
    sections: dict[str, list[str]] = {"Added connectors": [], "Changed connectors": [], "Removed connectors": []}
    added: set[str] = set()
    for dataset_id in sorted(old.keys() | new.keys()):
        folders = sorted({entry[0] for entry in (old.get(dataset_id), new.get(dataset_id)) if entry})
        touched = commits(base, head, *folders)
        if dataset_id not in old:
            section, cited = "Added connectors", touched[:1]
            added.update(sha for sha, _ in cited)
        elif dataset_id not in new:
            section, cited = "Removed connectors", touched[-1:]
        else:
            section, cited = "Changed connectors", [c for c in touched if c[0] not in core and notable(c[1])]
        if cited:
            name = (new.get(dataset_id) or old[dataset_id])[1]
            links = ", ".join(dict.fromkeys(link(*c) for c in cited))
            sections[section].append(f"- {name} (`{dataset_id}`): {links}")
    return sections, added


def framework_entries(base: str, head: str, skip: set[str]) -> list[str]:
    """List the connector PRs outside ``datasets/``, breaking ones first.

    Returns:
        The entries, without the commits in ``skip``.
    """
    entries = []
    for sha, subject in commits(base, head, PACKAGE, f":(exclude){DATASETS}"):
        match = SUBJECT.match(subject)
        if match and sha not in skip and notable(subject):
            breaking = "**Breaking:** " if match["bang"] else ""
            scope = f"**{match['scope']}:** " if match["scope"] else ""
            text = match["text"][:1].upper() + match["text"][1:]
            entries.append((not match["bang"], f"- {breaking}{scope}{text} ({link(sha, subject)})"))
    return [entry for _, entry in sorted(entries, key=lambda pair: pair[0])]


def notes(base: str, head: str) -> str:
    """Render the notes as markdown.

    Returns:
        The markdown, or an empty string when nothing in the package changed.
    """
    core = {sha for sha, _ in commits(base, head, "packages/timenet")}
    sections, added = connector_sections(base, head, core)
    sections["Connector framework"] = framework_entries(base, head, core | added)
    out = []
    for title, entries in sections.items():
        if entries:
            out += [f"### {title}", "", *entries, ""]
    return "\n".join(out)


def badge(release: str) -> str:
    """Render the PyPI badge of a ``<package>-vX.Y.Z`` tag, like the one cliff.toml writes.

    Returns:
        The markdown badge and a blank line.
    """
    package, version = release.rsplit("-v", 1)
    label = package.replace("-", "--")
    image = f"https://img.shields.io/badge/{label}-{version}-blue?logo=pypi&logoColor=white"
    return f"[![{package} {version}]({image})](https://pypi.org/project/{package}/{version}/)\n\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="the last release tag")
    parser.add_argument("head", help="the commit to release")
    parser.add_argument(
        "--release", help="the release tag, for example timenet-connectors-v0.12.0, to add a PyPI badge"
    )
    args = parser.parse_args()
    print((badge(args.release) if args.release else "") + notes(args.base, args.head), end="")
