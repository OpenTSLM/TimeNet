# /// script
# requires-python = ">=3.11"
# dependencies = ["GitPython>=3.1"]
#
# [tool.ty.rules]
# unresolved-import = "ignore"
# ///
"""Write the release notes of timenet-connectors between two refs.

The notes list the connectors that were added, changed, or removed. They also list the changes to
the shared connector code (the "framework"). They leave out core changes that only touched
connector files on the way. The timenet notes list those changes.

- A connector is a ``datasets/<org>/<name>/`` folder with a ``dataset.yaml`` card.
  ``scripts/gen_dataset_docs.py`` uses the same rule. The ``dataset_id`` in the card identifies the
  connector. A moved folder that keeps its id is a change. A new id is one removal and one addition.
- An added connector names the PR that added it. A removed connector names the PR that removed it.
  A changed connector names its connector PRs. A changed connector with no connector PR is not in
  the notes.
- The framework section lists the connector PRs outside ``datasets/``. Breaking changes come first.
- A connector PR is a ``feat``, a ``fix``, or a breaking change that did not touch
  ``packages/timenet``. A core PR also edits connectors on the way. The timenet notes list it.

.github/scripts/release.py imports this module.
"""

from functools import cache
import re

from git import Repo


PACKAGE = "packages/timenet-connectors"
DATASETS = f"{PACKAGE}/src/timenet_connectors/datasets"
# The title of a squash-merged PR: type(scope)!: description (#123).
SUBJECT = re.compile(r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<text>.+?)(?: \(#(?P<pr>\d+)\))?$")
# The commits that cliff.toml skips: release commits and development-version commits.
SKIP = re.compile(r"^chore(\(release\)|: start .* development)")


@cache
def repo() -> Repo:
    """Open the repository around the current directory.

    Returns:
        The repository.
    """
    return Repo(search_parent_directories=True)


def commits(base: str, head: str, *paths: str) -> list[tuple[str, str]]:
    """List the commits between ``base`` and ``head`` that touched ``paths``. The oldest is first.

    Returns:
        ``(short hash, subject)`` pairs. Release commits and development-version commits are not in
        the list.
    """
    found = repo().iter_commits(f"{base}..{head}", paths=list(paths), reverse=True)
    pairs = [(commit.hexsha[:9], str(commit.summary)) for commit in found]
    return [(sha, subject) for sha, subject in pairs if not SKIP.match(subject)]


def notable(subject: str) -> bool:
    """Return whether a commit is a feature, a fix or a breaking change."""
    match = SUBJECT.match(subject)
    return bool(match) and (match["type"] in {"feat", "fix"} or bool(match["bang"]))


def link(sha: str, subject: str) -> str:
    """Name the PR of a commit. GitHub turns the name into a link. If there is no PR number, name the commit.

    Returns:
        ``#123`` or the short hash.
    """
    number = re.search(r"\(#(\d+)\)$", subject)
    return f"#{number[1]}" if number else sha


def connectors(ref: str) -> dict[str, tuple[str, str]]:
    """Map each dataset id at ``ref`` to its folder and display name.

    Returns:
        ``{dataset_id: (folder, name)}``.
    """
    found = {}
    try:
        datasets = repo().commit(ref).tree / DATASETS
    except KeyError:
        return found
    for folder in (folder for org in datasets.trees for folder in org.trees):
        try:
            text = (folder / "dataset.yaml").data_stream.read().decode()
        except KeyError:
            continue
        dataset_id = re.search(r"^dataset_id:\s*(\S+)", text, re.M)
        name = re.search(r"^name:\s*(.+)$", text, re.M)
        key = dataset_id[1] if dataset_id else "/".join(folder.path.split("/")[-2:])
        found[key] = (folder.path, name[1].strip() if name else key)
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
    """List the connector PRs outside ``datasets/``. Breaking changes come first.

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
        The markdown, or an empty string if the package has no changes.
    """
    core = {sha for sha, _ in commits(base, head, "packages/timenet")}
    sections, added = connector_sections(base, head, core)
    sections["Connector framework"] = framework_entries(base, head, core | added)
    out = []
    for title, entries in sections.items():
        if entries:
            out += [f"### {title}", "", *entries, ""]
    return "\n".join(out)
