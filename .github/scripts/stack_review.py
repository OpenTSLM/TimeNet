# /// script
# requires-python = ">=3.11"
# dependencies = ["PyGithub>=2.9"]
#
# [tool.ty.rules]
# unresolved-import = "ignore"
# ///
"""Review a whole stack of pull requests with PR-Agent's local provider.

Called by ``.github/workflows/pr-agent-stack.yaml``. Starting from one PR it walks down to the
PR whose base is the target branch, and up through every open PR built on top of it. For each
path from the bottom PR to a top PR it checks out the top branch, points a local base branch at
the bottom PR's target, and runs ``pr-agent review`` on that combined diff. The result becomes one
comment per path, on the top PR, found again by a marker so reruns edit it in place. A path is
skipped when no PR head in it changed since the comment was written.

The script is trusted code. The workflow runs it from the default branch, never from the PR
under review, and the PR-Agent configuration is built here for the same reason. The review prompt
is ``.github/pr-review-instructions.md`` next to it.

Run it the way CI does::

    uv run --no-project .github/scripts/stack_review.py <repo-dir> <pr-number>
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argument vectors, no shell
import sys
from typing import Any

from github import Auth, Github
from github.PullRequest import PullRequest
from github.Repository import Repository


PR_AGENT = "git+https://github.com/The-PR-Agent/pr-agent.git@v0.50.0"
BASE_BRANCH = "stack-base"
MARKER = "<!-- pr-agent-stack-review:{top} -->"

INSTRUCTIONS = Path(__file__).parent.parent / "pr-review-instructions.md"

CONFIG = """\
[config]
git_provider = "local"
model = "openrouter/moonshotai/kimi-k3.1-pro"
fallback_models = []
max_model_tokens = 120000
custom_model_max_tokens = 128000
ai_timeout = 900
reasoning_effort = "low"
repo_context_files = ["AGENTS.md"]
output_run_details = true

[local]
review_path = "{review_path}"

[pr_reviewer]
enable_large_pr_chunking = true
max_number_of_calls = 6
enable_review_labels_security = false
enable_review_labels_effort = false
extra_instructions = {instructions}

[openrouter]
max_tokens = 16000
provider_only = ["nebius"]
allow_fallbacks = false
"""

PR = dict[str, Any]
"""An open PR as ``number``, ``title``, ``head`` and ``base`` branch names, and head ``sha``."""


def run(*cmd: str, cwd: Path) -> None:
    """Run a command, exiting with its output if it fails."""
    # ruff: ignore[subprocess-without-shell-equals-true]
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{done.stdout}\n{done.stderr}")


def stack_paths(prs: list[PR], number: int) -> list[list[PR]]:
    """Return every bottom-to-top path of open PRs that contains ``number``."""
    by_number = {pr["number"]: pr for pr in prs}
    by_head = {pr["head"]: pr for pr in prs}
    children: dict[str, list[PR]] = defaultdict(list)
    for pr in prs:
        children[pr["base"]].append(pr)

    bottom = [by_number[number]]
    while bottom[0]["base"] in by_head:
        parent = by_head[bottom[0]["base"]]
        if parent in bottom:
            sys.exit("Stack has a base cycle.")
        bottom.insert(0, parent)

    def up(path: list[PR]) -> list[list[PR]]:
        kids = children.get(path[-1]["head"], [])
        if not kids:
            return [path]
        return [p for kid in kids for p in up([*path, kid])]

    return up(bottom)


def review(repo_dir: Path, path: list[PR], tmp: Path) -> str:
    """Run PR-Agent on the combined diff of ``path``.

    Returns:
        The review Markdown written by PR-Agent.
    """
    root, leaf = path[0], path[-1]
    run("git", "checkout", "--detach", f"origin/{leaf['head']}", cwd=repo_dir)
    run("git", "branch", "-f", BASE_BRANCH, f"origin/{root['base']}", cwd=repo_dir)

    stack = "; ".join(f"#{pr['number']} {pr['title']}" for pr in path)
    review_path = tmp / "review.md"
    review_path.unlink(missing_ok=True)
    config = tmp / "stack-review.toml"
    prompt = f"{INSTRUCTIONS.read_text()}\nThe stack, bottom to top, is: {stack}."
    config.write_text(
        CONFIG.format(review_path=review_path, instructions=json.dumps(prompt, ensure_ascii=False)),
        encoding="utf-8",
    )
    run(
        "uvx", "--from", PR_AGENT, "pr-agent",
        "--pr_url", BASE_BRANCH, "--extra_config_url", str(config), "review",
        cwd=repo_dir,
    )  # fmt: skip
    if not review_path.exists():
        sys.exit("PR-Agent finished without writing a review.")
    return review_path.read_text()


def stamped_comments(pull: PullRequest, marker: str) -> list[Any]:
    """Return the comments on ``pull`` that hold ``marker``, oldest first."""
    return [c for c in pull.get_issue_comments() if marker in c.body]


def open_prs(repo: Repository) -> list[PR]:
    """Return the open PRs from branches of ``repo``.

    A fork PR can reuse a branch name such as `main`, so forks are left out.
    """
    return [
        {"number": p.number, "title": p.title, "head": p.head.ref, "base": p.base.ref, "sha": p.head.sha}
        for p in repo.get_pulls(state="open")
        if p.head.repo and p.head.repo.full_name == repo.full_name
    ]


def review_path_once(pull: PullRequest, path: list[PR], repo_dir: Path, tmp: Path) -> None:
    """Review ``path`` and write the result as the one stack comment on ``pull``.

    Skips the review when no PR head in the path changed since the comment was written. Two runs
    that finish together still leave one comment: it is looked up again right before writing, and
    any extra copy is deleted.
    """
    marker = MARKER.format(top=pull.number)
    commits = "\n".join(f"{pr['number']}:{pr['sha']}" for pr in path)
    stamp = f"<!-- fingerprint:{hashlib.sha256(commits.encode()).hexdigest()} -->"
    if any(stamp in c.body for c in stamped_comments(pull, marker)):
        print(f"Stack ending at #{pull.number} is unchanged since its last review.")
        return
    members = " \u2190 ".join(f"#{pr['number']} (`{pr['sha'][:7]}`)" for pr in path)
    header = f"## Stack review\nReviewed as one change, bottom to top: {members}\n\n"
    body = f"{marker}\n{stamp}\n{header}{review(repo_dir, path, tmp)}"
    comments = stamped_comments(pull, marker)
    if not comments:
        pull.create_issue_comment(body)
        return
    first, *copies = comments
    first.edit(body)
    for copy in copies:
        copy.delete()


def main() -> None:
    """Review every stack path through the PR given on the command line."""
    repo_dir, number = Path(sys.argv[1]).resolve(), int(sys.argv[2])
    tmp = Path(os.environ["RUNNER_TEMP"])

    with Github(auth=Auth.Token(os.environ["GH_TOKEN"])) as github:
        repo = github.get_repo(os.environ["GITHUB_REPOSITORY"])
        prs = open_prs(repo)
        if number not in {pr["number"] for pr in prs}:
            sys.exit(f"#{number} is not an open pull request from a branch of this repository.")
        paths = [path for path in stack_paths(prs, number) if len(path) > 1]
        if not paths:
            print(f"#{number} is not part of a stack; the per-PR review covers it.")
        for path in paths:
            review_path_once(repo.get_pull(path[-1]["number"]), path, repo_dir, tmp)


if __name__ == "__main__":
    main()
