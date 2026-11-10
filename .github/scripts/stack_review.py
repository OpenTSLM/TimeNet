# /// script
# requires-python = ">=3.11"
# dependencies = []
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


PR_AGENT = "git+https://github.com/The-PR-Agent/pr-agent.git@v0.50.0"
MODEL = "openrouter/moonshotai/kimi-k3.1-pro"
BASE_BRANCH = "stack-base"
MARKER = "<!-- pr-agent-stack-review:{leaf} -->"

INSTRUCTIONS = Path(__file__).parent.parent / "pr-review-instructions.md"

CONFIG = """\
[config]
git_provider = "local"
model = "{model}"
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


def run(*cmd: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run a command and return its stdout, exiting if it fails.

    Returns:
        The standard output of the command.
    """
    # ruff: ignore[subprocess-without-shell-equals-true]
    done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"{' '.join(cmd[:3])} failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


def stack_paths(prs: list[dict[str, Any]], number: int) -> list[list[dict[str, Any]]]:
    """Return every bottom-to-top path of open PRs that contains ``number``."""
    by_number = {pr["number"]: pr for pr in prs}
    by_head = {pr["headRefName"]: pr for pr in prs}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pr in prs:
        children[pr["baseRefName"]].append(pr)

    bottom = [by_number[number]]
    while bottom[0]["baseRefName"] in by_head:
        parent = by_head[bottom[0]["baseRefName"]]
        if parent in bottom:
            sys.exit("Stack has a base cycle.")
        bottom.insert(0, parent)

    def up(path: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        kids = children.get(path[-1]["headRefName"], [])
        if not kids:
            return [path]
        return [p for kid in kids for p in up([*path, kid])]

    return up(bottom)


def review(repo_dir: Path, path: list[dict[str, Any]], tmp: Path) -> str:
    """Run PR-Agent on the combined diff of ``path``.

    Returns:
        The review Markdown written by PR-Agent.
    """
    root, leaf = path[0], path[-1]
    run("git", "checkout", "--detach", f"origin/{leaf['headRefName']}", cwd=repo_dir)
    run("git", "branch", "-f", BASE_BRANCH, f"origin/{root['baseRefName']}", cwd=repo_dir)

    stack = "; ".join(f"#{pr['number']} {pr['title']}" for pr in path)
    review_path = tmp / "review.md"
    review_path.unlink(missing_ok=True)
    config = tmp / "stack-review.toml"
    config.write_text(
        CONFIG.format(
            model=MODEL,
            review_path=review_path,
            instructions=json.dumps(f"{INSTRUCTIONS.read_text()}\nThe stack, bottom to top, is: {stack}."),
        )
    )
    run(
        "uvx",
        "--from",
        PR_AGENT,
        "pr-agent",
        "--pr_url",
        BASE_BRANCH,
        "--extra_config_url",
        str(config),
        "review",
        cwd=repo_dir,
    )
    return review_path.read_text()


def find_comments(repo: str, number: int, marker: str) -> list[dict[str, Any]]:
    """Return the PR comments holding ``marker``, oldest first."""
    lines = run(
        "gh",
        "api",
        "--paginate",
        f"repos/{repo}/issues/{number}/comments",
        "--jq",
        f'.[] | select(.body | contains("{marker}")) | {{id, body}}',
    ).splitlines()
    return [json.loads(line) for line in lines]


def write_comment(repo: str, number: int, marker: str, body: str, tmp: Path) -> None:
    """Edit the oldest PR comment holding ``marker`` in place and delete any copies.

    Looks again right before writing, so two runs that reviewed the same stack at the same time
    end up with one comment.
    """
    body_file = tmp / "comment.md"
    body_file.write_text(body)
    comments = find_comments(repo, number, marker)
    if not comments:
        run("gh", "api", f"repos/{repo}/issues/{number}/comments", "-F", f"body=@{body_file}")
        return
    first, *copies = comments
    run("gh", "api", "-X", "PATCH", f"repos/{repo}/issues/comments/{first['id']}", "-F", f"body=@{body_file}")
    for copy in copies:
        run("gh", "api", "-X", "DELETE", f"repos/{repo}/issues/comments/{copy['id']}")


def main() -> None:
    """Review every stack path through the PR given on the command line.

    Each path gets one comment, on its top PR. A path whose PR head commits have not changed since
    the last review is skipped.
    """
    repo_dir, number = Path(sys.argv[1]).resolve(), int(sys.argv[2])
    repo = os.environ["GITHUB_REPOSITORY"]
    tmp = Path(os.environ["RUNNER_TEMP"])

    prs = json.loads(
        run(
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "number,title,headRefName,headRefOid,baseRefName",
        )
    )
    paths = [path for path in stack_paths(prs, number) if len(path) > 1]
    if not paths:
        print(f"#{number} is not part of a stack; the per-PR review covers it.")
        return

    for path in paths:
        top = path[-1]["number"]
        marker = MARKER.format(leaf=top)
        commits = "\n".join(f"{pr['number']}:{pr['headRefOid']}" for pr in path)
        stamp = f"<!-- fingerprint:{hashlib.sha256(commits.encode()).hexdigest()} -->"
        existing = find_comments(repo, top, marker)
        if any(stamp in comment["body"] for comment in existing):
            print(f"Stack ending at #{top} is unchanged since its last review.")
            continue
        members = " \u2190 ".join(f"#{pr['number']} (`{pr['headRefOid'][:7]}`)" for pr in path)
        header = f"## Stack review\nReviewed as one change, bottom to top: {members}\n\n"
        text = review(repo_dir, path, tmp)
        write_comment(repo, top, marker, f"{marker}\n{stamp}\n{header}{text}", tmp)


if __name__ == "__main__":
    main()
