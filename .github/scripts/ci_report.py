# /// script
# requires-python = ">=3.11"
# dependencies = ["PyGithub>=2.9"]
# ///
"""Upsert the single sticky CI comment on a pull request.

Called by the last step of ``.github/workflows/pre-commit-hooks.yaml`` once the checks have run. It
reads their outcomes from the environment, renders one Markdown report, appends that to the job
summary, and then creates or updates exactly one PR comment.

The comment is found by :data:`MARKER` rather than by "the last comment from this token", so a run
never edits somebody else's comment and never stacks up a new one per push. A first-time passing run
writes no comment at all; only a failure opens one, and later runs keep editing it in place.

Run it the way CI does::

    uv run .github/scripts/ci_report.py
"""

from __future__ import annotations

import os
from pathlib import Path

from github import Auth, Github


MARKER = "<!-- timenet-ci-report -->"
"""Hidden marker that identifies our comment among all the comments on a PR."""

_LABELS = {"success": "✅ Passed", "failure": "❌ Failed"}

_REPRODUCE = """```bash
make lint-fix   # auto-fix
make check      # ruff + ty
make test       # pytest
```"""


def _label(result: str) -> str:
    """Render one job result as a table cell.

    Args:
        result: A GitHub job result (``success``, ``failure``, ``cancelled``, ``skipped``).

    Returns:
        The human-readable status, defaulting to "did not run" for anything that isn't a
        pass or a fail.
    """
    return _LABELS.get(result, "⏭️ Did not run")


def render(results: dict[str, str], head_sha: str, run_url: str, run_number: str) -> str:
    """Render the report body.

    Args:
        results: Ordered mapping of display name to GitHub job result.
        head_sha: Head commit of the pull request.
        run_url: Link to the workflow run.
        run_number: The run's display number.

    Returns:
        The Markdown body, marker included.
    """
    failed = any(result != "success" for result in results.values())
    rows = "\n".join(f"| {name} | {_label(result)} |" for name, result in results.items())
    detail = f"\nFull output is in the [job logs]({run_url}). Reproduce locally:\n\n{_REPRODUCE}\n" if failed else ""
    return (
        f"{MARKER}\n"
        f"## {'⚠️ CI checks failed' if failed else '✅ CI checks passed'}\n\n"
        f"| Check | Status |\n| --- | --- |\n{rows}\n"
        f"{detail}\n"
        f"<sub>Updated for `{head_sha[:7]}` · [run #{run_number}]({run_url})</sub>\n"
    )


def main() -> None:
    """Render the report, append it to the job summary, and upsert the PR comment."""
    env = os.environ
    run_url = f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"

    results = {
        "Pre-commit hooks": env["PRECOMMIT"],
        "Tests (Python 3.13)": env["PYTEST"],
        "Tests (Python 3.11)": env["PYTEST_MIN"],
    }
    body = render(results, env["HEAD_SHA"], run_url, env["GITHUB_RUN_NUMBER"])

    if summary := env.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(body)

    with Github(auth=Auth.Token(env["GITHUB_TOKEN"])) as github:
        # Go through the pull request, not get_issue(): fetching the issue object needs an `issues`
        # permission, while the workflow only grants `pull-requests: write`.
        pull = github.get_repo(env["GITHUB_REPOSITORY"]).get_pull(int(env["PR"]))
        existing = next((c for c in pull.get_issue_comments() if c.body.startswith(MARKER)), None)
        if existing is not None:
            existing.edit(body)
        elif any(result != "success" for result in results.values()):
            pull.create_issue_comment(body)


if __name__ == "__main__":
    main()
