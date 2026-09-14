# AGENTS.md

TimeNet is infrastructure for a shared time-series format called TimeF. It has two sides: a consumer
SDK (`timenet`) to find, download, and read datasets, and a writer that compiles a builder's
declarative dataset into TimeF. Datasets are addressed by an `org/name` id.

## Scope
Instructions for contributors and coding agents working in this repository.

## Documentation Contract
- `README.md` is human-facing: installation, setup, and a tour of the make targets.
- `AGENTS.md` is contributor- and agent-facing: workflow rules, verification requirements, and repo conventions.
- Keep agent operating instructions here. Don't move them into the README.

## Documentation
`docs/` is the documentation site (built with Zensical). Start with `docs/index.md` and
`docs/architecture.md` for the big picture, then read the per-component pages for detail. The docs track
the code under `packages/`; trust the code where the two ever disagree.

Build it with `make docs`, or `make docs-serve` for a live-reloading preview. `make docs-preview`
serves exactly what GitHub Pages publishes. `docs/api/` is generated and git-ignored, along with the
`api-nav` region of `zensical.toml`, so change `scripts/gen_api_docs.py` rather than those. Every
other page under `docs/` is hand-written.

### Code blocks
- Hard-wrap prose near 100 columns. Wrap code inside a fence at 80, counting the fence's own
  indentation, since a fence nested in a list item or a `===` tab starts indented and its column
  is narrower.
- Tag every code fence with its language. `docs/stylesheets/extra.css` soft-wraps tagged fences,
  so a long line reflows to the reader's viewport instead of hiding behind a horizontal scrollbar.
- Leave a fence untagged only when column alignment carries meaning: ASCII diagrams, directory
  trees, `describe()` output. Those render as `.language-text` and deliberately keep scrolling,
  so keep them inside 80 columns yourself; they cannot soft-wrap without being mangled.
- Treat the CSS as a safety net for narrow screens. Wrap the source anyway, breaking lines where
  they read best rather than leaving the browser to choose.
- When a comment makes a line too long, put it on its own line above the code rather than
  squeezing the code. Re-split a long string with implicit concatenation so the value is unchanged.

Task-specific workflows live as agent skills under `.agents/skills/` (finding and reading datasets).
They load on demand, so they stay out of this file.

## Workspace Layout
This is a `uv` workspace. Code lives in one package under `packages/`:
- `packages/timenet` — the `timenet` SDK: the TimeF format, the DuckDB control plane
  with its reader and writer, the Parquet and Zarr values planes, and the registry client.

The package keeps source under `src/` and tests under `tests/`
(`packages/timenet/src`, `packages/timenet/tests`). The root `pyproject.toml` owns the
workspace definition and the shared ruff/ty/pytest config; the package's own
`pyproject.toml` owns its name, version, and dependencies.

`benchmarks/` and `spikes/` sit outside `packages/` and are not workspace members.

## Python And uv
- Use `uv` for all Python workflows. The build backend is `uv_build`.
- Configure the environment with `make sync`. It runs `uv sync --all-groups --all-extras`, so
  every dependency group and every `timenet` extra (`s3`, `torch`, `zarr`) is installed.
- Build distributables with `make build` (`uv build --package timenet`).
- For one-off scripts, use inline `uv` metadata and run with `uv run <script.py>`. Never `pip install`.
- Keep `uv.lock` committed; the `uv-lock` pre-commit hook enforces freshness.

## Verification
After any change, run these and make them pass before claiming the work is done:
- `make check` — `ruff format`, `ruff check`, `ty check`
- `make lint-fix` — auto-fix lint findings
- `make test` — the whole suite (`packages/` plus the end-to-end benchmark tests)

`make install-hooks` once after cloning to wire up pre-commit. To mirror the CI quick job,
run `make check-ci` and `make test-unit`. `make check-ci` runs the hooks over all files, the
same way CI does. It adds a `ty` pass against Python 3.11, because `[tool.ty.environment]`
pins 3.13. Without that pass, a 3.11-only typing error stays hidden until the 3.11 test job
finishes.

In final summaries, state which checks you ran and call out any you could not run.

## Stacked PRs
This repository uses GitHub Stacked PRs via the `gh stack` CLI extension. Each branch in a stack
maps to one PR whose base is the branch below it, so a large change ships as a chain of small,
independently reviewable diffs. A stack is an ordered `main ← branch-1 ← branch-2 ← ...` chain;
foundations go in lower branches and dependents above them, and the tooling (`gh stack init`,
`submit --auto`, `sync`, `rebase --upstack`) keeps the chain rebased and its PRs linked. Agents must
run every `gh stack` command non-interactively (always pass branch names to `init`/`add`, `--auto` to
`submit`, `--json` to `view`); make mid-stack changes on the branch that logically owns them and run
`gh stack rebase --upstack` to propagate, rather than mixing concerns into a higher branch.

## Conventions
- Dash-separated names for user-facing and distribution names (`timenet-datasets`, the
  `duckdb-control-plane` spike); underscores for Python import packages and modules
  (`timenet`, `timenet.control_plane`, `timenet.parquet`). Keep the layers distinct.
- Branch names follow [Conventional Branch](https://conventionalbranch.org/):
  `<type>/<description>` in lowercase with hyphens, e.g. `feature/dataset-register`,
  `bugfix/empty-timef-input`. Common types: `feature/`, `bugfix/`, `hotfix/`,
  `release/`, `chore/`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `<type>(<scope>): <summary>`, e.g. `feat(reader): add a windowed values read`,
  `fix(writer): handle empty TimeF input`. Drop the scope when none applies
  (`chore: refresh lockfile`). Mark breaking changes with `!` or a
  `BREAKING CHANGE:` footer.
- Never use `git commit --no-verify`. If a hook fails, fix the underlying issue
  (run `make check` / `make lint-fix`) and commit again.
- Raise TimeNet's own exceptions from `timenet.errors`, not raw `ValueError` / `Exception`:
  `TimeFValidationError` for a violated TimeF invariant or bad caller input, `TimeFFormatError`
  for a corrupt or unsupported on-disk artifact. `TimeFValidationError` subclasses `ValueError`,
  so existing `except ValueError` handlers keep working.

## Docstrings
- Write Google-style docstrings; ruff enforces them via `D` (pydocstyle) and `DOC`
  (pydoclint), so every public module, class, and function needs one.
- Document arguments and return values when they aren't obvious; `D417` is relaxed,
  so you don't have to document every parameter, but `DOC` checks that any
  documented args/returns match the signature.
- `__init__` and magic methods are exempt (`D107`, `D105`). Tests skip docstring
  rules entirely.

## Implementation Guidelines
- Prefer small, reviewable changes.
- Don't delete user-owned files unless explicitly asked.
- Match the existing style instead of reformatting adjacent code.
- Add type hints; the package ships `py.typed`, so `ty` must stay green.
