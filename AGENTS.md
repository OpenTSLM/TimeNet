# AGENTS.md

TimeNet is infrastructure for registering, querying, downloading, converting, and exploring time series datasets in a shared TimeF format.

## Scope
Instructions for contributors and coding agents working in this repository.

## Documentation Contract
- `README.md` is human-facing: installation, setup, and a tour of the make targets.
- `AGENTS.md` is contributor- and agent-facing: workflow rules, verification requirements, and repo conventions.
- Keep agent operating instructions here. Don't move them into the README.

## Workspace Layout
This is a `uv` workspace. Code lives in two packages under `packages/`:
- `packages/timenet` — the `timenet` SDK and CLI: the TimeF format plus dataset,
  reader, and writer definitions, and the `timenet` console script.
- `packages/timenet-connectors` — the `timenet_connectors` package: dataset-specific
  logic to fetch raw sources and convert them into TimeF. Depends on `timenet`, which
  it resolves locally via `[tool.uv.sources]`.

Each package keeps source under `src/` and tests under `tests/`
(`packages/<name>/src`, `packages/<name>/tests`). The root `pyproject.toml` owns the
workspace definition and the shared ruff/ty/pytest config; per-package
`pyproject.toml` files own their name, version, and dependencies.

## Python And uv
- Use `uv` for all Python workflows. The build backend is `uv_build`.
- Set up the environment with `make sync` (`uv sync --all-groups`), which installs
  every workspace member.
- Build distributables with `make build` (`uv build --package <name>` per member).
- For one-off scripts, use inline `uv` metadata and run with `uv run <script.py>`. Never `pip install`.
- Keep `uv.lock` committed; the `uv-lock` pre-commit hook enforces freshness.

## Verification
After any change, run these and make them pass before claiming the work is done:
- `make check` — `ruff format`, `ruff check`, `ty check`
- `make lint-fix` — auto-fix lint findings
- `make test` — `uv run pytest`

`make install-hooks` once after cloning to wire up pre-commit. To mirror CI exactly,
run `uv run pre-commit run --all-files`.

In final summaries, state which checks you ran and call out any you could not run.

## Conventions
- Dash-separated names for user-facing/CLI and distribution names (`timenet-connectors`);
  underscores for Python import packages and modules (`timenet`, `timenet.cli`,
  `timenet_connectors`). Keep the layers distinct.
- Branch names follow [Conventional Branch](https://conventionalbranch.org/):
  `<type>/<description>` in lowercase with hyphens, e.g. `feature/dataset-register`,
  `bugfix/empty-timef-input`. Common types: `feature/`, `bugfix/`, `hotfix/`,
  `release/`, `chore/`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `<type>(<scope>): <summary>`, e.g. `feat(cli): add dataset register command`,
  `fix(cli): handle empty TimeF input`. Drop the scope when none applies
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
- Add type hints; both packages ship `py.typed`, so `ty` must stay green.
