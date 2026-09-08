# AGENTS.md

TimeNet is infrastructure for a shared time-series format called TimeF. It has two sides: a consumer
SDK and CLI (`timenet`) to find, download, and load datasets, and a connectors package
(`timenet-connectors`) that converts external sources into TimeF. Datasets are addressed by an
`org/name` id.

## Scope
Instructions for contributors and coding agents working in this repository.

## Documentation Contract
- `README.md` is human-facing: installation, setup, and a tour of the make targets.
- `AGENTS.md` is contributor- and agent-facing: workflow rules, verification requirements, and repo conventions.
- A connector's `README.md` is dataset-facing: what the release states, the assumptions the
  conversion rests on, and every inconsistency it ships. It travels with the connector because it
  answers questions about that dataset, not about TimeNet.
- Keep agent operating instructions here. Don't move them into the README.

## Documentation
`docs/` is the documentation site (built with Zensical). Start with `docs/index.md` and
`docs/architecture.md` for the big picture, then read the per-component pages for detail. The docs track
the code under `packages/`; trust the code where the two ever disagree.

Build it with `make docs`, or `make docs-serve` for a live-reloading preview. `make docs-preview`
serves exactly what GitHub Pages publishes. `docs/api/` and `docs/catalog/datasets.{md,json}` are
generated and git-ignored, so change the generator under `scripts/` rather than those files.
Everything else under `docs/`, including `docs/catalog/benchmarks.md`, is hand-written.

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

Task-specific workflows live as agent skills under `.agents/skills/` (finding and loading datasets,
adding a dataset connector). They load on demand, so they stay out of this file.

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
- Configure the environment with `make sync`. It installs every workspace member and the
  workspace extras (`timenet[cli,torch]`). It does not install connector dependencies. Each
  connector declares its own in a `requirements.txt`, and `make test-connectors` runs that
  connector's tests and type-check in an environment built from it.
- Build distributables with `make build` (`uv build --package <name>` per member).
- For one-off scripts, use inline `uv` metadata and run with `uv run <script.py>`. Never `pip install`.
- Keep `uv.lock` committed; the `uv-lock` pre-commit hook enforces freshness.

## Verification
After any change, run these and make them pass before claiming the work is done:
- `make check` — `ruff format`, `ruff check`, `ty check`
- `make lint-fix` — auto-fix lint findings
- `make test` — core tests in the dev environment
- `make test-connectors` — each connector's tests and type-check in its own environment

`make install-hooks` once after cloning to wire up pre-commit. To mirror CI exactly,
run `uv run pre-commit run --all-files`.

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

A PR description states the **Problem** first and the **Changelog** second, in simple English, with
the ticket reference last. The problem is what the reviewer needs to judge the change. The list of
files is not. Each PR of a stack must stand on its own. It carries no comment, docstring, or line
that a later PR in the same stack deletes. It also carries no forward-looking chatter about work
that has not landed.

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
- Raise TimeNet's own exceptions from `timenet.errors`, not raw `ValueError` / `Exception`. They all
  descend from `TimeNetError`. Pick the one that names what went wrong:

  | error | raise it when |
  | --- | --- |
  | `TimeFValidationError` | a TimeF invariant is violated, or caller input is bad |
  | `TimeFEditError` | an edit to a dataset is not allowed |
  | `TimeFFormatError` | an on-disk artifact is corrupt or unsupported |
  | `TimeNetInvalidManifestError` | a manifest is corrupt (a `TimeFFormatError`) |
  | `TimeNetInvalidCardError` | a `dataset.yaml` fails to validate |
  | `TimeNetDatasetNotFoundError` | a dataset id resolves to nothing |
  | `TimeNetRegistryError` | a registry cannot be read or written |
  | `TimeNetAccessError` | the caller may not reach the data |
  | `TimeNetDownloadError` | a fetch fails |
  | `TimeNetBuildError` | a build fails for a reason none of the above names |

  Four of them subclass `ValueError`, so existing `except ValueError` handlers keep working:
  `TimeFValidationError`, `TimeFEditError` (through it), `TimeNetInvalidCardError`, and
  `TimeNetInvalidManifestError`. Warnings descend from `TimeNetWarning`.
  `SpanOutsideWindowWarning` is the one a connector meets.

## Docstrings
- Write Google-style docstrings; ruff enforces them via `D` (pydocstyle) and `DOC`
  (pydoclint), so every public module, class, and function needs one.
- Document arguments and return values when they aren't obvious; `D417` is relaxed,
  so you don't have to document every parameter, but `DOC` checks that any
  documented args/returns match the signature.
- `__init__` and magic methods are exempt (`D107`, `D105`). Tests skip docstring
  rules entirely.

## Python Habits

Rules the linters do not state and the code follows anyway.

- **`# noqa: CODE (reason)`** — parenthesized, lowercase, no trailing period. A suppression without
  a reason is a suppression nobody can retire.
  ```python
  def download(self, cache_dir: Path) -> list[None]:  # noqa: ARG002 (synthetic: nothing to fetch)
  ```
- **A suppression of a complexity rule takes its reason in prose above the `def`**, not inside the
  `noqa`. The reason is a paragraph and the `noqa` is a line.
- **`DOC502` is the sanctioned escape** when a helper raises a documented `Raises:`, and not the
  function itself. You can also reword the docstring to name the helper. Then the suppression is
  not necessary.
- **Neither package uses `typing.Final`.** Do not introduce it.
- **Document dataclass fields in `timenet`, not in a connector.** The core types are public API and
  carry a docstring under every field. A connector's handle and row dataclasses carry a class
  docstring instead, and annotate fields with a trailing `#` comment. They are plumbing between
  `download` and `convert`.
- **A constant used once lives at its use site.** A lookup table stays at module level whatever its
  use count, so it is not rebuilt per call. A source URL stays at module level too, even when it is
  used once, so a mirror can replace it. A top-level constant that needs an explanation takes a
  docstring below it, not a `#` comment. URLs are the exception: they take comments.
- **Python is `line-length = 120`, and `E501` is off** because the formatter owns it. The formatter
  will not split a long string or comment. Wrap those by hand. (Prose and fenced code in `docs/`
  follow the narrower rule stated earlier.)

### Errors

After you pick the right type from the table, two habits hold:

- **Some non-TimeNet errors are deliberate, not oversights.** `ImportError` carries a message that
  names the fix, for an optional extra that is not installed. `LookupError` covers a connector id
  that resolves to nothing. `FileNotFoundError` covers a path that does not resolve under a root a
  helper searched. `typer.BadParameter` covers bad CLI input, and the upstream library's own
  `DatasetNotFoundError` is deliberate too. Do not "fix" these into TimeNet types.
- **The distinction the code draws is *who is wrong*.** Bad configuration is the caller's fault:
  `TimeFValidationError`. A file that is present but unreadable is the artifact's fault:
  `TimeFFormatError`. A fetch that arrived without the parts it promised is the download's fault:
  `TimeNetDownloadError`.
- **The message shape is `<subject> <verb> <expectation>, got <value!r>`.** State the value that
  failed, `!r`-quoted. Then name the thing a reader must open:
  `"{workbook} row {row_number}: {field} holds {cell!r}"`. When one message covers two branches,
  bind it to a local and raise it twice. Do not repeat the literal.

### Type checking

`ty` is stricter than the tests' ergonomics, and the fix is always explicit narrowing, never a cast.

- `Annotation.span`, `Task.scope`, `TimeSeries.time_axis` and `TimeSeries.span_us` are unions.
  A read of `.start_us`, `.period_us` or `[1]` off one of them fails.
- In tests, write small `assert isinstance(...)`-and-return helpers and call those. Tests already
  ignore `S101`, so the assert is free.
- `Task` has no `target_schema`. Only `ClassificationTask` does. Narrow by `isinstance` before you
  touch a subclass field.
- **`pyarrow.compute` does not type-check.** `ty` rejects every `pc.*` call — "Module
  `pyarrow.compute` has no member `list_value_length`" — because those functions are generated at
  import time. Use the real methods instead: `ListArray.value_lengths()` and `.flatten()`. Use
  `numpy` for the rest, such as `np.isfinite` in place of `pc.is_finite`.

### Things that surprise you once

- **`ruff format` reflows and `ruff check` then complains.** After a large edit, run `make lint-fix`
  and `make check` until both are quiet. One pass is not enough.
- **`RUF069` bans `==` between floats**, tests included. Use `pytest.approx`, or compare the integer
  microseconds the format stores.
- **`TimeSeriesSpec` is a frozen dataclass**, so two identically-built specs are equal and dedupe.
- **`Record.start_time` refuses a bare `float` and a naive `datetime`** — seconds and microseconds
  are both plausible readings of a float. Pass a tz-aware `datetime`, or whole Unix microseconds as
  an `int`. The field is `datetime | int | None`.
- **`timenet/__init__.py` exports nothing.** Import from the submodule: `from timenet.client import
  TimeNet`.
- **`uv` is version-pinned** by `required-version` in the root `pyproject.toml`. A too-old `uv`
  refuses everything, `uv sync` included. Run `uv self update` first.
- **No connector reads an environment variable today.** If you add one, name it
  `TIMENET_<DATASET>_<THING>`. Raise `TimeFValidationError` on a bad value. `TIMENET_TESTING` and
  `TIMENET_ROW_LIMIT` are named in some older docs. They were never implemented. Do not write a test
  that depends on them.

## Implementation Guidelines
- Prefer small, reviewable changes.
- Don't delete user-owned files unless explicitly asked.
- Match the existing style instead of reformatting adjacent code.
- Add type hints; both packages ship `py.typed`, so `ty` must stay green.
