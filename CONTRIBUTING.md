# Contributing to TimeNet

Thanks for your interest in TimeNet. This guide covers how to set up the repo, the checks your change
needs to pass, and the naming conventions we use.

## Development setup

TimeNet is a [`uv`](https://docs.astral.sh/uv/) workspace with two packages under `packages/`. You
need Python 3.11 or newer (we test 3.11 through 3.13).

```bash
git clone https://github.com/OpenTSLM/TimeNet.git
cd TimeNet
make sync            # install both packages and the dev + docs groups
make install-hooks   # wire up pre-commit (run once)
```

`make sync` does not install connector dependencies. Each connector declares its own in a
`requirements.txt`, and `make test-connectors` builds a per-connector environment from it.

## Verify your change

Run these before you open a pull request, and make them pass:

- `make check` runs `ruff format`, `ruff check`, and `ty check`.
- `make lint-fix` auto-fixes what ruff can.
- `make test` runs the core test suite.
- `make test-unit` runs the in-memory part of `make test`, for a fast answer.
- `make test-connectors` runs each connector's tests and type-check in its own environment.
- `make license-check` fails the build if a copyleft dependency enters the tree.

To mirror the CI quick job, run `make check-ci` and `make test-unit`. `make check-ci` runs the
hooks over all files, the same way CI does, and adds a `ty` pass against Python 3.11.

Never commit with `--no-verify`. If a hook fails, fix the underlying issue and commit again.

## Commit and branch naming

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <summary>`, for example `feat(cli): add dataset register command`. Drop the scope
when none applies. Mark a breaking change with `!` or a `BREAKING CHANGE:` footer.

Branches follow [Conventional Branch](https://conventionalbranch.org/): `<type>/<description>` in
lowercase with hyphens, for example `feature/dataset-register` or `bugfix/empty-timef-input`.

## Licensing of contributions

TimeNet is released under the [MIT License](LICENSE). When you contribute, you agree that your
contribution is licensed under the same MIT License, so inbound equals outbound. You keep the
copyright to your work.

If you want that agreement recorded in the git history, sign your commits off with `git commit -s`.
This adds a `Signed-off-by` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/). It is optional.

## Code conventions

A few things the linters and reviewers expect:

- Add type hints. Both packages ship `py.typed`, so `ty` must stay green.
- Write Google-style docstrings on public modules, classes, and functions. Tests are exempt.
- Raise TimeNet's own exceptions from `timenet.errors` (`TimeFValidationError`, `TimeFFormatError`)
  rather than raw `ValueError` or `Exception`.
- Keep changes small and match the surrounding style.

See [AGENTS.md](AGENTS.md) for the full contributor and coding-agent conventions.
