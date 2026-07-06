# TimeNet

TimeNet is infrastructure for registering, querying, downloading, converting, and exploring time series datasets in a shared TimeF format.

Documentation: <https://ai-x-labs.github.io/TimeNet/>

## Terminology

| Term | What it is |
| --- | --- |
| **Dataset Card** | The human-authored `dataset.yaml` in a connector's folder (license, domains, tags, description); validated against a JSON Schema when loaded. |
| **Dataset Connector** | The Python recipe that fetches a raw source and converts it into a `TimeFDataset`. |
| **Dataset Curation** | The process of running a connector through the engine to produce a dataset. |
| **TimeF Version** | The semantic version of one serialized `TimeFDataset` in a registry. |
| **Dataset Manifest** | The compiled `manifest.json` (card metadata + derived schema + counts + file pointers); the single source of truth the SDK reads. |

See [docs/architecture.md](docs/architecture.md) for how these fit together.

## Installation

Requires Python 3.11 or newer (tested on 3.11–3.13).

### From source

Clone the repo and install locally:

```bash
git clone https://github.com/AI-X-Labs/TimeNet.git
cd TimeNet
uv sync --all-groups --all-extras   # dev/docs deps + optional extras (cli, torch, huggingface)
make install-hooks                  # set up pre-commit hooks
```

Once installed, the CLI is available:

```bash
uv run timenet
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, [ruff](https://docs.astral.sh/ruff/) for linting and formatting, [ty](https://github.com/astral-sh/ty) for type checking, and [Zensical](https://zensical.org/) for docs.

### Setup

Install uv if you don't have it, then sync the environment:

```bash
# install uv (skips if already present)
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

make sync           # uv sync --all-groups (runtime + dev + docs)
make install-hooks  # install pre-commit hooks (run once after cloning)
```

Dependencies are split in `pyproject.toml`:

- runtime — `timenet` needs numpy, pyarrow, pint, pydantic-settings, typer; `timenet-connectors` adds
  its own. Optional extras: `timenet[torch]`, `timenet-connectors[huggingface]`.
- `dev` — ruff, ty, pytest, pre-commit, hypothesis (installed by default)
- `docs` — zensical

### Make targets

```bash
make sync           # install all deps (uv sync --all-groups)
make test           # run pytest
make check          # format + lint + typecheck (ruff format, ruff check, ty check)
make lint-fix       # auto-fix lint issues with ruff
make docs           # build the docs into site/
make docs-serve     # serve the docs locally at http://127.0.0.1:8000
make install-hooks  # install pre-commit hooks (run once after cloning)
make clean          # remove .venv, caches, and built site/
```

### Pre-commit hooks

Every commit runs ruff (lint + format), ty type checking, `uv lock`, and basic file hygiene (trailing whitespace, end-of-file, TOML/JSON validation). After cloning:

```bash
make install-hooks
```

Don't bypass the hooks with `--no-verify`. If a hook fails, fix the underlying issue (run `make check` / `make lint-fix`) and commit again.

### Docs

```bash
make docs-serve   # live preview at http://127.0.0.1:8000
make docs         # build static site into site/
```

Published at <https://ai-x-labs.github.io/TimeNet/>, deployed from `main` by
`.github/workflows/docs.yml` (enable Pages in the repo settings with source "GitHub Actions").
