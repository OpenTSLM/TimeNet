# TimeNet

TimeNet is infrastructure for registering, querying, downloading, converting, and exploring time series datasets in a shared TimeF format.

## Terminology

| Term | What it is |
| --- | --- |
| **Dataset Card** | The human-authored `card.yaml` beside a connector (license, domains, tags, description). |
| **Dataset Connector** | The Python recipe that fetches a raw source and converts it into a `TimeFDataset`. |
| **Dataset Curation** | The process of running a connector through the engine to produce a dataset. |
| **TimeF Version** | The semantic version of one serialized `TimeFDataset` in a registry. |
| **Dataset Manifest** | The compiled `manifest.json` (card metadata + derived schema + counts + file pointers); the single source of truth the SDK reads. |

See [docs/architecture.md](docs/architecture.md) for how these fit together.

## Installation

### From source

Clone the repo and install locally:

```bash
git clone https://github.com/AI-X-Labs/TimeNet.git
cd TimeNet
uv sync --all-groups   # runtime + dev + docs dependencies
make install-hooks     # set up pre-commit hooks
```

Once installed, the CLI is available:

```bash
uv run timenet
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, [ruff](https://docs.astral.sh/ruff/) for linting and formatting, [ty](https://github.com/astral-sh/ty) for type checking, and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) for docs.

### Setup

Install uv if you don't have it, then sync the environment:

```bash
# install uv (skips if already present)
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

make sync           # uv sync --all-groups (runtime + dev + docs)
make install-hooks  # install pre-commit hooks (run once after cloning)
```

Dependencies are split into groups in `pyproject.toml`:

- runtime — `[project] dependencies` (none yet)
- `dev` — ruff, ty, pytest, pre-commit (installed by default)
- `docs` — mkdocs-material

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
