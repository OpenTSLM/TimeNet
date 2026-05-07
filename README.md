# TimeNet

TimeNet is infrastructure for registering, querying, downloading, converting, and exploring time series datasets in a shared TimeF format.

## Installation

### From source

Clone the repo and install locally:

```bash
git clone https://github.com/yourorg/timenet.git
cd timenet
uv sync --extra all         # or: pip install -e ".[all]"
make install-hooks
```


## Development

```bash
make sync           # install all deps (uv sync --extra all --extra dev)
make test           # run pytest
make check          # format + lint + typecheck
make lint-fix       # fix trivial linting issues by ruff
make install-hooks  # install pre-commit hooks (run once after cloning)
make clean          # remove .venv and caches
```

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to run ruff (lint + format) before every commit. After cloning, run:

```bash
make install-hooks
```
