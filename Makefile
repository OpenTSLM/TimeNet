.PHONY: sync test check install-hooks lint-fix build docs docs-serve clean

sync:
	uv sync --all-groups

test:
	uv run pytest

build:
	uv build --package timenet
	uv build --package timenet-connectors

check:
	uv run ruff format .
	uv run ruff check .
	uv run ty check .

lint-fix:
	uv run ruff check . --fix

install-hooks:
	uv run pre-commit install

docs:
	uv run --group docs mkdocs build

docs-serve:
	uv run --group docs mkdocs serve

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty site __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
