.PHONY: sync test check install-hooks lint-fix clean

sync:
	uv sync --extra all --extra dev

test:
	uv run pytest

check:
	uv run ruff format .
	uv run ruff check .
	uv run ty check .

lint-fix:
	uv run ruff check . --fix

install-hooks:
	uv run pre-commit install

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
