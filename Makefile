.PHONY: sync test check install-hooks lint-fix build docs docs-serve docs-preview docs-datasets docs-api clean

sync:
	uv sync --all-groups --all-extras

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

docs: docs-datasets docs-api
	uv run --group docs zensical build
	uv run --group docs --extra curation python scripts/gen_site_extras.py

docs-serve: docs-datasets docs-api
	uv run --group docs zensical serve

# Full production preview: build + enhancer (.md mirrors, llms.txt, social meta, copy button),
# served as static files. Unlike `docs-serve`, this reflects exactly what GitHub Pages publishes.
docs-preview: docs
	uv run python -m http.server -d site 8000

docs-datasets:
	uv run --group docs --extra curation python scripts/gen_dataset_docs.py

docs-api:
	uv run --group docs python scripts/gen_api_docs.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty site __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
