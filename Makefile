.PHONY: sync test test-connectors check install-hooks lint-fix build license-check docs docs-serve docs-preview docs-datasets docs-api clean

sync:
	uv sync --all-groups --all-extras

# Core tests in the dev environment. Each connector declares its own dependencies in a
# requirements.txt, so a connector's tests do not run here. They run in per-connector environments
# through `make test-connectors`.
test:
	uv run pytest --ignore-glob='*/timenet_connectors/datasets/*'

# Each connector's tests and type-check run in an environment built from that connector's own
# requirements, the same way a build runs. See scripts/check_connectors.py.
test-connectors:
	uv run scripts/check_connectors.py

# Fail the build if a copyleft dependency (GPL/LGPL/AGPL/SSPL/EUPL/CDDL/OSL) enters the resolved
# environment. Runs against the synced dev env, which pulls in every extra plus the connector deps in
# the dev group (wfdb, huggingface-hub, boto3). A dep declared only in a connector's requirements.txt
# is not seen here; current connectors are all permissive. Denylist over --allow-only on purpose:
# --allow-only trips on PEP 639 combined expressions (e.g. numpy's "BSD-3-Clause AND 0BSD AND ..."),
# while --partial-match --fail-on only fires when a copyleft token actually appears.
license-check:
	uv run --with "pip-licenses>=5.0" pip-licenses --partial-match \
		--fail-on="GPL;LGPL;AGPL;SSPL;EUPL;CDDL;OSL" \
		--format=markdown

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
	uv run --group docs --extra build python scripts/gen_site_extras.py

docs-serve: docs-datasets docs-api
	uv run --group docs zensical serve

# Full production preview: build + enhancer (.md mirrors, llms.txt, social meta, copy button),
# served as static files. Unlike `docs-serve`, this reflects exactly what GitHub Pages publishes.
docs-preview: docs
	uv run python -m http.server -d site 8000

docs-datasets:
	uv run --group docs --extra build python scripts/gen_dataset_docs.py

docs-api:
	uv run --group docs python scripts/gen_api_docs.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty site __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
