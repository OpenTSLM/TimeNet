.PHONY: sync test test-unit check check-ci install-hooks lint-fix build license-check docs docs-serve docs-preview docs-api clean

sync:
	uv sync --all-groups --all-extras

test:
	uv run pytest

# The in-memory part of `make test`, for the CI job that has to answer in under a minute. This is a
# preview of the suite, not a partition of it: every test listed here runs again under `make test`.
# What is left out (reader, registry, writer, the connector tests and the benchmark round-trips) is
# filesystem- and network-bound and accounts for most of the suite's runtime.
UNIT_TESTS = \
	packages/timenet/tests/types \
	packages/timenet/tests/format \
	packages/timenet/tests/manifest \
	packages/timenet/tests/schemas \
	packages/timenet/tests/control_plane \
	packages/timenet/tests/test_cache.py \
	packages/timenet/tests/test_config.py \
	packages/timenet/tests/test_errors.py \
	packages/timenet/tests/test_provenance.py \
	packages/timenet/tests/test_refs.py

test-unit:
	uv run pytest $(UNIT_TESTS)

# Fail the build if a copyleft dependency (GPL/LGPL/AGPL/SSPL/EUPL/CDDL/OSL) enters the resolved
# environment. Runs against the synced dev env, which pulls in every extra plus the source-reading
# deps in the dev group (wfdb, huggingface-hub, edfio, xlrd). Denylist over --allow-only on purpose:
# --allow-only trips on PEP 639 combined expressions (e.g. numpy's "BSD-3-Clause AND 0BSD AND ..."),
# while --partial-match --fail-on only fires when a copyleft token actually appears. A dep whose
# metadata declares no license reports UNKNOWN and is not gated; scan the printed table if one shows up.
license-check:
	uv run --with "pip-licenses>=5.0" pip-licenses --partial-match \
		--fail-on="GPL;LGPL;AGPL;SSPL;EUPL;CDDL;OSL" \
		--format=markdown

build:
	uv build --package timenet

check:
	uv run ruff format .
	uv run ruff check .
	uv run ty check .

# What the quick CI job runs. The hooks cover ruff and ty on the configured target, 3.13; the second
# command re-checks against the minimum supported Python, where a 3.11-only typing error (a symbol
# that only exists in a later typing module, say) would otherwise stay hidden until the 3.11 test
# job finishes minutes later.
#
# Both commands run even if the first one fails, so a lint nit cannot mask a 3.11 failure. The
# recipe exits non-zero if either did.
check-ci:
	@rc=0; uv run pre-commit run --all-files --show-diff-on-failure || rc=$$?; \
	uv run ty check --python-version 3.11 . || rc=$$?; \
	exit $$rc

lint-fix:
	uv run ruff check . --fix

install-hooks:
	uv run pre-commit install

docs: docs-api
	uv run --group docs zensical build
	uv run --group docs python scripts/gen_site_extras.py

docs-serve: docs-api
	uv run --group docs zensical serve

# Full production preview: build + enhancer (.md mirrors, llms.txt, social meta, copy button),
# served as static files. Unlike `docs-serve`, this reflects exactly what GitHub Pages publishes.
docs-preview: docs
	uv run python -m http.server -d site 8000

docs-api:
	uv run --group docs python scripts/gen_api_docs.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty site __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
