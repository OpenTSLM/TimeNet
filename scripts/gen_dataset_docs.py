"""Generate the datasets catalog page (and machine-readable index) from the dataset cards.

The catalog is a single interactive page: every dataset is one expandable row (no per-dataset
page), with category/tag filter dropdowns wired up by ``docs/javascripts/extra.js``. Rows and
filter options are generated here from the ``dataset.yaml`` cards, enriched with schema/counts
when a built ``manifest.json`` is available. Reads cards straight from disk (no network, no
connector imports).

Run it before building the docs::

    uv run --group docs --extra curation python scripts/gen_dataset_docs.py
    uv run --group docs --extra curation python scripts/gen_dataset_docs.py --registry ./local_registry

Output goes to ``docs/catalog/`` as ``datasets.md`` + ``datasets.json`` (git-ignored;
regenerated at build time), alongside the hand-written ``catalog/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from timenet.manifest import Manifest
from timenet.types import DatasetMetadata


REPO_ROOT = Path(__file__).resolve().parent.parent
CARDS_ROOT = REPO_ROOT / "packages" / "timenet-connectors" / "src" / "timenet_connectors" / "datasets"
OUTPUT_DIR = REPO_ROOT / "docs" / "catalog"

# Inline lucide icons (inner SVG only) per dataset category/domain.
_DOMAIN_ICONS = {
    "health": (
        '<path d="M11 2v2"/><path d="M5 2v2"/>'
        '<path d="M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1"/>'
        '<path d="M8 15a6 6 0 0 0 12 0v-3"/><circle cx="20" cy="10" r="2"/>'
    ),
    "cardiology": (
        '<path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 '
        '5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>'
        '<path d="M3.22 13H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>'
    ),
    "sleep": (
        '<path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215'
        '.825-.004.803.401"/>'
    ),
    "activity": (
        '<path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 '
        '8.68V16a2 2 0 1 1-4 0Z"/>'
        '<path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 '
        '2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/>'
    ),
    "economics": (
        '<path d="M10 18v-7"/>'
        '<path d="M11.119 2.205a2 2 0 0 1 1.762 0l7.84 3.846A.5.5 0 0 1 20.5 7h-17a.5.5 0 0 1-.22-.949z"/>'
        '<path d="M14 18v-7"/><path d="M18 18v-7"/><path d="M3 22h18"/><path d="M6 18v-7"/>'
    ),
    "finance": (
        '<rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="12" cy="12" r="2"/>'
        '<path d="M6 12h.01M18 12h.01"/>'
    ),
    "general": (
        '<path d="M8.3 10a.7.7 0 0 1-.626-1.079L11.4 3a.7.7 0 0 1 1.198-.043L16.3 8.9a.7.7 0 0 1-.572 1.1Z"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1"/><circle cx="17.5" cy="17.5" r="3.5"/>'
    ),
}


def _icon(domain: str) -> str:
    """Return an inline lucide SVG for a category, falling back to a generic shape.

    Args:
        domain: The category/domain name.

    Returns:
        An ``<svg>`` string sized via the ``.ds-icon`` class.
    """
    inner = _DOMAIN_ICONS.get(domain, _DOMAIN_ICONS["general"])
    return (
        '<svg class="ds-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{inner}</svg>'
    )


def _find_manifest(registry: Path | None, dataset_id: str) -> Manifest | None:
    """Load the latest built manifest for a dataset from a local registry, if one exists.

    Args:
        registry: A local registry directory, or ``None`` to skip manifest enrichment.
        dataset_id: The ``org/name`` dataset id.

    Returns:
        The parsed :class:`~timenet.manifest.Manifest`, or ``None`` when no built version exists.
    """
    if registry is None:
        return None
    manifests = sorted((registry / dataset_id).glob("*/manifest.json"))
    if not manifests:
        return None
    return Manifest.from_json(manifests[-1].read_text(encoding="utf-8"))


def _schema_html(manifest: Manifest) -> str:
    """Render a manifest's schema and counts as an HTML block for a catalog row detail.

    Args:
        manifest: The built manifest to read schema and counts from.

    Returns:
        The schema section's HTML.
    """
    counts = manifest.counts
    rows = "".join(
        f"<tr><td><code>{html.escape(s.spec_type)}</code></td><td>{html.escape(s.name)}</td>"
        f"<td>{html.escape(str(s.unit_value))}</td></tr>"
        for s in manifest.schema.time_series_specs
    )
    tasks = ", ".join(f"{html.escape(name)} ({n})" for name, n in sorted(counts.tasks.items())) or "n/a"
    return (
        '<div class="ds-schema"><table><thead><tr><th>Modality</th><th>Name</th><th>Unit</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        f"<p><strong>Samples:</strong> {counts.samples} · <strong>Annotations:</strong> {counts.annotations}"
        f" · <strong>Tasks:</strong> {tasks}</p></div>"
    )


def _source_link(url: str | None) -> str:
    """Render a card's source URL as a safe link, or plain text for non-web schemes.

    Only ``http(s)`` URLs become clickable; anything else (e.g. a ``javascript:`` payload from a
    community-contributed card) is shown as escaped text so it can't execute on the public site.

    Args:
        url: The card's ``source_url``, or ``None``.

    Returns:
        An anchor for web URLs, escaped text for other schemes, or ``"n/a"`` when absent.
    """
    if not url:
        return "n/a"
    safe = html.escape(url)
    if url.startswith(("http://", "https://")):
        return f'<a href="{safe}" target="_blank" rel="noopener">{safe}</a>'
    return safe


def _render_catalog_row(metadata: DatasetMetadata, manifest: Manifest | None) -> str:
    """Render one dataset as a table summary row plus a hidden detail row (raw HTML island).

    Args:
        metadata: The dataset's authored card metadata.
        manifest: A built manifest to enrich the detail with, or ``None``.

    Returns:
        Two ``<tr>`` elements: a one-line summary row and a collapsed detail row that the JS
        widget expands on click. The summary row carries the ``data-*`` filter attributes.
    """
    domains = [str(d) for d in metadata.domains]
    tags = list(metadata.tags)
    license_id = str(metadata.license)
    text = " ".join([metadata.dataset_id, metadata.name, metadata.description, *tags, *domains]).lower()
    cats = (
        "".join(f'<span class="ds-cat">{_icon(d)}<span>{html.escape(d)}</span></span>' for d in domains)
        or '<span class="ds-cat">uncategorized</span>'
    )
    tag_badges = "".join(f'<span class="ds-tag">{html.escape(t)}</span>' for t in tags) or "n/a"
    snippet = (
        f'from timenet.client import TimeNet\n\ndataset = TimeNet().load("{metadata.dataset_id}")\ndataset.describe()'
    )
    source = _source_link(metadata.source_url)
    return (
        f'<tr class="ds-row" data-domains="{html.escape(" ".join(domains))}" '
        f'data-tags="{html.escape(" ".join(tags))}" data-text="{html.escape(text)}" '
        'tabindex="0" role="button" aria-expanded="false">'
        '<td class="ds-toggle" aria-hidden="true"></td>'
        f'<td class="ds-id"><code>{html.escape(metadata.dataset_id)}</code></td>'
        f'<td class="ds-name">{html.escape(metadata.name)}</td>'
        f'<td class="ds-cats">{cats}</td>'
        f'<td class="ds-license">{html.escape(license_id)}</td>'
        "</tr>"
        '<tr class="ds-detail-row" hidden><td colspan="5"><div class="ds-detail">'
        f"<p>{html.escape(metadata.description)}</p>"
        '<dl class="ds-meta">'
        f"<dt>Version</dt><dd>{html.escape(str(metadata.dataset_version))}</dd>"
        f"<dt>Tags</dt><dd>{tag_badges}</dd>"
        f"<dt>Source</dt><dd>{source}</dd>"
        "</dl>"
        f'<pre><code class="language-python">{html.escape(snippet)}</code></pre>'
        f"{_schema_html(manifest) if manifest is not None else ''}"
        "</div></td></tr>"
    )


def _render_filters(domains: list[str], tags: list[str]) -> str:
    """Render the search box plus the Categories and Tags multi-select dropdowns.

    Args:
        domains: All distinct category names across the datasets, sorted.
        tags: All distinct tags across the datasets, sorted.

    Returns:
        The filter bar's HTML; the dropdown behavior is wired by ``extra.js``.
    """
    cat_opts = "".join(
        f'<label class="ds-opt"><input type="checkbox" value="{html.escape(d)}">{_icon(d)}'
        f"<span>{html.escape(d)}</span></label>"
        for d in domains
    )
    tag_opts = "".join(
        f'<label class="ds-opt"><input type="checkbox" value="{html.escape(t)}"><span>{html.escape(t)}</span></label>'
        for t in tags
    )
    return (
        '<div class="ds-filters">'
        '<div class="ds-search-wrap">'
        '<label class="ds-visually-hidden" for="ds-search">Search datasets</label>'
        '<input type="search" id="ds-search" class="ds-search" placeholder="Search datasets…" autocomplete="off">'
        "</div>"
        '<div class="ds-dd" data-facet="domain">'
        '<button type="button" class="ds-dd__btn" aria-expanded="false" aria-haspopup="true">'
        'Categories<span class="ds-dd__count"></span></button>'
        f'<div class="ds-dd__menu" role="group" aria-label="Categories" hidden>{cat_opts}</div>'
        "</div>"
        '<div class="ds-dd" data-facet="tag">'
        '<button type="button" class="ds-dd__btn" aria-expanded="false" aria-haspopup="true">'
        'Tags<span class="ds-dd__count"></span></button>'
        f'<div class="ds-dd__menu" role="group" aria-label="Tags" hidden>{tag_opts}</div>'
        "</div>"
        '<button type="button" class="ds-clear">Clear</button>'
        "</div>"
    )


def _render_index(entries: list[DatasetMetadata], manifests: dict[str, Manifest | None]) -> str:
    """Render the datasets catalog page.

    Args:
        entries: The dataset metadata, already sorted.
        manifests: Map of dataset id to its built manifest (or ``None``).

    Returns:
        The catalog page's Markdown (with a raw-HTML island the JS widget enhances).
    """
    domains = sorted({str(d) for m in entries for d in m.domains})
    tags = sorted({t for m in entries for t in m.tags})
    rows = "\n".join(_render_catalog_row(m, manifests.get(m.dataset_id)) for m in entries)
    return (
        "---\n"
        'description: "Browse and filter every dataset curated into the TimeNet registry."\n'
        "icon: lucide/database\n"
        "tags:\n"
        "  - catalog\n"
        "  - datasets\n"
        "---\n\n"
        "# Datasets\n\n"
        "Every curated dataset in the registry, generated from the `dataset.yaml` cards in\n"
        "`timenet-connectors`. Search and filter below, expand a row for details, or read\n"
        "[`datasets.json`](datasets.json) for programmatic use.\n\n"
        '<div class="ds-catalog">\n'
        f"{_render_filters(domains, tags)}\n"
        '<p class="ds-count" aria-live="polite"></p>\n'
        '<div class="ds-table-wrap">\n'
        '<table class="ds-table">\n'
        "<thead><tr><th></th><th>Dataset</th><th>Name</th><th>Categories</th><th>License</th></tr></thead>\n"
        f"<tbody>\n{rows}\n</tbody>\n"
        "</table>\n"
        "</div>\n"
        "</div>\n"
    )


def _render_catalog_json(entries: list[DatasetMetadata]) -> str:
    """Render the machine-readable dataset catalog.

    Args:
        entries: The dataset metadata, already sorted.

    Returns:
        A JSON array with one object per dataset.
    """
    data = [
        {
            "id": m.dataset_id,
            "name": m.name,
            "description": m.description,
            "version": str(m.dataset_version),
            "license": str(m.license),
            "domains": [str(d) for d in m.domains],
            "tags": list(m.tags),
            "source_url": m.source_url,
        }
        for m in entries
    ]
    return json.dumps(data, indent=2) + "\n"


def generate(registry: Path | None = None) -> list[str]:
    """Generate the datasets catalog page and JSON into ``docs/catalog/``.

    Args:
        registry: An optional local registry directory. When given, rows are enriched with
            schema and counts from each dataset's built ``manifest.json``.

    Returns:
        The sorted dataset ids that were written.
    """
    cards = sorted(CARDS_ROOT.glob("*/*/dataset.yaml"))
    entries = [DatasetMetadata.from_yaml(card) for card in cards]
    entries.sort(key=lambda m: m.dataset_id)
    manifests = {m.dataset_id: _find_manifest(registry, m.dataset_id) for m in entries}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Generated alongside the hand-written catalog/benchmarks.md (both live in docs/catalog/).
    (OUTPUT_DIR / "datasets.md").write_text(_render_index(entries, manifests), encoding="utf-8")
    (OUTPUT_DIR / "datasets.json").write_text(_render_catalog_json(entries), encoding="utf-8")

    return [m.dataset_id for m in entries]


def main() -> None:
    """Parse arguments and generate the datasets catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Local registry directory; enrich rows with schema/counts from built manifests.",
    )
    args = parser.parse_args()
    ids = generate(args.registry)
    print(f"Generated catalog for {len(ids)} datasets: {', '.join(ids)}")


if __name__ == "__main__":
    main()
