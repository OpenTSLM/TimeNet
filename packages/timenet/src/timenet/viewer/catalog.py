"""Registry-wide filtering with bounded catalog pages and stable facets."""

from collections.abc import Iterator

from timenet.manifest import Manifest
from timenet.reader.inspection.limits import MAX_TEXT
from timenet.registry import BaseRegistry, LocalRegistry
from timenet.registry.local import CatalogEntry
from timenet.viewer.responses import CatalogItem, CatalogPage
from timenet.viewer.schemas import CatalogQuery


_PAGE_SIZE = 50


def _entries(registry: BaseRegistry) -> Iterator[CatalogEntry]:
    if isinstance(registry, LocalRegistry):
        yield from registry.iter_catalog()
    else:
        for metadata in sorted(registry.list_datasets(), key=lambda item: item.dataset_id):
            yield CatalogEntry(
                metadata.dataset_id, registry.get_manifest(metadata.dataset_id, str(metadata.dataset_version))
            )


def catalog_page(registry: BaseRegistry, query: CatalogQuery) -> CatalogPage:
    """Filter across the registry before paginating; retain only one result page.

    Returns:
        Matching datasets, global facets, and a continuation cursor.
    """
    items: list[CatalogItem] = []
    issues = []
    domains: set[str] = set()
    tasks: set[str] = set()
    licenses: set[str] = set()
    cursor = None
    last = None
    count = 0
    for entry in _entries(registry):
        item = None
        if entry.error is None:
            if entry.manifest is None:
                continue
            item = _catalog_item(entry.manifest)
            domains.update(item["domains"])
            tasks.update(item["tasks"])
            licenses.add(item["license"])
            metadata = entry.manifest.metadata
            text = f"{metadata.dataset_id} {metadata.name} {metadata.description}".casefold()
            if not _matches(item, query, text):
                continue
            count += 1
        if entry.dataset_id <= query.after:
            continue
        if len(items) + len(issues) >= _PAGE_SIZE:
            cursor = last
            continue
        if entry.error is not None:
            issues.append({"dataset_id": entry.dataset_id, "message": entry.error})
        elif item is not None:
            items.append(item)
        last = entry.dataset_id
    return {
        "api_version": 1,
        "items": items,
        "issues": issues,
        "next_cursor": cursor,
        "total_matches": count,
        "facets": {"domains": sorted(domains), "tasks": sorted(tasks), "licenses": sorted(licenses)},
    }


def _matches(item: CatalogItem, query: CatalogQuery, text: str) -> bool:
    return all(
        (
            query.query.casefold() in text,
            not query.domain or query.domain in item["domains"],
            not query.task or query.task in item["tasks"],
            not query.license or query.license == item["license"],
        )
    )


def _catalog_item(manifest: Manifest) -> CatalogItem:
    metadata = manifest.metadata
    return {
        "dataset_id": metadata.dataset_id,
        "version": str(metadata.dataset_version),
        "name": metadata.name[:MAX_TEXT],
        "description": metadata.description[:MAX_TEXT],
        "description_truncated": len(metadata.description) > MAX_TEXT,
        "domains": [domain.value for domain in metadata.domains],
        "license": str(metadata.license),
        "access": metadata.access.value,
        "tasks": [str(task.task_type) for task in manifest.dataset_schema.tasks],
    }
