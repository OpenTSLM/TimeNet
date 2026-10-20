"""Read-only HTTP catalog, manifest, and task browsing for a TimeF registry."""

from importlib.resources import files
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from timenet.errors import TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.registry import BaseRegistry
from timenet.types import DatasetMetadata, Version
from timenet.types.metadata_fields import validate_dataset_id
from timenet.viewer.records import RecordLimits, install_record_routes
from timenet.viewer.security import install_security
from timenet.viewer.tasks import install_task_routes


_QUERY_MAX_LENGTH = 100
_FILTER_MAX_LENGTH = 80
_CATALOG_PAGE_SIZE = 24
_CATALOG_MAX_PAGE_SIZE = 50

# Packaged frontend assets and their media types. An explicit list, rather than a static files
# mount, keeps every other path under /static a 404.
_STATIC_MEDIA_TYPES = {
    "app.js": "application/javascript",
    "app.css": "text/css",
    "REACT-LICENSE": "text/plain",
}


class CatalogCounts(BaseModel):
    """Object counts shown on a catalog card."""

    records: int
    sources: int
    signals: int
    tasks: dict[str, int]
    annotation_occurrences: int


class CatalogItem(BaseModel):
    """One dataset's metadata and the manifest facts shown on its catalog card."""

    metadata: DatasetMetadata
    counts: CatalogCounts
    task_types: list[str]
    signal_types: list[str]
    annotation_types: list[str]

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "CatalogItem":
        """Summarize a committed manifest for the catalog.

        Returns:
            The catalog card facts.
        """
        counts = manifest.counts
        schema = manifest.dataset_schema
        return cls(
            metadata=manifest.metadata,
            counts=CatalogCounts(
                records=counts.records,
                sources=counts.sources,
                signals=counts.signals,
                tasks=dict(counts.tasks),
                annotation_occurrences=counts.annotation_occurrences,
            ),
            task_types=[str(task.task_type) for task in schema.tasks],
            signal_types=[spec.spec_type for spec in schema.time_series_specs],
            annotation_types=[annotation.key for annotation in schema.annotations],
        )


class CatalogPage(BaseModel):
    """One filtered page of the catalog, with the filter values available across all datasets."""

    items: list[CatalogItem]
    total: int
    offset: int
    limit: int
    domains: list[str]
    licenses: list[str]


def create_app(
    registry: BaseRegistry,
    *,
    token: str,
    origin: str,
    initial_manifest: Manifest | None = None,
    record_limits: RecordLimits | None = None,
) -> FastAPI:
    """Create a read-only app for a registry and optional direct-launch manifest.

    Args:
        registry: Registry to browse.
        token: Secret required by all API routes.
        origin: Exact browser origin accepted by the request boundary.
        initial_manifest: Already resolved manifest for a direct dataset launch.
        record_limits: Optional record and signal operation caps.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    install_security(app, token=token, origin=origin)

    @app.get("/api/v1/catalog")
    def catalog(
        query: str = Query("", max_length=_QUERY_MAX_LENGTH),
        domain: str = Query("", max_length=_FILTER_MAX_LENGTH),
        license: str = Query("", max_length=_FILTER_MAX_LENGTH),
        offset: int = Query(0, ge=0),
        limit: int = Query(_CATALOG_PAGE_SIZE, ge=1, le=_CATALOG_MAX_PAGE_SIZE),
    ) -> CatalogPage:
        """Return a filtered metadata page enriched from its pinned manifests.

        Each request lists the registry again, so datasets built while the viewer runs appear.
        Only the manifests on the returned page are read.

        Raises:
            HTTPException: If this is a direct launch or the registry has no catalog.
        """
        if initial_manifest is not None:
            raise HTTPException(status_code=404, detail="catalog unavailable in direct dataset mode")
        try:
            metadatas = registry.list_datasets()
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        matched = [item for item in metadatas if _matches(item, query, domain, license)]
        return CatalogPage(
            items=[
                CatalogItem.from_manifest(registry.get_manifest(item.dataset_id, str(item.dataset_version)))
                for item in matched[offset : offset + limit]
            ],
            total=len(matched),
            offset=offset,
            limit=limit,
            domains=sorted({str(value) for item in metadatas for value in item.domains}),
            licenses=sorted({str(item.license) for item in metadatas}),
        )

    @app.get("/api/v1/manifest")
    def overview(dataset_id: str, version: str) -> dict[str, Any]:
        """Return the complete manifest for an explicit committed version, in its JSON wire form.

        Raises:
            HTTPException: If the reference is invalid or missing.
        """
        try:
            validate_dataset_id(dataset_id)
            # Parsing accepts spellings such as "01.0.0"; use the canonical form from here on.
            version = str(Version.parse(version))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if initial_manifest is not None and (
            dataset_id == initial_manifest.dataset_id and version == str(initial_manifest.metadata.dataset_version)
        ):
            return initial_manifest.to_dict()
        if initial_manifest is not None:
            raise HTTPException(status_code=404, detail="dataset version not available in this viewer")
        try:
            return registry.get_manifest(dataset_id, version).to_dict()
        except TimeNetDatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    install_task_routes(app, registry, initial_manifest)
    install_record_routes(app, registry, initial_manifest, record_limits or RecordLimits())

    @app.get("/", response_class=HTMLResponse)
    def shell() -> str:
        """Return the browser shell bundled with the Python package."""
        return files("timenet.viewer").joinpath("static", "index.html").read_text(encoding="utf-8")

    @app.get("/static/{asset}")
    def static_asset(asset: str) -> Response:
        """Return a known packaged frontend asset."""
        if asset not in _STATIC_MEDIA_TYPES:
            return Response(status_code=404)
        content = files("timenet.viewer").joinpath("static", asset).read_bytes()
        return Response(content, media_type=_STATIC_MEDIA_TYPES[asset])

    return app


def _matches(metadata: DatasetMetadata, query: str, domain: str, license: str) -> bool:
    """Match catalog filters against registry metadata.

    Returns:
        Whether the metadata matches every supplied filter.
    """
    if domain and domain not in {str(value) for value in metadata.domains}:
        return False
    if license and license != str(metadata.license):
        return False
    haystack = " ".join((metadata.dataset_id, metadata.name, metadata.description, *metadata.tags)).casefold()
    return query.casefold() in haystack
