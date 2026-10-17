"""Read-only HTTP catalog and manifest contract for a TimeF registry."""

from importlib.resources import files
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.registry import BaseRegistry
from timenet.types import DatasetMetadata, Version
from timenet.types.metadata_fields import validate_dataset_id
from timenet.viewer.security import install_security


def create_app(
    registry: BaseRegistry,
    *,
    token: str,
    origin: str,
    initial_manifest: Manifest | None = None,
) -> FastAPI:
    """Create a read-only app for a registry and optional direct-launch manifest.

    Args:
        registry: Registry to browse.
        token: Secret required by all API routes.
        origin: Exact browser origin accepted by the request boundary.
        initial_manifest: Already resolved manifest for a direct dataset launch.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    install_security(app, token=token, origin=origin)

    @app.get("/api/v1/catalog")
    def catalog(
        query: str = Query("", max_length=100),
        domain: str = Query("", max_length=80),
        license: str = Query("", max_length=80),
        offset: int = Query(0, ge=0),
        limit: int = Query(24, ge=1, le=50),
    ) -> dict[str, Any]:
        """Return a filtered metadata page enriched from its pinned manifests.

        Raises:
            HTTPException: If this is a direct launch or the registry has no catalog.
        """
        if initial_manifest is not None:
            raise HTTPException(status_code=404, detail="catalog unavailable in direct dataset mode")
        # TODO: If catalog scale makes this repeated full-list scan expensive,
        # add registry-side filtering and pagination. Manifest reads below are page-bounded.
        try:
            metadatas = registry.list_datasets()
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        domains = sorted({str(value) for item in metadatas for value in item.domains})
        licenses = sorted({str(item.license) for item in metadatas})
        matched = [item for item in metadatas if _matches(item, query, domain, license)]
        items = []
        for item in matched[offset : offset + limit]:
            # TODO: Map stale catalog entries and expected registry read failures
            # to explicit API responses instead of letting them become 500s.
            manifest = registry.get_manifest(item.dataset_id, str(item.dataset_version))
            raw = manifest.to_dict()
            items.append(
                {
                    "metadata": raw["metadata"],
                    "counts": {
                        key: raw["counts"][key]
                        for key in ("records", "sources", "signals", "tasks", "annotation_occurrences")
                    },
                    "task_types": [task["task_type"] for task in raw["schema"]["tasks"]],
                    "signal_types": [spec["spec_type"] for spec in raw["schema"]["time_series_specs"]],
                    "annotation_types": [annotation["key"] for annotation in raw["schema"]["annotations"]],
                }
            )
        return {
            "items": items,
            "total": len(matched),
            "offset": offset,
            "limit": limit,
            "domains": domains,
            "licenses": licenses,
        }

    @app.get("/api/v1/manifest")
    def overview(dataset_id: str, version: str) -> dict[str, Any]:
        """Return the complete manifest for an explicit committed version.

        Raises:
            HTTPException: If the reference is invalid or missing.
        """
        try:
            validate_dataset_id(dataset_id)
            Version.parse(version)
        except (TimeFValidationError, ValueError) as exc:
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

    @app.get("/", response_class=HTMLResponse)
    def shell() -> str:
        """Return the browser shell bundled with the Python package."""
        return files("timenet.viewer").joinpath("static", "index.html").read_text(encoding="utf-8")

    @app.get("/static/{asset}")
    def static_asset(asset: str) -> Response:
        """Return a known packaged frontend asset."""
        media_types = {
            "app.js": "application/javascript",
            "app.css": "text/css",
            "REACT-LICENSE": "text/plain",
        }
        if asset not in media_types:
            return Response(status_code=404)
        content = files("timenet.viewer").joinpath("static", asset).read_bytes()
        return Response(content, media_type=media_types[asset])

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
