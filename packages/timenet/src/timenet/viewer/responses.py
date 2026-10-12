"""Viewer lifecycle responses used by the generated client contract."""

from typing import Literal, NotRequired

from typing_extensions import TypedDict


class Session(TypedDict):
    """The active viewer mode and optional dataset identity."""

    api_version: int
    registry: str
    mode: NotRequired[Literal["registry", "dataset"]]
    dataset_id: NotRequired[str]
    limits: NotRequired[dict[str, int]]


class CatalogItem(TypedDict):
    """Validated catalog metadata."""

    dataset_id: str
    version: str
    name: str
    description: str
    description_truncated: bool
    domains: list[str]
    license: str
    access: str
    tasks: list[str]


class CatalogIssue(TypedDict):
    """An incompatible committed dataset."""

    dataset_id: str
    message: str


class CatalogFacets(TypedDict):
    """Filter choices from every compatible dataset in the registry."""

    domains: list[str]
    tasks: list[str]
    licenses: list[str]


class CatalogPage(TypedDict):
    """A bounded catalog page, including invalid entries."""

    api_version: int
    items: list[CatalogItem]
    issues: list[CatalogIssue]
    next_cursor: str | None
    total_matches: int
    facets: CatalogFacets


class ModeResult(TypedDict):
    """A completed dataset selection transition."""

    api_version: int
    mode: Literal["registry", "dataset"]
