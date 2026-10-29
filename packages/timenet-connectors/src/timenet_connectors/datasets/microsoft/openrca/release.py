"""Load the pinned OpenRCA release description."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class TableRelease:
    """One telemetry CSV schema and its temporal/grouping roles."""

    path: str
    kind: str
    timestamp: str
    timestamp_unit: str
    groups: tuple[str, ...]
    columns: dict[str, str]
    text_columns: frozenset[str]
    nullable_strings: frozenset[str]


@dataclass(frozen=True)
class SystemRelease:
    """Paths, input context, and telemetry tables for one evaluated system."""

    key: str
    prefix: str
    display_name: str
    context: str
    component_candidates: tuple[str, ...]
    reason_candidates: tuple[str, ...]
    tables: tuple[TableRelease, ...]
    deployment_workbook: str | None


@dataclass(frozen=True)
class OpenRcaRelease:
    """All trusted facts needed to fetch and interpret the release."""

    upstream_commit: str
    repository: str
    revision: str
    timezone: str
    task_fields: dict[str, tuple[str, ...]]
    systems: tuple[SystemRelease, ...]


def _table(raw: dict[str, Any]) -> TableRelease:
    """Convert one YAML telemetry-table mapping.

    Returns:
        The immutable table definition.
    """
    return TableRelease(
        path=str(raw["path"]),
        kind=str(raw["kind"]),
        timestamp=str(raw["timestamp"]),
        timestamp_unit=str(raw["timestamp_unit"]),
        groups=tuple(cast("list[str]", raw.get("groups", []))),
        columns={str(name): str(dtype) for name, dtype in cast("dict[str, str]", raw["columns"]).items()},
        text_columns=frozenset(cast("list[str]", raw.get("text_columns", []))),
        nullable_strings=frozenset(cast("list[str]", raw.get("nullable_strings", []))),
    )


def _load_release() -> OpenRcaRelease:
    """Load the checked-in release YAML into immutable definitions.

    Returns:
        The complete pinned release definition.
    """
    release_path = Path(__file__).with_name("release.yaml")
    raw = cast("dict[str, Any]", yaml.safe_load(release_path.read_text(encoding="utf-8")))
    systems = tuple(
        SystemRelease(
            key=key,
            prefix=str(config["prefix"]),
            display_name=str(config["display_name"]),
            context=str(config["context"]),
            component_candidates=tuple(cast("list[str]", config["component_candidates"])),
            reason_candidates=tuple(cast("list[str]", config["reason_candidates"])),
            tables=tuple(_table(item) for item in cast("list[dict[str, Any]]", config["tables"])),
            deployment_workbook=(
                None if config.get("deployment_workbook") is None else str(config["deployment_workbook"])
            ),
        )
        for key, config in cast("dict[str, dict[str, Any]]", raw["systems"]).items()
    )
    return OpenRcaRelease(
        upstream_commit=str(raw["upstream_commit"]),
        repository=str(raw["repository"]),
        revision=str(raw["revision"]),
        timezone=str(raw["timezone"]),
        task_fields={
            str(task): tuple(fields) for task, fields in cast("dict[str, list[str]]", raw["task_fields"]).items()
        },
        systems=systems,
    )


RELEASE = _load_release()
"""Pinned release definition shipped with the connector."""
