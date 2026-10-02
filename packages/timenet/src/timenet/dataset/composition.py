"""In-memory declarations for objects imported from parent datasets."""

from dataclasses import dataclass

from timenet.types import ObjectRef


@dataclass(frozen=True)
class RecordImport:
    """A record reused from a parent without copying its hierarchy or values."""

    parent_alias: str
    reference: ObjectRef
    inherited_annotation_ids: frozenset[str]
