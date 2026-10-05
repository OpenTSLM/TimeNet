"""Packaged JSON Schema documents for TimeNet's authored and serialized artifacts.

The schemas describe the wire formats of the in-memory Pydantic models. ``dataset-card.schema.json``
documents the human-authored card YAML validated by :meth:`~timenet.types.DatasetMetadata.from_yaml`.
``manifest.schema.json`` is the published contract for the compiled ``manifest.json`` that
:meth:`~timenet.manifest.Manifest.to_dict` produces. Both files use Pydantic's native JSON Schema.
"""

from timenet.schemas._registry import (
    DATASET_CARD_SCHEMA as DATASET_CARD_SCHEMA,
    MANIFEST_SCHEMA as MANIFEST_SCHEMA,
)
