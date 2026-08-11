"""Packaged JSON Schema documents for TimeNet's authored and serialized artifacts.

The schemas describe the wire formats, not the in-memory dataclasses. ``dataset-card.schema.json``
validates the human-authored card YAML (via :meth:`~timenet.types.DatasetMetadata.from_yaml`).
``manifest.schema.json`` is the published contract for the compiled ``manifest.json`` produced by
:meth:`~timenet.manifest.Manifest.to_dict`. A drift test keeps their enum blocks in lockstep with
the Python enums.
"""

from timenet.schemas._registry import (
    DATASET_CARD_SCHEMA as DATASET_CARD_SCHEMA,
    MANIFEST_SCHEMA as MANIFEST_SCHEMA,
)
