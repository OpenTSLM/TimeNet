"""Packaged JSON Schema documents for TimeNet's authored artifacts.

The schemas describe the wire formats, not the in-memory dataclasses: ``dataset-card.schema.json``
validates the human-authored card YAML read by
:meth:`~timenet.types.DatasetMetadata.from_yaml`, so an authoring mistake surfaces as one aggregated
message instead of a stack trace from deep inside construction.
"""

from timenet.schemas._registry import DATASET_CARD_SCHEMA as DATASET_CARD_SCHEMA
