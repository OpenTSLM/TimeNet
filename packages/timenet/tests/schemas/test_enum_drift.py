"""Validate representative enum values in the generated JSON Schemas."""

import jsonschema
import pytest

from timenet.schemas import DATASET_CARD_SCHEMA


@pytest.mark.parametrize(
    "domain",
    [
        "respiratory",
        "motion",
        "environment",
        "energy",
        "transport",
        "observability",
        "audio",
    ],
)
def test_card_schema_accepts_benchmark_domain(domain):
    card = {
        "dataset_id": "demo/thing",
        "dataset_version": "1.0.0",
        "name": "Demo",
        "description": "A demo dataset.",
        "license": "MIT",
        "domains": [domain],
    }
    jsonschema.validate(card, DATASET_CARD_SCHEMA)


def test_card_schema_rejects_unknown_domain():
    card = {
        "dataset_id": "demo/thing",
        "dataset_version": "1.0.0",
        "name": "Demo",
        "description": "A demo dataset.",
        "license": "MIT",
        "domains": ["not-a-domain"],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(card, DATASET_CARD_SCHEMA)
