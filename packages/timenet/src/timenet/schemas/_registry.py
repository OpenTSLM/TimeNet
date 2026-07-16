"""Load the packaged JSON Schema documents from this package's data files."""

from importlib import resources
import json
from typing import Any


def _load(name: str) -> dict[str, Any]:
    """Load a packaged JSON Schema document by file name.

    Args:
        name: The schema file name within the ``timenet.schemas`` package.

    Returns:
        The parsed schema as a dict.
    """
    return json.loads((resources.files("timenet.schemas") / name).read_text(encoding="utf-8"))


DATASET_CARD_SCHEMA: dict[str, Any] = _load("dataset-card.schema.json")
MANIFEST_SCHEMA: dict[str, Any] = _load("manifest.schema.json")
