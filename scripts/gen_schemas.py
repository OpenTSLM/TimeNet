"""Generate TimeNet's published JSON Schemas from its Pydantic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from timenet.manifest import Manifest
from timenet.types import DatasetMetadata


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "packages" / "timenet" / "src" / "timenet" / "schemas"
_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def generated_schemas() -> dict[Path, str]:
    """Return every generated schema keyed by its checked-in path.

    Returns:
        Deterministic JSON text for the card and manifest schemas.
    """
    documents = {
        SCHEMA_DIR / "dataset-card.schema.json": {
            "$schema": _DRAFT,
            "$id": "https://docs.timenet.ai/schemas/dataset-card-v1.schema.json",
            **DatasetMetadata.model_json_schema(),
        },
        SCHEMA_DIR / "manifest.schema.json": {
            "$schema": _DRAFT,
            "$id": "https://docs.timenet.ai/schemas/manifest.schema.json",
            **Manifest.model_json_schema(),
        },
    }
    return {path: json.dumps(value, indent=2, ensure_ascii=False) + "\n" for path, value in documents.items()}


def main() -> int:
    """Generate schemas or check that the checked-in copies are current.

    Returns:
        Zero on success and one when ``--check`` finds stale files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of writing when a schema is stale")
    args = parser.parse_args()
    stale: list[Path] = []
    for path, content in generated_schemas().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(ROOT))
        else:
            path.write_text(content, encoding="utf-8")
    if stale:
        for path in stale:
            print(f"stale generated schema: {path}", file=sys.stderr)
        print("run 'make schemas' to regenerate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
