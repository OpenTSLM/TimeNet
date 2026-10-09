"""Generate the inspector's request contract for its TypeScript build."""

import json
from pathlib import Path

from timenet.viewer import schemas as app


def main() -> None:
    """Write deterministic OpenAPI components from the actual Pydantic models."""
    schemas = {}
    for name, model in vars(app).items():
        if isinstance(model, type) and issubclass(model, app.StrictQuery) and model is not app.StrictQuery:
            schema = model.model_json_schema(ref_template="#/components/schemas/{model}")  # noqa: RUF027 - Pydantic template
            schemas.update(schema.pop("$defs", {}))
            schemas[name] = schema
    # Integer microseconds accept decimal strings on the wire to avoid browser rounding.
    for schema in schemas.values():
        for name, field in schema.get("properties", {}).items():
            if name in {"start_us", "end_us", "gap_threshold_us"}:
                field.setdefault("anyOf", []).append({"type": "string", "pattern": "^-?[0-9]+$"})
    document = {
        "openapi": "3.1.0",
        "info": {"title": "TimeNet inspector", "version": "1"},
        "paths": {},
        "components": {"schemas": schemas},
    }
    path = Path(__file__).resolve().parents[2] / "frontend/schema.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
