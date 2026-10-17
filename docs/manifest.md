---
icon: lucide/file-json
description: "The manifest.json entry point, schema summary, file list, and commit marker."
tags:
  - reference
  - manifest
---

# Manifest

The **Dataset Manifest** (`manifest.json`) is the entry point for a TimeF version. It contains the
card metadata, derived schema, counts, and file descriptors. The
[writer](timef-writer.md) writes it last, so its presence marks a committed version. The
[reader](timef-reader.md) reads it before opening `control.duckdb` or Signal values. Manifest types
live in `timenet.manifest`.

The [`DatasetSchema`](types.md#datasetschema) type already holds flat descriptors. The manifest's
`schema` block is a direct serialization of this type. The manifest has no separate "entry" types
to keep in sync.

The packaged `manifest.schema.json` (JSON Schema draft 2020-12) pins the on-disk shape. This file
is the formal contract for external consumers. It is published as
[`manifest.schema.json`](https://docs.timenet.ai/schemas/manifest.schema.json) and is
available in Python as `timenet.schemas.MANIFEST_SCHEMA`. A test validates the output of
`to_dict()` against this schema.

---

## `Manifest`

```python
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles

Manifest(
    dataset_id="physionet/ecg-qa-cot",
    metadata=metadata,          # DatasetMetadata
    files=files,                # ManifestFiles (required)
    dataset_schema=schema,      # Serialized as the `schema` block
    counts=counts,              # ManifestCounts (default: empty)
    build_env=None,             # environment provenance (see below)
    timef_format_version=1,     # required; validated against the supported set {1}
)
```

`files` lists every artifact, keyed by kind. Each kind records the backend that wrote it. See
[`ManifestFiles`](#manifestfiles).

`build_env` records the environment that produced the version. It gives the interpreter version and
every installed package with its version. `timenet.provenance.build_env` collects this data. Like
`files.time_series.encoding`, `build_env` is provenance only. No code reads it to interpret the
data.

The values locator is backend-neutral. One schema covers both scalar and multidimensional specs. A
multidimensional spec records its shape in `value_shape` and `dimension_names`. It does not need a
separate format version.

A spec records its `nullable` flag. Parquet stores nullability in Arrow validity bitmaps. Zarr stores
it in validity arrays next to the values.

Direct construction and parsing with an unsupported `timef_format_version` raise Pydantic's
`ValidationError`.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls). |
| `model_validate(data)` / `model_validate_json(text)` | Parse, tolerating missing optional blocks. |

`model_validate` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`. `schema`
and `counts` default to empty. Unknown fields are rejected. A malformed block raises Pydantic's
`ValidationError` with its native field locations and error details.

### Serialization notes

- Known units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`). An
  unknown unit serializes as `null`. The field remains required.
- Device and origin information belongs to the hierarchy's `Source`, not to `TimeSeriesSpec`.
- Tasks serialize as `{"task_type": ...}`. On read, the reader resolves them against the built-in
  `TASKS` registry. An unknown `task_type` raises Pydantic's `ValidationError`. The annotation
  `value_type` round-trips as a string. The reader uses it to decode values.

---

## `ManifestCounts`

The fields count `records`, `sources`, `signals`, `axes`, reusable `annotation_contents`,
`annotation_occurrences`, and `signal_chunks`. `tasks` maps task type to count, and
`signals_by_spec` maps specification type to Signal count. All fields default to `0` or `{}`.

## `ManifestFiles`

`ManifestFiles` keys the artifacts by kind. Both kinds are required. A reader uses these lists and
never uses a directory glob.

```json
"files": {
  "control": {
    "backend": "duckdb",
    "parts": [
      {"path": "control.duckdb", "checksum": "sha256:...", "size": 1048576}
    ]
  },
  "time_series": {
    "backend": "parquet",
    "encoding": {"ecg": "dictionary"},
    "parts": [
      {"path": "time_series/part-00000000.parquet",
       "checksum": "sha256:...", "size": 4194304}
    ]
  }
}
```

| Kind | Model | Fields |
| --- | --- | --- |
| `control` | `ControlFiles` | `backend` (always `"duckdb"`), `parts` (exactly one file) |
| `time_series` | `TimeSeriesFiles` | `backend` (`"parquet"` or `"zarr"`), `encoding`, `parts` |

`files.time_series.backend` names the [values backend](timef-writer.md#values-settings) that wrote
the values plane. The reader uses this value to select the backend. `TimeFReader.values_backend`
returns it.

`files.time_series.encoding` maps each spec type to the
[values encoding](timef-writer.md#values-settings) of its shards. No code reads this field. Parquet
records the applied encoding in the footer of each file, so the reader does not need it. The field
lets a builder see which encoding a build selected. The key is required, and it is `{}` for Zarr.
Validation rejects a non-empty `encoding` on a Zarr group. The `control` group has no `encoding`
field, and unknown keys fail validation.

Each entry in `parts` is a `FilePart`. A `FilePart` carries the file's `path` (version-relative),
its `checksum` (with the `sha256:` prefix), and its `size` in bytes. So the path and the digest
never live in separate structures.

`all_files()` returns every descriptor, control file first. `all_parts()` returns only the paths.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
