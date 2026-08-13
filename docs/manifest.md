---
icon: lucide/file-json
description: "The compiled manifest.json: the single source of truth the SDK reads."
tags:
  - reference
  - manifest
---

# Manifest

The **Dataset Manifest** (`manifest.json`) is the compiled single source of truth the SDK reads: the
card's metadata plus the schema derived from the data, counts, and file pointers. It is pure data with
no file I/O. The [writer](timef-writer.md) writes it last (its presence marks a committed version) and
the [reader](timef-reader.md) reads it first. Lives in `timenet.manifest`.

Because [`DatasetSchema`](types.md#datasetschema) already holds flat descriptors, the manifest's
`schema` block is a direct serialization of it. There are no separate "entry" types to keep in sync.

The on-disk shape is pinned by the packaged `manifest.schema.json` (JSON Schema draft 2020-12), the
formal contract for external consumers. It is published as
[`manifest-v1.schema.json`](https://docs.timenet.ai/schemas/manifest-v1.schema.json) and available in
Python as `timenet.schemas.MANIFEST_SCHEMA`; a test validates `to_dict()` output against it.

---

## `Manifest`

```python
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles

Manifest(
    dataset_id="physionet/ecg-qa-cot",
    metadata=metadata,          # DatasetMetadata
    files=files,                # ManifestFiles (required)
    schema=schema,              # DatasetSchema (default: empty)
    counts=counts,              # ManifestCounts (default: empty)
    checksums={},               # relpath -> "sha256:..." (default: empty)
    id_encoding={},             # logical id -> "uuid16" (absent => string)
    values_backend="parquet",   # "parquet" (default) or "zarr"
    value_encoding={},          # spec_type -> the encoding its shards carry
    derived_from=None,          # copy-on-write lineage (see below)
    timef_format_version=1,     # validated against the supported set {1}
)
```

`id_encoding` records which logical ids the [writer](timef-writer.md#id-storage) stored as `binary(16)`;
an absent entry means that id is a UTF-8 string. `values_backend` names the
[values backend](timef-writer.md#values-backends) that wrote `files.time_series`; the reader dispatches
on it, and a format-v2 manifest without the key reads as `"parquet"`. `derived_from` is set only on a version
produced by a [copy-on-write edit](timef-writer.md#copy-on-write-edits).

`value_encoding` reports the [values encoding](timef-writer.md#values-encoding) each modality's shards
were written with. Nothing dispatches on it: Parquet records the applied encoding in every file's footer,
so it exists for a curator inspecting what a build chose. It is empty for a backend with no such choice.

The backend-neutral values locator schema introduced in format v2 applies to both scalar and
multidimensional datasets. Multidimensional specs also require v2 so older readers reject their
incompatible values layout explicitly rather than attempting to interpret it as scalar data.

Constructing a `Manifest` (or parsing one) with an unsupported `timef_format_version` raises
`InvalidManifestError`.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls). |
| `from_dict(data)` / `from_json(text)` | Parse, tolerating missing optional blocks. |

`from_dict` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`; `schema` and
`counts` default to empty. Unmodeled metadata keys are dropped. Malformed blocks raise
`InvalidManifestError` naming the offending block.

### Serialization notes

- Units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`) and back via the
  shared registry.
- A spec carries its data source inline, so nothing is resolved against a side table on read.
- Tasks serialize as `{"task_type": ...}` and resolve on read against the built-in `TASKS` registry
  (an unknown `task_type` raises `InvalidManifestError`); annotation `value_type` round-trips as a
  string and is used by the reader to decode values.

---

## `ManifestCounts`

`samples`, `annotations`, `tasks` (dict `task_type -> count`), `time_series_chunks`,
`time_series_index_rows`, `time_series_specs` (dict `spec_type -> series count`). All default to
`0` / `{}`.

## `ManifestFiles`

Relative paths within the version directory: `samples`, `annotations`, `time_series_index` (required),
plus `tasks` and `time_series` (tuples, default empty). Readers use this list, never a directory glob.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
