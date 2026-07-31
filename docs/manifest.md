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

The on-disk shape is pinned by `manifest-v1.schema.json` (JSON Schema draft 2020-12), the formal
contract for external consumers. It ships in the `timenet` package (`timenet.schemas.MANIFEST_SCHEMA`);
a test validates `to_dict()` output against it. Registries will serve it alongside their datasets.

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
    id_encoding={},             # logical id -> "uuid16" (absent => stored as string)
    values_backend="parquet",   # values-plane storage: "parquet" or "zarr" (absent => parquet)
    derived_from=None,          # copy-on-write lineage, e.g. {"dataset_version": "1.0.0", "op": ...}
    timef_format_version=1,     # validated against the supported set {1}
)
```

`id_encoding` records which logical ids the [writer](timef-writer.md#id-storage) stored as `binary(16)`;
an absent entry means that id is a UTF-8 string. `values_backend` names the
[values backend](timef-writer.md#values-backends) that wrote `files.time_series`; the reader dispatches
on it, and a manifest without the key reads as `"parquet"`. `derived_from` is set only on a version
produced by a [copy-on-write edit](timef-writer.md#copy-on-write-edits).

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
- A spec references its data source by `data_source_type`; the full record lives once in
  `schema.data_sources` and is resolved back on read.
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
