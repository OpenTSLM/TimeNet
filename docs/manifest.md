---
icon: lucide/file-json
description: "The compiled manifest.json: the single source of truth the SDK reads."
tags:
  - reference
  - manifest
---

# Manifest

The **Dataset Manifest** (`manifest.json`) is the compiled single source of truth that the SDK
reads. It contains the card's metadata, the schema derived from the data, counts, and file
pointers. It is pure data and does no file I/O. The [writer](timef-writer.md) writes it last, so
its presence marks a committed version. The [reader](timef-reader.md) reads it first. The manifest
lives in `timenet.manifest`.

The [`DatasetSchema`](types.md#datasetschema) type already holds flat descriptors. The manifest's
`schema` block is a direct serialization of this type. The manifest has no separate "entry" types
to keep in sync.

The packaged `manifest.schema.json` (JSON Schema draft 2020-12) pins the on-disk shape. This file
is the formal contract for external consumers. It is published as
[`manifest-v1.schema.json`](https://docs.timenet.ai/schemas/manifest-v1.schema.json) and is
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
    schema=schema,              # DatasetSchema (default: empty)
    counts=counts,              # ManifestCounts (default: empty)
    id_encoding={},             # logical id -> "uuid16" (absent => string)
    values_backend="parquet",   # "parquet" (default) or "zarr"
    value_encoding={},          # spec_type -> the encoding its shards carry
    derived_from=None,          # copy-on-write lineage (see below)
    build_env=None,             # environment provenance (see below)
    timef_format_version=1,     # validated against the supported set {1}
)
```

`id_encoding` records which logical ids the [writer](timef-writer.md#id-storage) stored as
`binary(16)`. If an entry is absent, that id is a UTF-8 string. `values_backend` names the
[values backend](timef-writer.md#values-backends) that wrote `files.time_series`. The reader uses
this value to choose the backend. A format-v2 manifest without this key reads as `"parquet"`. Only
a version from a [copy-on-write edit](timef-writer.md#copy-on-write-edits) has `derived_from` set.

`value_encoding` reports the [values encoding](timef-writer.md#values-encoding) used to write each
modality's shards. No code reads this field to make a decision: Parquet already records the applied
encoding in each file's footer. The field exists so that a curator can inspect what a build chose.
The field is empty for a backend that has no such choice.

`build_env` records the environment that produced the version: the interpreter version and every
installed package with its version. `timenet.provenance.build_env` collects this data. Like
`value_encoding`, `build_env` is provenance only, so no code reads it to interpret the data.

Format v2 introduces a backend-neutral schema for the values locator. This schema applies to both
scalar and multidimensional datasets. Multidimensional specs also require format v2. As a result, an
older reader rejects the incompatible values layout. It does not try to read the layout as scalar
data.

If you construct or parse a `Manifest` with an unsupported `timef_format_version`, it raises
`InvalidManifestError`.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls). |
| `from_dict(data)` / `from_json(text)` | Parse, tolerating missing optional blocks. |

`from_dict` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`. `schema` and
`counts` default to empty. The parser drops unmodeled metadata keys. A malformed block raises
`InvalidManifestError`. This error names the offending block.

### Serialization notes

- Units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`). The shared
  registry converts them back.
- A spec carries its data source inline. As a result, the reader does not resolve it against a
  side table.
- Tasks serialize as `{"task_type": ...}`. On read, the reader resolves them against the built-in
  `TASKS` registry. An unknown `task_type` raises `InvalidManifestError`. The annotation
  `value_type` round-trips as a string. The reader uses it to decode values.

---

## `ManifestCounts`

The fields are `samples`, `annotations`, `tasks` (a dict of `task_type -> count`),
`time_series_chunks`, `time_series_index_rows`, and `time_series_specs` (a dict of
`spec_type -> series count`). All fields default to `0` or `{}`.

## `ManifestFiles`

`ManifestFiles` groups file descriptors by kind: `samples`, `annotations`, and
`time_series_index` (required), plus `tasks` and `time_series` (tuples, empty by default). A
reader uses this list. It never uses a directory glob.

Each entry is a `FilePart`. A `FilePart` carries the file's `path` (version-relative), its
`checksum` (with the `sha256:` prefix), and its `size` in bytes. So the path and the digest never
live in separate structures.

`all_files()` returns every descriptor. `all_parts()` returns only the paths.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
