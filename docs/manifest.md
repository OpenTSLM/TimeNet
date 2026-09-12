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
    schema=schema,              # DatasetSchema (default: empty)
    counts=counts,              # ManifestCounts (default: empty)
    values_backend="parquet",   # "parquet" (default) or "zarr"
    value_encoding={},          # spec_type -> the encoding its shards carry
    build_env=None,             # environment provenance (see below)
    timef_format_version=1,     # validated against the supported set {1}
)
```

`values_backend` names the [values backend](timef-writer.md#values-backends) that wrote
`files.time_series`. The reader uses this value to select the backend. If the key is absent, the
reader uses `"parquet"`.

`value_encoding` gives the [values encoding](timef-writer.md#values-encoding) that wrote the shards
of each spec type. No code reads this field. Parquet records the applied encoding in the footer of
each file, so the reader does not need it. The field lets a builder see which encoding a build
selected. The field is empty for a backend that has no such choice.

`build_env` records the environment that produced the version. It gives the interpreter version and
every installed package with its version. `timenet.provenance.build_env` collects this data. Like
`value_encoding`, `build_env` is provenance only. No code reads it to interpret the data.

The values locator is backend-neutral. One schema covers both scalar and multidimensional specs. A
multidimensional spec records its shape in `value_shape` and `dimension_names`. It does not need a
separate format version.

A spec records its `nullable` flag in the same way. When an older manifest omits the flag, the current
SDK reads it as `False`. The current SDK can therefore read artifacts written before nullability
support without changing their missing-value behavior.

Nullable artifacts also use `timef_format_version=1`. This does not guarantee that older SDKs can
read newer nullable artifacts correctly. SDKs from before nullability support can ignore the Zarr
validity arrays, which mark present timesteps. Those SDKs can treat missing-value placeholders as
observations.

For nullable artifacts, use an SDK that supports nullability. Format version 1 alone does not show
whether a reader supports the `nullable` flag and its storage representation.

If you construct or parse a `Manifest` with an unsupported `timef_format_version`, it raises
`TimeNetInvalidManifestError`.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls). |
| `from_dict(data)` / `from_json(text)` | Parse, tolerating missing optional blocks. |

`from_dict` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`. `schema` and
`counts` default to empty. The parser drops unmodeled metadata keys. A malformed block raises
`TimeNetInvalidManifestError`. This error names the offending block.

### Serialization notes

- Units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`). The shared
  registry converts them back.
- A spec carries its data source inline. As a result, the reader does not resolve it against a
  side table.
- Tasks serialize as `{"task_type": ...}`. On read, the reader resolves them against the built-in
  `TASKS` registry. An unknown `task_type` raises `TimeNetInvalidManifestError`. The annotation
  `value_type` round-trips as a string. The reader uses it to decode values.

---

## `ManifestCounts`

The fields are `records`, `annotations`, `registered_annotations` (the count of task-referenced
annotations that no record carries; a reader can skip its annotation-recovery scan when this is
zero), `tasks` (a dict of `task_type -> count`), `time_series_chunks`, `time_series_index_rows`,
and `time_series_specs` (a dict of `spec_type -> series count`). All fields default to `0` or `{}`.

## `ManifestFiles`

`ManifestFiles` groups file descriptors by kind: `records`, `annotations`, and
`time_series_index` (required), plus `tasks` and `time_series` (tuples, empty by default). A
reader uses this list. It never uses a directory glob.

Each entry is a `FilePart`. A `FilePart` carries the file's `path` (version-relative), its
`checksum` (with the `sha256:` prefix), and its `size` in bytes. So the path and the digest never
live in separate structures.

`all_files()` returns every descriptor. `all_parts()` returns only the paths.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
