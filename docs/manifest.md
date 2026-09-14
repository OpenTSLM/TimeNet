---
icon: lucide/file-json
description: "The compiled manifest.json: the contract a consumer reads before it opens anything."
tags:
  - reference
  - manifest
---

# Manifest

`manifest.json` is the contract for a dataset version. It carries the dataset's identity, a
checksummed descriptor for every file, the counts the build recorded, and which values backend
wrote the values plane. It is pure data and does no file I/O: the [writer](timef-writer.md) writes
it last, so its presence marks a committed version, and a consumer reads it first. It lives in
`timenet.manifest`.

---

## `Manifest`

```python
from timenet.manifest import Manifest, ManifestCounts, ManifestFiles

Manifest(
    dataset_id="demo/bedside-monitor",
    metadata=metadata,          # DatasetMetadata
    files=files,                # ManifestFiles (required)
    counts=counts,              # ManifestCounts (default: empty)
    values_backend="parquet",   # "parquet" (default) or "zarr"
    value_encoding={},          # spec_type -> the encoding its shards carry
    build_env={},               # environment provenance
    timef_format_version=1,     # validated against the supported set {1}
)
```

`values_backend` names the backend that wrote `files.time_series`. The control database says the
same thing per artifact in its `values_artifacts` table, and that is what the reader dispatches on.
The manifest copy is for the consumer that has only the manifest: it can see that a version needs
the `zarr` extra, or that a Zarr store's thousands of small files are one logical artifact, before
downloading anything. An absent value means `parquet`.

`value_encoding` gives the [values encoding](timef-format.md#choosing-the-values-encoding) that
wrote each spec type's shards. No code reads it. Parquet records the applied encoding in every
file's footer, so a reader decodes a shard without it; the field lets a builder see what a build
chose. It is empty for a backend that makes no such choice.

`build_env` records the environment that produced the version: the interpreter version and every
installed package with its version. `timenet.provenance.build_env` collects it. Like
`value_encoding`, it is provenance only.

Constructing or parsing a `Manifest` with an unsupported `timef_format_version` raises
`TimeNetInvalidManifestError`.

### Codec

| Method | Purpose |
| --- | --- |
| `to_dict()` / `to_json()` | Canonical serialization (all keys present, explicit nulls). |
| `from_dict(data)` / `from_json(text)` | Parse, tolerating missing optional blocks. |

`from_dict` requires `timef_format_version`, `dataset_id`, `metadata`, and `files`. `schema` and
`counts` default to empty. The parser drops unmodeled metadata keys. A malformed block raises
`TimeNetInvalidManifestError`, naming the block at fault.

### Serialization notes

- Units serialize to their pint names (`"hertz"`, `"millivolt"`, `"dimensionless"`). The shared
  registry converts them back.
- A spec carries its data source inline, so the reader does not resolve it against a side table.

---

## `ManifestFiles`

`ManifestFiles` describes every artifact of a version. A reader uses this list; it never globs the
directory.

| Field | What it holds |
| --- | --- |
| `control_db` | The DuckDB control-plane database. One `FilePart`, not a tuple: the database pages itself and never shards. |
| `time_series` | The values plane's parts, one `FilePart` each. |

Each entry is a `FilePart` carrying the file's `path` (version-relative), its `checksum` (with the
`sha256:` prefix), and its `size` in bytes, so a path and its digest never live in separate
structures. `all_files()` returns every descriptor; `all_parts()` returns only the paths, which is
what a download iterates.

A version carries either `control_db` or the Parquet control tables it replaces (`records`,
`annotations`, `time_series_index` and `tasks`), never both, so those four tuples are empty
whenever `control_db` names a database.

## `ManifestCounts`

What the writer records: `records`, `annotations` (distinct payloads), `registered_annotations`
(attachments of those payloads), `time_series_chunks`, and `time_series_index_rows`. Two more fields
exist and stay empty: `tasks` and `time_series_specs`, both keyed breakdowns the control database
answers directly. All fields default to `0` or `{}`.

---

See the [API reference for `timenet.manifest`](api/manifest.md) for the full symbol listing.
